"""Same quadrature IB equations, compact separable stencils and fused execution."""
from dataclasses import dataclass
import torch
from .transfer import FETransfer
from .execution import tensor_kernel
from .mass_solver import MassSolver
from ..ib import peskin4


@dataclass(frozen=True)
class CompactStencil:
    base: torch.Tensor  # (3,P,3), lower neighbor for each staggered lattice
    phi: torch.Tensor   # (3,P,3,4), four 1D weights per axis

    @property
    def storage_bytes(self):
        return self.base.numel()*self.base.element_size()+self.phi.numel()*self.phi.element_size()


def _prepare(points,origins,spacing,offsets):
    scaled=(points[None]-origins[:,None])/spacing
    base=torch.floor(scaled-1).long()
    nodes=base[...,None]+offsets
    phi=peskin4(scaled[...,None]-nodes.to(points.dtype))
    return base,phi


def _evaluate(value,N,cells):
    return torch.einsum('qa,eai->eqi',N,value[cells])


def _weighted(value,N,cells,W):
    return (_evaluate(value,N,cells)*W[...,None]).reshape(-1,3)


def _assemble(point_velocity,N,cells,W,template):
    local=torch.einsum('qa,eq,eqi->eai',N,W,point_velocity.reshape(*W.shape,3))
    return torch.zeros_like(template).index_add(0,cells.reshape(-1),local.reshape(-1,3))


class CompactFETransfer(FETransfer):
    def __init__(self,*args,mass_backend='pcg',**kwargs):
        if mass_backend not in ('pcg','graph'):
            raise ValueError('mass backend must be pcg or graph')
        super().__init__(*args,**kwargs)
        self.mass_backend=mass_backend
        device=self.geometry.weights.device
        self.execution_backend='triton+compile' if device.type=='cuda' else 'buffered-cpu'
        self._prepare_kernel=tensor_kernel(_prepare,device)
        self._evaluate_kernel=tensor_kernel(_evaluate,device)
        self._weighted_kernel=tensor_kernel(_weighted,device)
        self._assemble_kernel=tensor_kernel(_assemble,device)
        self._support_kernel=tensor_kernel(self._support_flags,device)
        self._finite_kernel=tensor_kernel(lambda a,b,c:torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(c).all(),device)
        self._nodal_finite_kernel=tensor_kernel(lambda value:torch.isfinite(value).all(),device)
        cast=self.geometry.weights.new_tensor
        self.origins=cast([self.grid.face_origin(c) for c in range(3)])
        self.spacing=cast(self.grid.spacing)
        self.origin=cast(self.grid.origin)
        self.limits=cast(self.grid.shape)-2
        if mass_backend=='graph':
            from .mass_graph import GraphMassSolver
            self.mass_solver=GraphMassSolver(self.mass,self.diagonal,self.options)
        else:
            self.mass_solver=MassSolver(self.mass,self.diagonal,self.options)

    def _support_flags(self,points):
        scaled=(points-self.origin)/self.spacing
        return torch.isfinite(points).all() & (scaled>=2).all() & (scaled<self.limits).all()

    def check_support(self,points):
        if not self._support_kernel(points):
            raise ValueError('MAC IB support reaches a wall or is nonfinite; enlarge/refine the fluid box')

    def _nodal(self,value):
        g=self.geometry
        if (value.shape!=(g.node_count,3) or value.device!=g.weights.device or
                value.dtype!=g.weights.dtype or not self._nodal_finite_kernel(value)):
            raise ValueError('invalid FE nodal field')

    def evaluate(self,value):
        return self._evaluate_kernel(value,self.geometry.values,self.geometry.cells)

    @torch.no_grad()
    def prepare(self,x):
        points=self.interaction_points(x)
        self.check_support(points)
        return self.from_points(points)

    def from_points(self,points):
        # Called only after support/finite-value acceptance, including cached
        # next-step preparation in the optimized coupled driver.
        # The same two-cell wall margin guarantees every staggered 4-point
        # stencil is complete. No clipping or renormalization is introduced.
        base,phi=self._prepare_kernel(points,self.origins,self.spacing,self.axis_offsets)
        return CompactStencil(base.contiguous(),phi.contiguous())

    def solve_mass(self,rhs,initial):
        return self.mass_solver.solve(rhs,initial)

    def weighted_force(self,value):
        g=self.geometry
        return self._weighted_kernel(value,g.values,g.cells,g.weights)

    def assemble_velocity(self,value):
        g=self.geometry
        return self._assemble_kernel(value,g.values,g.cells,g.weights,self.diagonal)

    def _expanded_component(self,stencil,c,start,stop):
        base=stencil.base[c,start:stop]
        phi=stencil.phi[c,start:stop]
        nodes=base[...,None]+self.axis_offsets
        shape=self.grid.face_shape(c)
        ids=((nodes[:,0,:,None,None]*shape[1]+nodes[:,1,None,:,None])*shape[2]+nodes[:,2,None,None,:])
        weights=(phi[:,0,:,None,None]*phi[:,1,None,:,None])*phi[:,2,None,None,:]
        return ids.reshape(-1,64),weights.reshape(-1,64)

    def spread_grid(self,force_q,stencil):
        if force_q.is_cuda:
            from ._triton_ib import spread
            return spread(self.grid,force_q.contiguous(),stencil)
        fields=self.grid.zeros(device=force_q.device,dtype=force_q.dtype)
        for start in range(0,len(force_q),4096):
            stop=min(start+4096,len(force_q))
            for c,out in enumerate(fields):
                ids,w=self._expanded_component(stencil,c,start,stop)
                out.reshape(-1).index_add_(0,ids.reshape(-1),(w*force_q[start:stop,c,None]/self.grid.volume).reshape(-1))
        return fields

    def gather_grid(self,velocity,stencil):
        if velocity[0].is_cuda:
            from ._triton_ib import gather
            return gather(self.grid,tuple(u.contiguous() for u in velocity),stencil)
        count=stencil.base.shape[1]
        out=stencil.phi.new_empty((count,3))
        for start in range(0,count,4096):
            stop=min(start+4096,count)
            for c,u in enumerate(velocity):
                ids,w=self._expanded_component(stencil,c,start,stop)
                out[start:stop,c]=(u.reshape(-1)[ids]*w).sum(-1)
        return out

    @torch.no_grad()
    def spread(self,nodal_force,stencil):
        self._nodal(nodal_force)
        coefficient,info=self.solve_mass(nodal_force,self._force_coefficient if self.warm_start else None)
        if self.warm_start:
            self._force_coefficient=coefficient
        return self.spread_grid(self.weighted_force(coefficient),stencil),info

    @torch.no_grad()
    def interpolate(self,velocity,stencil):
        self.grid.check_velocity(velocity)
        g=self.geometry
        if any(u.device!=g.weights.device or u.dtype!=g.weights.dtype for u in velocity) or not self._finite_kernel(*velocity):
            raise ValueError('invalid MAC interpolation field')
        rhs=self.assemble_velocity(self.gather_grid(velocity,stencil))
        result,info=self.solve_mass(rhs,self._velocity_coefficient if self.warm_start else None)
        if self.warm_start:
            self._velocity_coefficient=result
        return result,info
