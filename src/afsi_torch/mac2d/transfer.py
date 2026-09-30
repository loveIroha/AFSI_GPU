"""Adjoint quadrature FE transfer, with odd velocity ghosts at channel walls.

Identical reflected weights in gather and spread account for wall reactions.
No clipping or renormalization. Horizontal inlet/outlet support is rejected.
"""
from dataclasses import dataclass
import torch
from ..ib import peskin4
from ..fluid.solvers import SolverOptions
from ..mac.mass_solver import MassSolver
from ..mac.mass_graph import GraphMassSolver
from ..mac.execution import tensor_kernel


@dataclass(frozen=True)
class Stencil:
    indices: tuple
    weights: tuple


class TriangleTransfer:
    def __init__(self,grid,geometry,*,mass_backend='graph',warm_start=True,fused=True,optimized=False):
        if mass_backend not in ('pcg','graph'):
            raise ValueError('mass backend must be pcg or graph')
        self.grid,self.geometry=grid,geometry
        self.warm_start=warm_start
        self.mass_backend=mass_backend
        self.optimized=optimized
        self.force_initial,self.velocity_initial=None,None
        g=geometry
        local=torch.einsum('eq,qa,qb->eab',g.weights,g.values,g.values)
        row=g.cells[:,:,None].expand_as(local).reshape(-1)
        col=g.cells[:,None,:].expand_as(local).reshape(-1)
        self.mass=torch.sparse_coo_tensor(torch.stack((row,col)),local.reshape(-1),
            (g.node_count,g.node_count),device=g.weights.device,dtype=g.weights.dtype,check_invariants=True).coalesce().to_sparse_csr()
        diagonal=g.weights.new_zeros(g.node_count).index_add(0,g.cells.reshape(-1),local.diagonal(dim1=-2,dim2=-1).reshape(-1))
        self.diagonal=diagonal[:,None].expand(-1,2)
        options=SolverOptions(rtol=1e-12,atol=1e-13,max_iterations=500,check_every=4)
        self.solver=(GraphMassSolver if mass_backend=='graph' else MassSolver)(self.mass,self.diagonal,options)
        self.templates=grid.zeros(device=g.weights.device,dtype=g.weights.dtype)
        self.offsets=torch.cartesian_prod(torch.arange(4,device=g.weights.device),torch.arange(4,device=g.weights.device))
        if fused:
            for name in ('evaluate','_prepare','_spread','_gather','_assemble'):
                setattr(self,name,tensor_kernel(getattr(self,name),g.weights.device))
            if optimized:
                self.support_valid=tensor_kernel(self.support_valid,g.weights.device)

    def reset_warm_start(self):
        self.force_initial,self.velocity_initial=None,None

    def evaluate(self,value):
        return torch.einsum('qa,eai->eqi',self.geometry.values,value[self.geometry.cells])

    def _prepare(self,points):
        h=points.new_tensor(self.grid.spacing)
        indices,weights=[],[]
        ny=self.grid.shape[1]
        for c in range(2):
            origin=h*points.new_tensor([0 if c==0 else .5,0 if c==1 else .5])
            scaled=(points-origin)/h
            nodes=torch.floor(scaled-1).long()[:,None]+self.offsets[None]
            w=peskin4(scaled[:,None]-nodes.to(points.dtype)).prod(-1)
            j=nodes[...,1]
            if c==0:
                outside=(j<0)|(j>=ny)
                reflected=torch.where(j<0,-j-1,torch.where(j>=ny,2*ny-j-1,j))
            else:
                outside=(j<0)|(j>ny)
                reflected=torch.where(j<0,-j,torch.where(j>ny,2*ny-j,j))
            w=torch.where(outside,-w,w)
            ids=nodes[...,0]*self.grid.face_shape(c)[1]+reflected
            indices.append(ids)
            weights.append(w)
        return tuple(indices),tuple(weights)

    def prepare(self,x):
        points=self.evaluate(x).reshape(-1,2)
        hx,hy=self.grid.spacing
        # Reflection is valid only at the horizontal walls. Solid motion beyond
        # this halo or into inlet/outlet is a failed step, never an index clamp.
        if self.optimized:
            if not self.support_valid(points).item():
                raise ValueError('valve IB points leave the channel support region')
        elif (not torch.isfinite(points).all() or (points[:,0]<2*hx).any() or
            (points[:,0]>self.grid.lengths[0]-2*hx).any() or
            (points[:,1]<-hy).any() or (points[:,1]>self.grid.lengths[1]+hy).any()):
            raise ValueError('valve IB points leave the channel support region')
        return self.from_points(points)

    def support_valid(self,points):
        hx,hy=self.grid.spacing
        return (torch.isfinite(points).all() & (points[:,0]>=2*hx).all() &
            (points[:,0]<=self.grid.lengths[0]-2*hx).all() &
            (points[:,1]>=-hy).all() & (points[:,1]<=self.grid.lengths[1]+hy).all())

    def from_points(self,points):
        # Caller must validate support; used after the combined acceptance check.
        ids,w=self._prepare(points)
        return Stencil(ids,w)

    def _spread(self,coefficient,indices,weights):
        force_q=(self.evaluate(coefficient)*self.geometry.weights[...,None]).reshape(-1,2)
        return tuple(torch.zeros_like(self.templates[c]).reshape(-1).index_add(0,indices[c].reshape(-1),
            (force_q[:,c,None]*weights[c]/self.grid.volume).reshape(-1)).reshape_as(self.templates[c]) for c in range(2))

    def spread(self,force,stencil):
        coefficient,info=self.solver.solve(force,self.force_initial if self.warm_start else None)
        if self.warm_start:
            self.force_initial=coefficient
        return self._spread(coefficient,stencil.indices,stencil.weights),info

    def _gather(self,velocity,indices,weights):
        return torch.stack([(u.reshape(-1)[ids]*w).sum(-1) for u,ids,w in zip(velocity,indices,weights)],-1)

    def _assemble(self,value):
        g=self.geometry
        local=torch.einsum('qa,eq,eqi->eai',g.values,g.weights,value.reshape(*g.weights.shape,2))
        return torch.zeros_like(self.diagonal).index_add(0,g.cells.reshape(-1),local.reshape(-1,2))

    def interpolate(self,velocity,stencil):
        rhs=self._assemble(self._gather(velocity,stencil.indices,stencil.weights))
        result,info=self.solver.solve(rhs,self.velocity_initial if self.warm_start else None)
        if self.warm_start:
            self.velocity_initial=result
        return result,info
