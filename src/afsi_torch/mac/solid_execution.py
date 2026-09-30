"""Forward-only compiled execution of the existing P2 LV weak form.

The reference LVSolid remains available for derivatives, checkpoint validation
and diagnostics. Caches are local to a driver and checked against tensor versions.
"""
import torch
from .. import solid, boundary as bd
from ..geometry import cavity_volume
from ..mechanics import determinant3
from .execution import tensor_kernel


class SolidExecution:
    def __init__(self,model):
        self.model=model
        f=model.fields
        self.axes=torch.stack((f.fiber,f.sheet,f.normal),-1)
        p=model.parameters
        self.strain_weights=model.mesh.X.new_tensor([[p.bf,p.bfs,p.bfs],[p.bfs,p.bt,p.bt],[p.bfs,p.bt,p.bt]])
        self.identity=torch.eye(3,device=model.mesh.X.device,dtype=model.mesh.X.dtype)
        self.loads=model.mesh.X.new_empty(2)
        self._geometry_kernel=tensor_kernel(self._geometry,model.mesh.X.device)
        self._force_kernel=tensor_kernel(self._force,model.mesh.X.device)
        self._cached_x=None
        self._cached_version=None
        self._cached_geometry=None

    def _geometry(self,x):
        m=self.model
        F=solid.deformation_gradient(x,m.geometry)
        J=determinant3(F)
        endo=bd.area_vectors(x,m.endo)
        base=bd.area_vectors(x,m.base)
        volume=cavity_volume(x,m.cavity)
        def surface_ok(area,surface):
            scale=torch.linalg.vector_norm(surface.reference_area_vectors,dim=-1)[:,None]
            return torch.isfinite(area).all() & (torch.linalg.vector_norm(area,dim=-1)>100*torch.finfo(x.dtype).eps*scale).all()
        flags=torch.stack((torch.isfinite(J).all() & (J>0).all(),
            surface_ok(endo,m.endo),surface_ok(base,m.base),torch.isfinite(volume) & (volume>0)))
        return F,endo,flags

    @torch.no_grad()
    def validate(self,x):
        m=self.model
        if x.shape!=m.mesh.X.shape or x.device!=m.mesh.X.device or x.dtype!=m.mesh.X.dtype:
            raise ValueError('x must match the prepared mesh shape, device and dtype')
        # Inference tensors have no version counter; do not reuse their cache.
        version=None if torch.is_inference(x) else x._version
        if version is not None and x is self._cached_x and version==self._cached_version:
            return
        geometry=self._geometry_kernel(x)
        flags=geometry[-1].tolist()
        if not all(flags):
            self._cached_x=None
            self._cached_geometry=None
            raise ValueError('invalid LV deformation: det(F), surface or cavity volume')
        self._cached_x,self._cached_version,self._cached_geometry=x,version,geometry

    def _stress(self,F,loads):
        m=self.model
        E=.5*(F.transpose(-1,-2)@F-self.identity)
        local=self.axes.transpose(-1,-2)@E@self.axes
        Q=(self.strain_weights*local.square()).sum((-2,-1))
        S=m.parameters.C*torch.exp(Q)[...,None,None]*(self.axes@(self.strain_weights*local)@self.axes.transpose(-1,-2))
        # J F^{-T} is the cofactor. Explicit 3x3 cross products avoid a batched
        # LU/inverse call and its status synchronization, with identical PK1.
        cof=torch.stack((torch.linalg.cross(F[...,1,:],F[...,2,:]),
                         torch.linalg.cross(F[...,2,:],F[...,0,:]),
                         torch.linalg.cross(F[...,0,:],F[...,1,:])),-2)
        J=(F[...,0,:]*cof[...,0,:]).sum(-1)
        P=F@S+2*m.parameters.kappa*(J-1)[...,None,None]*cof
        fiber=m.fields.fiber
        Ff=(F@fiber[...,None]).squeeze(-1)
        P=P+loads[1]*Ff[..., :,None]*fiber[...,None,:]
        return P

    def _assemble(self,x,P,endo_area,loads):
        m=self.model
        internal=solid.assemble_pk1(P,m.geometry)
        surface=m.endo
        traction=-loads[0]*endo_area
        pressure=bd._scatter(torch.einsum('q,qa,bqi->bai',surface.quadrature_weights,surface.values,traction),surface)
        return internal+pressure+bd.spring_force(x,m.base,m.beta)

    def _force(self,x,F,endo_area,loads):
        return self._assemble(x,self._stress(F,loads),endo_area,loads)

    @torch.no_grad()
    def force(self,x,time):
        self.validate(x)
        self._set_loads(time)
        F,endo,_=self._cached_geometry
        return self._force_kernel(x,F,endo,self.loads)

    def _set_loads(self,time):
        pressure,tension=self.model.loads.at(time)
        # Tensor loads prevent recompilation for every Python time/float value.
        self.loads[0].fill_(pressure)
        self.loads[1].fill_(tension)

    @torch.no_grad()
    def force_with_geometry(self,x,time):
        # Pending geometry is never cached before all coupled acceptance checks
        # pass. This exposes existing validity flags for one combined host read.
        m=self.model
        if x.shape!=m.mesh.X.shape or x.device!=m.mesh.X.device or x.dtype!=m.mesh.X.dtype:
            raise ValueError('x must match the prepared mesh shape, device and dtype')
        geometry=self._geometry_kernel(x)
        self._set_loads(time)
        F,endo,_=geometry
        return self._force_kernel(x,F,endo,self.loads),geometry

    def remember_geometry(self,x,geometry):
        self._cached_x=x
        self._cached_version=None if torch.is_inference(x) else x._version
        self._cached_geometry=geometry
