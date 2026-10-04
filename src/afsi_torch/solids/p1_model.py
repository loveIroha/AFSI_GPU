"""Composable P1 mechanics: PK1 material, boundary forces and validity checks.

Callbacks use single-element tensors and are vectorized on the device. Forces
are integrated nodal forces in CGS; the tangent is their derivative dforce/dx.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from ..p1 import prepare_p1
from ..mechanics import determinant3


@dataclass(frozen=True)
class BoundaryForce:
    connectivity: torch.Tensor
    local_force: object        # (local_nodes, fields_tuple, scalar_time_tensor) -> (local_nodes,3)
    fields: tuple=()          # first axis indexes boundary elements
    validity: object=None    # optional pure tensor predicate(x) -> bool tensor

    def force(self,x,time):
        local=torch.vmap(lambda nodes,*fields:self.local_force(nodes,fields,time))(
            x[self.connectivity],*self.fields)
        return torch.zeros_like(x).index_add(0,self.connectivity.reshape(-1),local.reshape(-1,3))


class P1Solid:
    """Mesh-independent affine tetrahedra, with replaceable material/boundaries."""
    def __init__(self,mesh,stress,*,cell_fields=(),boundaries=(),degree=5,validity_checks=()):
        if mesh.cells.shape[1]!=4 or mesh.X.ndim!=2 or mesh.X.shape[1]!=3:
            raise ValueError('P1Solid requires 3D affine tetrahedra')
        if not callable(stress):
            raise TypeError('stress must be a pure tensor PK1 callback')
        self.mesh,self.stress=mesh,stress
        self.cell_fields,self.boundaries,self.validity_checks=tuple(cell_fields),tuple(boundaries),tuple(validity_checks)
        self.geometry=prepare_p1(mesh.X,mesh.cells,degree)
        self.gradients=self.geometry.gradients[:,0]
        self.volumes=self.geometry.weights.sum(1)
        for b in self.boundaries:
            if not isinstance(b,BoundaryForce) or not callable(b.local_force):
                raise TypeError('boundaries must contain BoundaryForce objects')
            if b.connectivity.ndim!=2 or b.connectivity.dtype!=torch.int64 or b.connectivity.device!=mesh.X.device:
                raise ValueError('boundary connectivity must be a device-local int64 matrix')
            if b.connectivity.numel() and ((b.connectivity<0).any() or (b.connectivity>=len(mesh.X)).any()):
                raise ValueError('boundary connectivity references an absent node')
        for fields,count in ((self.cell_fields,len(mesh.cells)),*((b.fields,len(b.connectivity)) for b in self.boundaries)):
            for f in fields:
                if f.ndim<1 or f.shape[0]!=count or f.device!=mesh.X.device or f.dtype!=mesh.X.dtype:
                    raise ValueError('cell/boundary fields must match element count, device and floating dtype')
        self.validate(mesh.X)

    @staticmethod
    def time_tensor(x,time):
        if not isfinite(time) or time<0:
            raise ValueError('finite nonnegative solid load time required')
        return x.new_tensor(time)

    def element_gradient(self,x):
        return torch.einsum('eai,eaJ->eiJ',x[self.mesh.cells],self.gradients)

    def geometry_state(self,x):
        F=self.element_gradient(x)
        J=determinant3(F)
        flags=[torch.isfinite(J).all()&(J>0).all()]
        flags += [check(x).all() for check in self.validity_checks]
        flags += [b.validity(x).all() for b in self.boundaries if b.validity is not None]
        return F,torch.stack(flags)

    def check_coordinates(self,x):
        if x.shape!=self.mesh.X.shape or x.device!=self.mesh.X.device or x.dtype!=self.mesh.X.dtype:
            raise ValueError('coordinates must match the prepared solid mesh')

    def validate(self,x):
        self.check_coordinates(x)
        if not self.geometry_state(x)[-1].all():
            raise ValueError('invalid P1 solid deformation or user validity check')

    def force_from_geometry(self,x,F,time):
        P=torch.vmap(lambda f,*fields:self.stress(f,fields,time))(F,*self.cell_fields)
        if P.shape!=F.shape:
            raise ValueError('PK1 callback must return a 3x3 tensor per cell')
        local=-self.volumes[:,None,None]*torch.einsum('eiJ,eaJ->eai',P,self.gradients)
        force=torch.zeros_like(x).index_add(0,self.mesh.cells.reshape(-1),local.reshape(-1,3))
        for boundary in self.boundaries:
            force=force+boundary.force(x,time)
        return force

    def force(self,x,time):
        return self.force_from_geometry(x,self.element_gradient(x),self.time_tensor(x,time))

    def tangent_factory(self,chunk_size=2048):
        from .csr import P1Tangent
        return P1Tangent(self,chunk_size)

    def execution_factory(self):
        return P1Execution(self)

    def diagnostics(self,x):
        J=determinant3(self.element_gradient(x))
        return dict(wall_volume_cm3=(J*self.volumes).sum().item(),minimum_detF=J.min().item(),
            maximum_detF=J.max().item(),max_total_displacement_cm=torch.linalg.vector_norm(x-self.mesh.X,dim=-1).max().item())


class P1Execution:
    """Compiled material/force kernels with owned, version-checked geometry."""
    def __init__(self,model):
        from ..mac.execution import tensor_kernel
        self.model=model
        self._geometry_kernel=tensor_kernel(model.geometry_state,model.mesh.X.device)
        self._force_kernel=tensor_kernel(model.force_from_geometry,model.mesh.X.device)
        self._time=model.mesh.X.new_zeros(())
        self._cached_x=self._cached_version=self._cached_geometry=None

    def validate(self,x):
        self.model.check_coordinates(x)
        version=None if torch.is_inference(x) else x._version
        if version is not None and x is self._cached_x and version==self._cached_version:
            return
        geometry=self._geometry_kernel(x)
        if not geometry[-1].all():
            self._cached_x=self._cached_geometry=None
            raise ValueError('invalid P1 solid deformation or user validity check')
        self.remember_geometry(x,geometry)

    def remember_geometry(self,x,geometry):
        self._cached_x=x
        self._cached_version=None if torch.is_inference(x) else x._version
        self._cached_geometry=geometry

    def _set_time(self,time):
        if not isfinite(time) or time<0:
            raise ValueError('finite nonnegative solid load time required')
        self._time.fill_(time)

    def force(self,x,time):
        self.validate(x)
        self._set_time(time)
        return self._force_kernel(x,self._cached_geometry[0],self._time)

    def force_with_geometry(self,x,time):
        self.model.check_coordinates(x)
        geometry=self.checked_geometry(x)
        if geometry is None:
            geometry=self._geometry_kernel(x)
        self._set_time(time)
        return self._force_kernel(x,geometry[0],self._time),geometry

    def checked_geometry(self,x):
        if (not torch.is_inference(x) and x is self._cached_x and
                x._version==self._cached_version):
            return self._cached_geometry
        return None
