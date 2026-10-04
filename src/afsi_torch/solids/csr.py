"""Reusable CSR pattern and element-local derivatives for affine P1 models."""
import torch


class CSRPattern:
    def __init__(self,reference,connectivity):
        self.size=reference.numel()
        self.connectivity=tuple(connectivity)
        keys,self.offsets=[],[0]
        for cells in self.connectivity:
            if cells.ndim!=2 or cells.dtype!=torch.int64 or cells.device!=reference.device:
                raise ValueError('connectivity must be a device-local int64 matrix')
            if cells.numel() and ((cells<0).any() or (cells>=len(reference)).any()):
                raise ValueError('connectivity references an absent solid node')
            dofs=(3*cells[...,None]+torch.arange(3,device=cells.device)).reshape(len(cells),3*cells.shape[1])
            keys.append((dofs[:,:,None]*self.size+dofs[:,None,:]).reshape(-1))
            self.offsets.append(self.offsets[-1]+keys[-1].numel())
        keys,self.inverse=torch.unique(torch.cat(keys),sorted=True,return_inverse=True)
        row,self.col=keys//self.size,keys%self.size
        self.diagonal_indices=torch.where(row==self.col)[0]
        self.diagonal_rows=row[self.diagonal_indices]
        self.crow=torch.cat((row.new_zeros(1),torch.bincount(row,minlength=self.size).cumsum(0)))

    def add(self,values,which,start,blocks):
        width=3*self.connectivity[which].shape[1]
        begin=self.offsets[which]+start*width*width
        values.index_add_(0,self.inverse[begin:begin+blocks.numel()],blocks.reshape(-1))

    def matrix(self,values):
        if not torch.isfinite(values).all():
            raise FloatingPointError('nonfinite assembled solid-force tangent')
        return torch.sparse_csr_tensor(self.crow,self.col,values,size=(self.size,self.size),
            device=values.device,dtype=values.dtype,check_invariants=False)

    def diagonal(self,tangent):
        return tangent.values().new_zeros(self.size).index_add_(0,self.diagonal_rows,
            tangent.values()[self.diagonal_indices])


class P1Tangent:
    """Differentiate small PK1/face kernels, then assemble global CSR on device."""
    def __init__(self,model,chunk_size=2048):
        if type(chunk_size) is not int or chunk_size<1:
            raise ValueError('positive tangent chunk_size required')
        self.model,self.chunk_size=model,chunk_size
        self.pattern=CSRPattern(model.mesh.X,(model.mesh.cells,*(b.connectivity for b in model.boundaries)))
        from ..mac.execution import tensor_kernel
        self._volume_kernel=tensor_kernel(self._volume,model.mesh.X.device)
        self._boundary_kernels=tuple(tensor_kernel(self._boundary_kernel(b),model.mesh.X.device) for b in model.boundaries)

    def _volume(self,F,gradients,volumes,time,fields):
        stress=lambda f,*values:self.model.stress(f,values,time)
        D=torch.vmap(torch.func.jacrev(stress,argnums=0))(F,*fields)
        return -torch.einsum('e,eaJ,eiJkL,ebL->eaibk',volumes,gradients,D,gradients)

    @staticmethod
    def _boundary_kernel(boundary):
        def kernel(nodes,time,fields):
            force=lambda x,*values:boundary.local_force(x,values,time)
            return torch.vmap(torch.func.jacrev(force,argnums=0))(nodes,*fields)
        return kernel

    def diagonal(self,tangent):
        return self.pattern.diagonal(tangent)

    @torch.no_grad()
    def assemble(self,x,time):
        m=self.model
        values=x.new_zeros(len(self.pattern.col))
        F=m.element_gradient(x)
        time=m.time_tensor(x,time)
        for start in range(0,len(F),self.chunk_size):
            selection=slice(start,start+self.chunk_size)
            local=self._volume_kernel(F[selection],m.gradients[selection],m.volumes[selection],time,
                tuple(f[selection] for f in m.cell_fields))
            self.pattern.add(values,0,start,local)
        for which,(boundary,kernel) in enumerate(zip(m.boundaries,self._boundary_kernels),1):
            for start in range(0,len(boundary.connectivity),self.chunk_size):
                selection=slice(start,start+self.chunk_size)
                local=kernel(x[boundary.connectivity[selection]],time,tuple(f[selection] for f in boundary.fields))
                self.pattern.add(values,which,start,local)
        return self.pattern.matrix(values)
