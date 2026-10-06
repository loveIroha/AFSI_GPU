"""Assembled P1 H-O nodal-force Jacobian, including follower and basal terms.

Autodifferentiate small element/face kernels; sum them into a cached global
CSR pattern on the tensor device. No dense global matrix or CPU assembly.
"""
import torch
from .holzapfel_ogden import ho_pk1


class HOTangentAssembler:
    def __init__(self, model, chunk_size=2048):
        self.model, self.chunk_size = model, chunk_size
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('positive tangent chunk_size required')
        self.size = 3*len(model.mesh.X)
        self.connectivity = (model.mesh.cells, model.endo.faces, model.base.faces)
        keys, self.offsets = [], [0]
        for cells in self.connectivity:
            dofs = (3*cells[..., None]+torch.arange(3, device=cells.device)).reshape(len(cells), -1)
            keys.append((dofs[:, :, None]*self.size+dofs[:, None, :]).reshape(-1))
            self.offsets.append(self.offsets[-1]+keys[-1].numel())
        keys, self.inverse = torch.unique(torch.cat(keys), sorted=True, return_inverse=True)
        row, self.col = keys//self.size, keys % self.size
        self.diagonal_indices = torch.where(row==self.col)[0]
        self.diagonal_rows = row[self.diagonal_indices]
        self.block_indices = torch.where(row//3==self.col//3)[0]
        block_row,block_col = row[self.block_indices],self.col[self.block_indices]
        self.block_offsets = (block_row//3)*9+(block_row%3)*3+block_col%3
        self.crow = torch.cat((row.new_zeros(1), torch.bincount(row, minlength=self.size).cumsum(0)))
        # d(basal nodal force)/dx = beta*integral Na Nb*(rhat*rhat^T-I).
        base, radial = model.base, model.radial
        q = radial.new_zeros((*radial.shape[:2], 3, 3))
        q[..., :2, :2] = radial[..., :, None]*radial[..., None, :]
        q -= torch.eye(3, device=q.device, dtype=q.dtype)
        self.basal = model.beta*torch.einsum('fq,qa,qb,fqik->faibk',
            base.reference_weights, base.values, base.values, q)
        from .mac.execution import tensor_kernel
        self._volume_kernel = tensor_kernel(self._volume, model.mesh.X.device)
        self._follower_kernel = tensor_kernel(self._follower, model.mesh.X.device)

    def diagonal(self, tangent):
        """Extract the diagonal with cached indices, avoiding a full row array."""
        return tangent.values().new_zeros(self.size).index_add(0,self.diagonal_rows,
            tangent.values()[self.diagonal_indices])

    def nodal_diagonal_blocks(self, tangent):
        """3x3 force-Jacobian blocks, with the CSR index selection cached."""
        return tangent.values().new_zeros(3*self.size).index_add(0,self.block_offsets,
            tangent.values()[self.block_indices]).reshape(-1,3,3)

    def _volume(self, F, fiber, sheet, gradients, volumes, tension):
        def stress(F, fiber, sheet):
            return ho_pk1(F, fiber, sheet, self.model.parameters, tension)
        D = torch.vmap(torch.func.jacrev(stress, argnums=0))(F, fiber, sheet)
        return -torch.einsum('e,eaJ,eiJkL,ebL->eaibk', volumes, gradients, D, gradients)

    def _follower(self, nodes, pressure):
        surface = self.model.endo
        def force(nodes):
            tangent = torch.einsum('ai,qaj->qij', nodes, surface.derivatives)
            area = torch.linalg.cross(tangent[..., 0], tangent[..., 1])
            return -pressure*torch.einsum('q,qa,qi->ai', surface.quadrature_weights, surface.values, area)
        return torch.vmap(torch.func.jacrev(force))(nodes)

    @torch.no_grad()
    def assemble(self, x, time):
        m = self.model
        pressure, tension = m.loads.at(time)
        F = m.element_gradient(x)
        values = x.new_zeros(len(self.col))
        def add(which, start, local):
            width = 3*self.connectivity[which].shape[1]
            begin = self.offsets[which]+start*width*width
            ids = self.inverse[begin:begin+local.numel()]
            values.index_add_(0, ids, local.reshape(-1))
        pressure, tension = x.new_tensor(pressure), x.new_tensor(tension)
        for start in range(0, len(F), self.chunk_size):
            stop = start+self.chunk_size
            local = self._volume_kernel(F[start:stop], m.mesh.fiber[start:stop], m.mesh.sheet[start:stop],
                                        m.gradients[start:stop],m.volumes[start:stop],tension)
            add(0, start, local)
        surface = m.endo
        for start in range(0, len(surface.faces), self.chunk_size):
            add(1, start, self._follower_kernel(x[surface.faces[start:start+self.chunk_size]], pressure))
        add(2, 0, self.basal)
        if not torch.isfinite(values).all():
            raise FloatingPointError('nonfinite H-O force tangent')
        return torch.sparse_csr_tensor(self.crow, self.col, values,
            size=(self.size, self.size), device=x.device, dtype=x.dtype, check_invariants=False)
