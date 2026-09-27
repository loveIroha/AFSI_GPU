"""One-time PyTorch COO assembly, coalescing and CSR storage on the mesh device.

Constant coefficients and an affine, fixed background mesh are required.
Vector mass/stiffness share one scalar matrix across the three components.
Convection remains an element operation because its coefficients change.
"""
import torch


def _assemble(rows, cols, local, shape):
    nr, nc = rows.shape[1], cols.shape[1]
    indices = torch.stack((rows[:, :, None].expand(-1, nr, nc).reshape(-1),
                           cols[:, None, :].expand(-1, nr, nc).reshape(-1)))
    values = local.reshape(1, nr, nc).expand(len(rows), -1, -1).reshape(-1)
    return torch.sparse_coo_tensor(indices, values, shape, device=local.device,
                                  dtype=local.dtype, check_invariants=True).coalesce().to_sparse_csr()


def _apply(matrix, field):
    return torch.sparse.mm(matrix, field[:, None]).squeeze(1) if field.ndim == 1 else torch.sparse.mm(matrix, field)


class CSRFluidOperators:
    """Drop-in fixed operator actions, preserving the quadrature reference API."""
    def __init__(self, reference):
        self.reference = reference
        self.mesh = reference.mesh
        op, mesh = reference, self.mesh
        nv, np = len(mesh.velocity_coordinates), len(mesh.pressure_coordinates)
        self.matrices = {}
        for prefix, cells, N, dN, size in (
                ('velocity', mesh.velocity_cells, op.N, op.dN, nv),
                ('pressure', mesh.pressure_cells, op.Q, op.dQ, np)):
            for name, local in (
                    ('mass', torch.einsum('qa,qb,q->ab', N, N, op.weights)),
                    ('stiffness', torch.einsum('qaj,qbj,q->ab', dN, dN, op.weights))):
                self.matrices[prefix+'_'+name] = _assemble(cells, cells, local, (size, size))
        vector_cells = (3*mesh.velocity_cells[..., None]+
                        torch.arange(3, device=op.N.device)).reshape(len(mesh.velocity_cells), -1)
        # G=(v,grad p) and D=(q,div u) are assembled independently: no assumption G=-D.T.
        G = torch.einsum('qa,qbj,q->ajb', op.N, op.dQ, op.weights)
        D = torch.einsum('qa,qbj,q->abj', op.Q, op.dN, op.weights)
        self.matrices['gradient'] = _assemble(vector_cells, mesh.pressure_cells, G, (3*nv, np))
        self.matrices['divergence'] = _assemble(mesh.pressure_cells, vector_cells, D, (np, 3*nv))

    def __getattr__(self, name):
        return getattr(self.reference, name)

    def velocity_mass(self, u):
        self._field(u, vector=True)
        return _apply(self.matrices['velocity_mass'], u)

    def velocity_stiffness(self, u):
        self._field(u, vector=True)
        return _apply(self.matrices['velocity_stiffness'], u)

    def pressure_mass(self, p):
        self._field(p, pressure=True)
        return _apply(self.matrices['pressure_mass'], p)

    def pressure_stiffness(self, p):
        self._field(p, pressure=True)
        return _apply(self.matrices['pressure_stiffness'], p)

    def gradient(self, p):
        self._field(p, pressure=True)
        return _apply(self.matrices['gradient'], p).reshape(-1, 3)

    def divergence(self, u):
        self._field(u, vector=True)
        return _apply(self.matrices['divergence'], u.reshape(-1))

    def density_load(self, density):
        return self.velocity_mass(density)

    def tentative_matrix(self, mass_scale, stiffness_scale):
        M, K = (self.matrices['velocity_'+key] for key in ('mass', 'stiffness'))
        # Both assemblies retain the same full cell-pair sparsity, including zeros.
        return torch.sparse_csr_tensor(M.crow_indices(), M.col_indices(),
            mass_scale*M.values()+stiffness_scale*K.values(), size=M.shape,
            device=M.device, dtype=M.dtype, check_invariants=True)

