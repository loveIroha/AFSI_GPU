"""Reusable V-cycle storage, with tensor kernels for CPU verification.

CUDA selects Triton kernels lazily. The CPU implementation verifies the same
workspace algorithm but is not a fused-performance backend. Each solver owns
its storage and must not be used for concurrent solves.
"""
import torch
import torch.nn.functional as F
from .grid import negative_laplacian


class TensorKernels:
    name = 'buffered-cpu'

    def smooth(self, p, rhs, diagonal, out, spacing):
        out.copy_(p+(2/3)*(rhs-negative_laplacian(p, spacing))/diagonal)

    def residual(self, p, rhs, out, spacing):
        out.copy_(rhs-negative_laplacian(p, spacing))

    def restrict(self, fine, out):
        out.copy_(F.avg_pool3d(fine[None,None], 2, 2)[0,0])

    def prolong_add(self, coarse, fine):
        fine.add_(F.interpolate(coarse[None,None], size=fine.shape,
                                mode='trilinear', align_corners=False)[0,0])


class MGWorkspace:
    def __init__(self, solver):
        sample = solver.diagonals[0]
        if sample.device.type == 'cuda':
            try:
                from ._triton_mg import TritonKernels
            except ImportError as exc:
                raise ImportError('Fused CUDA multigrid requires Triton. Install the '
                                  'Triton dependency matching your Linux CUDA PyTorch '
                                  'installation, or use pressure_backend="torch".') from exc
            self.kernels = TritonKernels()
        elif sample.device.type == 'cpu':
            self.kernels = TensorKernels()
        else:
            raise ValueError('fused multigrid supports CPU verification or CUDA')
        self.options = solver.options
        self.spacings = solver.spacings
        self.diagonals = solver.diagonals
        self.coarse_inverse = solver.coarse_inverse
        self.p = [torch.empty_like(d) for d in solver.diagonals]
        self.ping = [torch.empty_like(d) for d in solver.diagonals]
        self.rhs = [torch.empty_like(d) for d in solver.diagonals]
        self.residuals = [torch.empty_like(d) for d in solver.diagonals[:-1]]
        self.allocated_bytes = sum(t.numel()*t.element_size()
            for group in (self.p, self.ping, self.rhs, self.residuals) for t in group)

    def initialize(self, p, rhs):
        self.p[0].copy_(p)
        self.rhs[0].copy_(rhs)

    def smooth(self, level):
        for _ in range(self.options.smooth):
            self.kernels.smooth(self.p[level], self.rhs[level], self.diagonals[level],
                                self.ping[level], self.spacings[level])
            self.p[level], self.ping[level] = self.ping[level], self.p[level]
        self.p[level].sub_(self.p[level].mean())

    def cycle(self, level=0):
        if level == len(self.p)-1:
            torch.mv(self.coarse_inverse, self.rhs[level].reshape(-1),
                     out=self.p[level].reshape(-1))
            self.p[level].sub_(self.p[level].mean())
            return self.p[level]
        self.smooth(level)
        self.kernels.residual(self.p[level], self.rhs[level], self.residuals[level],
                              self.spacings[level])
        self.kernels.restrict(self.residuals[level], self.rhs[level+1])
        self.rhs[level+1].sub_(self.rhs[level+1].mean())
        self.p[level+1].zero_()
        correction = self.cycle(level+1)
        self.kernels.prolong_add(correction, self.p[level])
        self.smooth(level)
        return self.p[level]
