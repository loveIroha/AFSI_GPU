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

    def residual_restrict(self,p,rhs,out,spacing):
        out.copy_(F.avg_pool3d((rhs-negative_laplacian(p,spacing))[None,None],2,2)[0,0])

    def prolong_add(self, coarse, fine):
        fine.add_(F.interpolate(coarse[None,None], size=fine.shape,
                                mode='trilinear', align_corners=False)[0,0])


class MGWorkspace:
    def __init__(self, solver, *, optimized=False, graphs=False):
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
        self.optimized=optimized
        self.spacings = solver.spacings
        self.diagonals = solver.diagonals
        self.coarse_inverse = solver.coarse_inverse
        self.p = [torch.empty_like(d) for d in solver.diagonals]
        self.ping = [torch.empty_like(d) for d in solver.diagonals]
        self.rhs = [torch.empty_like(d) for d in solver.diagonals]
        self.residuals = [] if optimized else [torch.empty_like(d) for d in solver.diagonals[:-1]]
        self.fine_residual=self.residuals[0] if self.residuals else torch.empty_like(sample)
        self.allocated_bytes = sum(t.numel()*t.element_size()
            for group in (self.p, self.ping, self.rhs, self.residuals) for t in group)
        if not self.residuals:
            self.allocated_bytes+=self.fine_residual.numel()*self.fine_residual.element_size()
        self.graphs={}
        if graphs:
            if not sample.is_cuda:
                raise ValueError('pressure CUDA graphs require a CUDA device')
            if self.options.check_every>16:
                raise ValueError('pressure graph check_every must be <=16')
            self._capture()

    def initialize(self, p, rhs):
        self.p[0].copy_(p)
        self.rhs[0].copy_(rhs)

    def smooth(self, level):
        p,out=self.p[level],self.ping[level]
        for _ in range(self.options.smooth):
            self.kernels.smooth(p,self.rhs[level],self.diagonals[level],out,self.spacings[level])
            p,out=out,p
        # Canonical addresses are stable across graph capture/replay, including
        # odd Jacobi counts and a one-cycle final partial block.
        if p is not self.p[level]:
            self.p[level].copy_(p)
        self.p[level].sub_(self.p[level].mean())

    def cycle(self, level=0):
        if level == len(self.p)-1:
            torch.mv(self.coarse_inverse, self.rhs[level].reshape(-1),
                     out=self.p[level].reshape(-1))
            self.p[level].sub_(self.p[level].mean())
            return self.p[level]
        self.smooth(level)
        if self.optimized:
            self.kernels.residual_restrict(self.p[level],self.rhs[level],self.rhs[level+1],self.spacings[level])
        else:
            self.kernels.residual(self.p[level],self.rhs[level],self.residuals[level],self.spacings[level])
            self.kernels.restrict(self.residuals[level],self.rhs[level+1])
        self.rhs[level+1].sub_(self.rhs[level+1].mean())
        self.p[level+1].zero_()
        correction = self.cycle(level+1)
        self.kernels.prolong_add(correction, self.p[level])
        self.smooth(level)
        return self.p[level]

    def _block(self,count):
        for _ in range(count):
            self.cycle()

    def _capture(self):
        with torch.cuda.device(self.p[0].device):
            self.p[0].zero_(); self.rhs[0].zero_()
            stream=torch.cuda.Stream(device=self.p[0].device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._block(self.options.check_every)
            torch.cuda.current_stream().wait_stream(stream)
            for count in range(1,self.options.check_every+1):
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    self._block(count)
                self.graphs[count]=graph
            torch.cuda.current_stream().wait_stream(stream)

    def advance(self,count):
        if self.graphs:
            self.graphs[count].replay()
        else:
            self._block(count)
        return self.p[0]

    def residual_norm(self):
        self.kernels.residual(self.p[0],self.rhs[0],self.fine_residual,self.spacings[0])
        return torch.linalg.vector_norm(self.fine_residual)
