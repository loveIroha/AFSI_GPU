"""Geometric multigrid with zero physical-face pressure, no nullspace fixing."""
from math import isfinite, prod
import torch
import torch.nn.functional as nnf
from .multigrid import MGOptions
from .open_boundary import negative_laplacian, pressure_diagonal
from .execution import tensor_kernel


class DirichletMultigrid:
    def __init__(self, grid, *, device='cpu', dtype=torch.float64, options=None, backend='torch'):
        if backend not in ('torch', 'fused', 'workspace', 'graph'):
            raise ValueError('invalid Dirichlet multigrid execution')
        self.options = options or MGOptions()
        self.backend = 'torch-dirichlet' if backend == 'torch' else 'compiled-dirichlet'
        self.shapes, self.spacings, self.diagonals = [], [], []
        like = torch.empty((), device=device, dtype=dtype)
        shape, spacing = grid.shape, grid.spacing
        while True:
            self.shapes.append(shape); self.spacings.append(spacing)
            self.diagonals.append(pressure_diagonal(shape, spacing, like))
            if min(shape) <= 4 or any(n % 2 for n in shape):
                break
            shape = tuple(n//2 for n in shape)
            spacing = tuple(2*h for h in spacing)
        if prod(shape) > 512:
            raise ValueError('Dirichlet pressure grid must coarsen to <=512 cells')
        count = prod(shape)
        # Setup only: exact coarse operator, including wall-centre distance.
        eye = torch.eye(count, device=device, dtype=dtype)
        matrix = torch.vmap(lambda p: negative_laplacian(p.reshape(shape), spacing).reshape(-1))(eye).T
        self.coarse_inverse = torch.linalg.inv(matrix)
        self.smooth = []
        for spacing, diagonal in zip(self.spacings, self.diagonals):
            def smooth(p, rhs, h=spacing, d=diagonal):
                for _ in range(self.options.smooth):
                    p = p+(2/3)*(rhs-negative_laplacian(p, h))/d
                return p
            self.smooth.append(smooth if backend == 'torch' else tensor_kernel(smooth, device))
        self._norm = lambda rhs, p: torch.linalg.vector_norm(rhs-negative_laplacian(p, grid.spacing))
        if backend != 'torch':
            self._norm = tensor_kernel(self._norm, device)
        self.p, self.rhs = like.new_zeros(grid.shape), like.new_zeros(grid.shape)
        self.graph = None
        self.graph_requested = backend == 'graph' and like.is_cuda

    @staticmethod
    def prolong(correction, shape):
        # Odd extension respects zero pressure at the physical wall, rather
        # than the constant padding of plain trilinear interpolation.
        padded = correction
        for axis in range(3):
            from .grid import slab
            padded = torch.cat((-padded[slab(axis, 0, 1)], padded,
                                -padded[slab(axis, -1, None)]), axis)
        fine = nnf.interpolate(padded[None, None], scale_factor=2,
                               mode='trilinear', align_corners=False)[0, 0]
        return fine[2:shape[0]+2, 2:shape[1]+2, 2:shape[2]+2]

    def cycle(self, level, p, rhs):
        if level == len(self.shapes)-1:
            return (self.coarse_inverse@rhs.reshape(-1)).reshape(rhs.shape)
        p = self.smooth[level](p, rhs)
        residual = rhs-negative_laplacian(p, self.spacings[level])
        coarse = nnf.avg_pool3d(residual[None, None], 2, 2)[0, 0]
        correction = self.cycle(level+1, torch.zeros_like(coarse), coarse)
        return self.smooth[level](p+self.prolong(correction, p.shape), rhs)

    def advance(self):
        for _ in range(self.options.check_every):
            self.p.copy_(self.cycle(0, self.p, self.rhs))

    def capture(self):
        from .graph_capture import capture_initialization
        with capture_initialization():
            self._capture()

    def _capture(self):
        # Warm compilation outside capture; input contents are restored by solve.
        saved = self.p.clone()
        stream = torch.cuda.Stream(device=self.p.device)
        stream.wait_stream(torch.cuda.current_stream(self.p.device))
        with torch.cuda.stream(stream):
            self.advance(); self.advance()
        torch.cuda.current_stream(self.p.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.advance()
        self.p.copy_(saved)
        self.backend = 'compiled-dirichlet-graph'

    @torch.no_grad()
    def solve(self, rhs, initial=None):
        if rhs.shape != self.rhs.shape or rhs.dtype != self.rhs.dtype or rhs.device != self.rhs.device:
            raise ValueError('Dirichlet pressure RHS shape/dtype/device mismatch')
        if initial is not None and (initial.shape != rhs.shape or initial.dtype != rhs.dtype or initial.device != rhs.device):
            raise ValueError('invalid initial Dirichlet pressure')
        self.rhs.copy_(rhs)
        self.p.zero_() if initial is None else self.p.copy_(initial)
        metrics = torch.stack((torch.isfinite(self.rhs).all().to(rhs.dtype),
                               torch.isfinite(self.p).all().to(rhs.dtype),
                               torch.linalg.vector_norm(self.rhs), self._norm(self.rhs, self.p))).tolist()
        finite, finite_p, norm, residual = metrics
        if not finite or not finite_p or not isfinite(norm) or not isfinite(residual):
            raise ValueError('nonfinite Dirichlet pressure input')
        tolerance = max(self.options.atol, self.options.rtol*norm)
        if self.graph_requested and self.graph is None:
            self.capture()
        cycles = 0
        while residual > tolerance and cycles < self.options.max_cycles:
            count = min(self.options.check_every, self.options.max_cycles-cycles)
            if count == self.options.check_every:
                self.advance() if self.graph is None else self.graph.replay()
            else:
                for _ in range(count):
                    self.p.copy_(self.cycle(0, self.p, self.rhs))
            cycles += count
            residual = self._norm(self.rhs, self.p).item()
            if not isfinite(residual):
                raise RuntimeError('nonfinite Dirichlet multigrid residual')
        if residual > tolerance:
            raise RuntimeError(f'Dirichlet multigrid failed: {residual:g} > {tolerance:g}')
        return self.p.clone(), dict(cycles=cycles, residual_norm=residual, tolerance=tolerance, backend=self.backend)
