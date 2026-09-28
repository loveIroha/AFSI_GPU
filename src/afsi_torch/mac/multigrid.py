"""Matrix-free geometric V cycles for the closed-box MAC pressure equation."""
from dataclasses import dataclass
from math import isfinite, prod
import torch
import torch.nn.functional as F
from .grid import negative_laplacian, slab


@dataclass(frozen=True)
class MGOptions:
    rtol: float = 1e-10
    atol: float = 1e-12
    max_cycles: int = 100
    smooth: int = 4
    check_every: int = 2

    def __post_init__(self):
        if (any(not isfinite(x) or x < 0 for x in (self.rtol, self.atol)) or
                self.rtol+self.atol == 0 or any(type(n) is not int or n < 1 for n in
                (self.max_cycles, self.smooth, self.check_every))):
            raise ValueError('invalid multigrid options')


class GeometricMultigrid:
    def __init__(self, grid, *, device='cpu', dtype=torch.float64, options=None):
        self.options = MGOptions() if options is None else options
        self.shapes, self.spacings, self.diagonals = [], [], []
        shape, spacing = grid.shape, grid.spacing
        while True:
            self.shapes.append(tuple(shape))
            self.spacings.append(tuple(spacing))
            diag = torch.zeros(shape, device=device, dtype=dtype)
            for c, h in enumerate(spacing):
                diag[slab(c, None, -1)] += 1/h**2
                diag[slab(c, 1, None)] += 1/h**2
            self.diagonals.append(diag)
            if min(shape) <= 4 or any(n % 2 for n in shape):
                break
            shape = tuple(n//2 for n in shape)
            spacing = tuple(2*h for h in spacing)
        if prod(shape) > 512:
            raise ValueError('MAC dimensions must coarsen to <=512 cells; use powers of two')
        # Setup-only small coarse inverse on the selected device. The rank-one
        # term fixes the constant nullspace without pinning one physical cell.
        count = prod(shape)
        ids = torch.arange(count, device=device).reshape(shape)
        A = torch.zeros((count, count), device=device, dtype=dtype)
        for c, h in enumerate(spacing):
            a, b = ids[slab(c, None, -1)].reshape(-1), ids[slab(c, 1, None)].reshape(-1)
            w = torch.full_like(a, 1/h**2, dtype=dtype)
            for row, col, value in ((a,a,w),(b,b,w),(a,b,-w),(b,a,-w)):
                A.index_put_((row, col), value, accumulate=True)
        self.coarse_inverse = torch.linalg.inv(A+torch.ones_like(A)/count)

    def _smooth(self, level, p, rhs):
        for _ in range(self.options.smooth):
            p = p+(2/3)*(rhs-negative_laplacian(p, self.spacings[level]))/self.diagonals[level]
        return p-p.mean()

    def _cycle(self, level, p, rhs):
        if level == len(self.shapes)-1:
            solution = (self.coarse_inverse@rhs.reshape(-1)).reshape(rhs.shape)
            return solution-solution.mean()
        p = self._smooth(level, p, rhs)
        residual = rhs-negative_laplacian(p, self.spacings[level])
        coarse = F.avg_pool3d(residual[None, None], 2, 2)[0, 0]
        coarse = coarse-coarse.mean()
        correction = self._cycle(level+1, torch.zeros_like(coarse), coarse)
        p = p+F.interpolate(correction[None,None], size=p.shape,
                             mode='trilinear', align_corners=False)[0,0]
        return self._smooth(level, p, rhs)

    @torch.no_grad()
    def solve(self, rhs, initial=None):
        if (rhs.shape != self.shapes[0] or rhs.device != self.diagonals[0].device or
                rhs.dtype != self.diagonals[0].dtype or not torch.isfinite(rhs).all()):
            raise ValueError('invalid pressure RHS')
        if abs(rhs.mean().item()) > 1e-12+1e-10*rhs.abs().mean().item():
            raise ValueError('incompatible Neumann pressure RHS: net flux must vanish')
        rhs = rhs-rhs.mean()
        p = torch.zeros_like(rhs) if initial is None else initial.clone()
        if p.shape != rhs.shape or p.device != rhs.device or p.dtype != rhs.dtype or not torch.isfinite(p).all():
            raise ValueError('invalid initial pressure')
        p -= p.mean()
        tolerance = max(self.options.atol, self.options.rtol*torch.linalg.vector_norm(rhs).item())
        residual = torch.linalg.vector_norm(rhs-negative_laplacian(p, self.spacings[0])).item()
        if residual <= tolerance:
            return p, dict(cycles=0, residual_norm=residual, tolerance=tolerance)
        for cycle in range(1, self.options.max_cycles+1):
            p = self._cycle(0, p, rhs)
            if cycle % self.options.check_every == 0 or cycle == self.options.max_cycles:
                residual = torch.linalg.vector_norm(rhs-negative_laplacian(p, self.spacings[0])).item()
                if not isfinite(residual):
                    raise RuntimeError('nonfinite multigrid residual')
                if residual <= tolerance:
                    return p, dict(cycles=cycle, residual_norm=residual, tolerance=tolerance)
        raise RuntimeError(f'pressure multigrid did not converge: {residual:g} > {tolerance:g}')
