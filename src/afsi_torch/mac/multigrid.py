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
    def __init__(self, grid, *, device='cpu', dtype=torch.float64, options=None, backend='torch'):
        if backend not in ('torch','fused','workspace','graph'):
            raise ValueError('pressure backend must be torch, fused, workspace or graph')
        self.backend = 'torch'
        self.workspace = None
        self._rhs_metrics = None
        self._residual_norm = lambda rhs,p: torch.linalg.vector_norm(rhs-negative_laplacian(p,self.spacings[0]))
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
        if backend!='torch':
            from .mg_workspace import MGWorkspace
            self.workspace=MGWorkspace(self,optimized=backend in ('workspace','graph'),graphs=backend=='graph')
            self.backend=self.workspace.kernels.name+('-workspace' if backend=='workspace' else '-graph' if backend=='graph' else '')
            if self.workspace.optimized:
                from .execution import tensor_kernel
                self._start_metrics=tensor_kernel(self._start_metrics,device)

    def _start_metrics(self,rhs,p):
        valid_rhs,valid_p=torch.isfinite(rhs).all(),torch.isfinite(p).all()
        mean,absolute_mean=rhs.mean(),rhs.abs().mean()
        rhs.sub_(mean); p.sub_(p.mean())
        return torch.stack((valid_rhs.to(rhs.dtype),valid_p.to(rhs.dtype),mean,absolute_mean,
            torch.linalg.vector_norm(rhs),torch.linalg.vector_norm(rhs-negative_laplacian(p,self.spacings[0]))))

    def _solve_workspace(self,rhs,initial):
        w=self.workspace
        w.rhs[0].copy_(rhs)
        w.p[0].zero_() if initial is None else w.p[0].copy_(initial)
        finite,valid_p,mean,absolute_mean,rhs_norm,residual=self._start_metrics(w.rhs[0],w.p[0]).tolist()
        if not finite:
            raise ValueError('invalid pressure RHS')
        if abs(mean)>1e-12+1e-10*absolute_mean:
            raise ValueError('incompatible Neumann pressure RHS: net flux must vanish')
        if not valid_p:
            raise ValueError('invalid initial pressure')
        if not isfinite(rhs_norm) or not isfinite(residual):
            raise RuntimeError('nonfinite multigrid residual or RHS norm')
        tolerance=max(self.options.atol,self.options.rtol*rhs_norm)
        cycle=0
        while residual>tolerance and cycle<self.options.max_cycles:
            count=min(self.options.check_every,self.options.max_cycles-cycle)
            w.advance(count); cycle+=count
            residual=w.residual_norm().item()
            if not isfinite(residual):
                raise RuntimeError('nonfinite multigrid residual')
        if residual>tolerance:
            raise RuntimeError(f'pressure multigrid did not converge: {residual:g} > {tolerance:g}')
        return w.p[0].clone(),dict(cycles=cycle,residual_norm=residual,tolerance=tolerance,backend=self.backend)

    def enable_tensor_fusion(self):
        from .execution import tensor_kernel
        device=self.diagonals[0].device
        self._rhs_metrics=tensor_kernel(lambda rhs:torch.stack((torch.isfinite(rhs).all().to(rhs.dtype),
            rhs.mean(),rhs.abs().mean())),device)
        self._residual_norm=tensor_kernel(self._residual_norm,device)

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
                rhs.dtype != self.diagonals[0].dtype):
            raise ValueError('invalid pressure RHS')
        if initial is not None and (initial.shape!=rhs.shape or initial.device!=rhs.device or initial.dtype!=rhs.dtype):
            raise ValueError('invalid initial pressure')
        if self.workspace is not None and self.workspace.optimized:
            return self._solve_workspace(rhs,initial)
        if self._rhs_metrics is None:
            if not torch.isfinite(rhs).all():
                raise ValueError('invalid pressure RHS')
            mean,absolute_mean=rhs.mean().item(),rhs.abs().mean().item()
        else:
            finite,mean,absolute_mean=self._rhs_metrics(rhs).tolist()
            if not finite:
                raise ValueError('invalid pressure RHS')
        if abs(mean)>1e-12+1e-10*absolute_mean:
            raise ValueError('incompatible Neumann pressure RHS: net flux must vanish')
        rhs = rhs-rhs.mean()
        p = torch.zeros_like(rhs) if initial is None else initial.clone()
        if p.shape != rhs.shape or p.device != rhs.device or p.dtype != rhs.dtype or not torch.isfinite(p).all():
            raise ValueError('invalid initial pressure')
        p -= p.mean()
        rhs_norm=torch.linalg.vector_norm(rhs).item()
        residual = self._residual_norm(rhs,p).item()
        if not isfinite(rhs_norm) or not isfinite(residual):
            raise RuntimeError('nonfinite multigrid residual or RHS norm')
        tolerance=max(self.options.atol,self.options.rtol*rhs_norm)
        if residual <= tolerance:
            return p, dict(cycles=0, residual_norm=residual, tolerance=tolerance, backend=self.backend)
        if self.workspace is not None:
            self.workspace.initialize(p, rhs)
        for cycle in range(1, self.options.max_cycles+1):
            p = (self._cycle(0, p, rhs) if self.workspace is None else self.workspace.cycle())
            if cycle % self.options.check_every == 0 or cycle == self.options.max_cycles:
                residual = self._residual_norm(rhs,p).item()
                if not isfinite(residual):
                    raise RuntimeError('nonfinite multigrid residual')
                if residual <= tolerance:
                    # The result belongs to the caller, not to reusable work storage.
                    result = p if self.workspace is None else p.clone()
                    return result, dict(cycles=cycle, residual_norm=residual,
                                        tolerance=tolerance, backend=self.backend)
        raise RuntimeError(f'pressure multigrid did not converge: {residual:g} > {tolerance:g}')
