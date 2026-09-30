"""Geometric pressure V cycles with mixed channel boundary conditions."""
from math import prod,isfinite
import torch
import torch.nn.functional as F
from ..mac.multigrid import MGOptions
from ..mac.execution import tensor_kernel
from .grid import negative_laplacian


def smooth_block(p,rhs,spacing,diagonal,iterations):
    for _ in range(iterations):
        p=p+(2/3)*(rhs-negative_laplacian(p,spacing))/diagonal
    return p


class ChannelMultigrid:
    def __init__(self,grid,*,device='cpu',dtype=torch.float64,options=None,fused=True,backend='reference'):
        if backend not in ('reference','workspace','graph'):
            raise ValueError('pressure backend must be reference, workspace or graph')
        self.options=options or MGOptions()
        self.shapes,self.spacings,self.diagonals=[],[],[]
        shape,spacing=grid.shape,grid.spacing
        while True:
            self.shapes.append(tuple(shape))
            self.spacings.append(tuple(spacing))
            hx,hy=spacing
            diag=torch.zeros(shape,device=device,dtype=dtype)
            diag[:-1]+=1/hx**2
            diag[1:]+=1/hx**2
            diag[:,:-1]+=1/hy**2
            diag[:,1:]+=1/hy**2
            diag[-1]+=2/hx**2
            self.diagonals.append(diag)
            if min(shape)<=4 or any(n%2 for n in shape):
                break
            shape=tuple(n//2 for n in shape)
            spacing=tuple(2*h for h in spacing)
        count=prod(shape)
        if count>512:
            raise ValueError('channel mesh must coarsen to <=512 cells')
        eye=torch.eye(count,device=device,dtype=dtype)
        # Small setup-only coarse inverse; the Dirichlet outlet removes the
        # nullspace, so neither mean subtraction nor a pressure pin is applied.
        A=torch.stack([negative_laplacian(v.reshape(shape),spacing).reshape(-1) for v in eye],1)
        self.inverse=torch.linalg.inv(A)
        self.actions=[]
        self.spacing_tensors=[]
        for h in self.spacings:
            h=self.diagonals[0].new_tensor(h)
            self.spacing_tensors.append(h)
            action=lambda p,h=h:negative_laplacian(p,h)
            self.actions.append(tensor_kernel(action,device) if fused else action)
        self.smooth_block=tensor_kernel(smooth_block,device) if fused else smooth_block
        self.backend='compiled-2d' if fused and torch.device(device).type=='cuda' else 'torch-2d'
        self.workspace=None
        if backend!='reference':
            from .mg_workspace import ChannelWorkspace
            self.workspace=ChannelWorkspace(self,fused=fused,graphs=backend=='graph')
            self.backend=self.workspace.kernels.name+('-graph' if backend=='graph' else '')
            self._start_metrics=tensor_kernel(self._start_metrics,device) if fused else self._start_metrics

    def _start_metrics(self,rhs,p,residual):
        valid=torch.isfinite(rhs).all() & torch.isfinite(p).all()
        return torch.stack((valid.to(rhs.dtype),torch.linalg.vector_norm(rhs),torch.linalg.vector_norm(residual)))

    def _smooth(self,level,p,rhs):
        return self.smooth_block(p,rhs,self.spacing_tensors[level],self.diagonals[level],self.options.smooth)

    def _cycle(self,level,p,rhs):
        if level==len(self.shapes)-1:
            return (self.inverse@rhs.reshape(-1)).reshape(rhs.shape)
        p=self._smooth(level,p,rhs)
        residual=rhs-self.actions[level](p)
        coarse=F.avg_pool2d(residual[None,None],2,2)[0,0]
        correction=self._cycle(level+1,torch.zeros_like(coarse),coarse)
        # Interpolate through boundary ghosts: even at Neumann boundaries,
        # odd at p=0 outlet. Border-value clamping would violate the outlet
        # correction and leave slowly converging low-frequency errors.
        padded=torch.cat((correction[:1],correction,-correction[-1:]),0)
        padded=torch.cat((padded[:,:1],padded,padded[:,-1:]),1)
        prolong=F.interpolate(padded[None,None],scale_factor=2,mode='bilinear',align_corners=False)[0,0,2:-2,2:-2]
        p=p+prolong
        return self._smooth(level,p,rhs)

    @torch.no_grad()
    def solve(self,rhs,initial=None):
        if rhs.shape!=self.shapes[0] or rhs.device!=self.diagonals[0].device or rhs.dtype!=self.diagonals[0].dtype:
            raise ValueError('invalid channel pressure RHS')
        if initial is not None and (initial.shape!=rhs.shape or initial.dtype!=rhs.dtype or initial.device!=rhs.device):
            raise ValueError('invalid channel pressure initial guess')
        if self.workspace is not None:
            return self._solve_workspace(rhs,initial)
        p=torch.zeros_like(rhs) if initial is None else initial.clone()
        if p.shape!=rhs.shape or not torch.isfinite(rhs).all() or not torch.isfinite(p).all():
            raise ValueError('finite pressure RHS/initial guess required')
        rhs_norm=torch.linalg.vector_norm(rhs).item()
        residual=torch.linalg.vector_norm(rhs-self.actions[0](p)).item()
        if not isfinite(rhs_norm) or not isfinite(residual):
            raise RuntimeError('nonfinite channel pressure residual or RHS norm')
        tol=max(self.options.atol,self.options.rtol*rhs_norm)
        if residual<=tol:
            return p,dict(cycles=0,residual_norm=residual,tolerance=tol,backend=self.backend)
        for cycle in range(1,self.options.max_cycles+1):
            p=self._cycle(0,p,rhs)
            if cycle%self.options.check_every==0 or cycle==self.options.max_cycles:
                residual=torch.linalg.vector_norm(rhs-self.actions[0](p)).item()
                if not isfinite(residual):
                    raise RuntimeError('nonfinite channel pressure residual')
                if residual<=tol:
                    return p,dict(cycles=cycle,residual_norm=residual,tolerance=tol,backend=self.backend)
        raise RuntimeError(f'channel MG did not converge: {residual:g} > {tol:g}')

    def _solve_workspace(self,rhs,initial):
        w=self.workspace
        w.p[0].zero_() if initial is None else w.p[0].copy_(initial)
        w.rhs[0].copy_(rhs)
        w.kernels.residual(w.p[0],w.rhs[0],w.residual,self.spacings[0])
        valid,rhs_norm,residual=self._start_metrics(w.rhs[0],w.p[0],w.residual).tolist()
        if not valid:
            raise ValueError('finite pressure RHS/initial guess required')
        if not isfinite(rhs_norm) or not isfinite(residual):
            raise RuntimeError('nonfinite channel pressure residual or RHS norm')
        tol=max(self.options.atol,self.options.rtol*rhs_norm)
        cycle=0
        while residual>tol and cycle<self.options.max_cycles:
            count=min(self.options.check_every,self.options.max_cycles-cycle)
            w.advance(count); cycle+=count
            residual=w.residual_norm().item()
            if not isfinite(residual):
                raise RuntimeError('nonfinite channel pressure residual')
        if residual>tol:
            raise RuntimeError(f'channel MG did not converge: {residual:g} > {tol:g}')
        # Return independent storage: later solves/graph replays cannot mutate it.
        return w.p[0].clone(),dict(cycles=cycle,residual_norm=residual,tolerance=tol,backend=self.backend)
