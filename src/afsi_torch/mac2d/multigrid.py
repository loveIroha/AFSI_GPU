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
    def __init__(self,grid,*,device='cpu',dtype=torch.float64,options=None,fused=True):
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
        p=torch.zeros_like(rhs) if initial is None else initial.clone()
        if p.shape!=rhs.shape or not torch.isfinite(rhs).all() or not torch.isfinite(p).all():
            raise ValueError('finite pressure RHS/initial guess required')
        tol=max(self.options.atol,self.options.rtol*torch.linalg.vector_norm(rhs).item())
        residual=torch.linalg.vector_norm(rhs-self.actions[0](p)).item()
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
