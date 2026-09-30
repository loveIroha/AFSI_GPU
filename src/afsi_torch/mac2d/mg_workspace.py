"""Fixed channel V-cycle storage and optional graph blocks between residual checks."""
import torch
import torch.nn.functional as F
from .grid import negative_laplacian
from ..mac.graph_capture import capture_initialization


class TensorKernels:
    name='buffered-channel'

    def smooth(self,p,rhs,diagonal,out,spacing):
        out.copy_(p+(2/3)*(rhs-negative_laplacian(p,spacing))/diagonal)

    def residual(self,p,rhs,out,spacing):
        out.copy_(rhs-negative_laplacian(p,spacing))

    def residual_restrict(self,p,rhs,out,spacing):
        residual=rhs-negative_laplacian(p,spacing)
        out.copy_(F.avg_pool2d(residual[None,None],2,2)[0,0])

    def prolong_add(self,coarse,fine):
        padded=torch.cat((coarse[:1],coarse,-coarse[-1:]),0)
        padded=torch.cat((padded[:,:1],padded,padded[:,-1:]),1)
        fine.add_(F.interpolate(padded[None,None],scale_factor=2,mode='bilinear',
                               align_corners=False)[0,0,2:-2,2:-2])


class ChannelWorkspace:
    def __init__(self,solver,*,fused=True,graphs=False):
        sample=solver.diagonals[0]
        self.kernels=TensorKernels()
        if fused and sample.is_cuda:
            try:
                from ._triton_mg import TritonKernels
            except ImportError as exc:
                raise ImportError('Channel GPU kernels require matching Triton: pip install -e ".[fused]"') from exc
            self.kernels=TritonKernels()
        self.options=solver.options
        self.spacings=solver.spacings
        self.diagonals=solver.diagonals
        self.inverse=solver.inverse
        self.p=[torch.empty_like(v) for v in self.diagonals]
        self.rhs=[torch.empty_like(v) for v in self.diagonals]
        self.other=[torch.empty_like(v) for v in self.diagonals]
        self.residual=torch.empty_like(sample)
        self.allocated_bytes=sum(v.numel()*v.element_size() for group in (self.p,self.rhs,self.other) for v in group)+self.residual.numel()*self.residual.element_size()
        self.graphs={}
        if graphs:
            if not sample.is_cuda:
                raise ValueError('pressure CUDA graphs require a CUDA device')
            if self.options.check_every>16:
                raise ValueError('pressure graph check_every must be <=16')
            self._capture()

    def _smooth(self,level):
        p,out=self.p[level],self.other[level]
        for _ in range(self.options.smooth):
            self.kernels.smooth(p,self.rhs[level],self.diagonals[level],out,self.spacings[level])
            p,out=out,p
        if p is not self.p[level]:
            self.p[level].copy_(p)

    def _cycle(self,level):
        if level==len(self.p)-1:
            torch.mv(self.inverse,self.rhs[level].reshape(-1),out=self.p[level].reshape(-1))
            return
        self._smooth(level)
        self.kernels.residual_restrict(self.p[level],self.rhs[level],self.rhs[level+1],self.spacings[level])
        self.p[level+1].zero_()
        self._cycle(level+1)
        self.kernels.prolong_add(self.p[level+1],self.p[level])
        self._smooth(level)

    def _block(self,count):
        for _ in range(count):
            self._cycle(0)

    def _capture(self):
        with torch.cuda.device(self.p[0].device), capture_initialization():
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

    def initialize(self,p,rhs):
        self.p[0].copy_(p); self.rhs[0].copy_(rhs)

    def advance(self,count):
        if self.graphs:
            self.graphs[count].replay()
        else:
            self._block(count)
        return self.p[0]

    def residual_norm(self):
        self.kernels.residual(self.p[0],self.rhs[0],self.residual,self.spacings[0])
        return torch.linalg.vector_norm(self.residual)
