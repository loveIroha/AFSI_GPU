"""Fixed CN Jacobi polynomial in bounded buffers; optional CUDA graph replay.

Only the sweep chain is captured. Initialization fuses pressure subtraction
with boundary masking and the diagonal initial guess. Public results are owned;
the Stokes implementation may borrow scratch until its next Helmholtz call.
"""
from collections import OrderedDict
from contextlib import nullcontext
import torch
from .grid import gradient, zero_normal, velocity_laplacian
from .graph_capture import capture_initialization


class TorchHelmholtzKernels:
    """CPU oracle for workspace lifetime and fixed-polynomial tests."""
    def initialize(self,b,p,diagonal,rhs,u,parameters,grid):
        gp = gradient(p,grid.spacing) if p is not None else None
        values = zero_normal(b if gp is None else tuple(
            r-parameters[1]*g for r,g in zip(b,gp)))
        for r,v,d,dest in zip(rhs,values,diagonal,u):
            r.copy_(v)
            torch.div(r,d,out=dest)

    def sweep(self,u,rhs,diagonal,out,parameters,grid):
        values = zero_normal(tuple(v+(r-v+parameters[0]*velocity_laplacian(v,c,grid.spacing))/d
            for c,(v,r,d) in enumerate(zip(u,rhs,diagonal))))
        for dest,value in zip(out,values):
            dest.copy_(value)


class HelmholtzWorkspace:
    def __init__(self,grid,diagonal,*,backend='graph',kernels=None):
        if backend not in ('torch','triton','graph'):
            raise ValueError('invalid Helmholtz workspace backend')
        like = diagonal[0]
        if backend=='graph' and not like.is_cuda:
            raise ValueError('Helmholtz graph backend requires CUDA')
        self.grid,self.diagonal,self.backend = grid,diagonal,backend
        self.rhs,self.first,self.second = [tuple(torch.empty_like(d) for d in diagonal) for _ in range(3)]
        self.parameters = like.new_zeros(2)
        self._parameter_values = None
        if kernels is None:
            if backend=='torch':
                kernels = TorchHelmholtzKernels()
            else:
                from ._triton_helmholtz import HelmholtzKernels
                kernels = HelmholtzKernels()
        self.kernels = kernels
        self.graphs = OrderedDict()
        self._stream = None
        self.solves = self.sweeps = self.graph_builds = 0

    def _block(self,count):
        u,out = self.first,self.second
        for _ in range(count):
            self.kernels.sweep(u,self.rhs,self.diagonal,out,self.parameters,self.grid)
            u,out = out,u
        return u

    def _capture(self,count):
        # Bounded LRU avoids retaining a graph for every possible refined dt.
        if len(self.graphs)>=4:
            self.graphs.popitem(last=False)
        with torch.cuda.device(self.parameters.device), capture_initialization():
            stream = torch.cuda.Stream(device=self.parameters.device)
            stream.wait_stream(torch.cuda.current_stream(self.parameters.device))
            with torch.cuda.stream(stream):
                self._block(count)  # compile all kernels before entering capture
            torch.cuda.current_stream(self.parameters.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                self._block(count)
            torch.cuda.current_stream(self.parameters.device).wait_stream(stream)
        self.graphs[count] = graph
        self.graph_builds += 1

    @torch.no_grad()
    def solve(self,b,*,alpha,count,pressure=None,pressure_scale=0.,owned=True):
        self.grid.check_velocity(b)
        if type(count) is not int or count<1:
            raise ValueError('positive fixed Helmholtz iteration count required')
        if any(v.device!=self.parameters.device or v.dtype!=self.parameters.dtype for v in b):
            raise ValueError('Helmholtz workspace dtype/device mismatch')
        if pressure is not None and (pressure.shape!=self.grid.shape or
                pressure.device!=self.parameters.device or pressure.dtype!=self.parameters.dtype):
            raise ValueError('Helmholtz pressure shape/device/dtype mismatch')
        if self.parameters.is_cuda:
            stream = torch.cuda.current_stream(self.parameters.device).cuda_stream
            if self._stream is not None and self._stream!=stream:
                raise ValueError('Helmholtz workspace requires its original CUDA stream')
            self._stream = stream
        context = torch.cuda.device(self.parameters.device) if self.parameters.is_cuda else nullcontext()
        with context:
            parameters = float(alpha),float(pressure_scale)
            if parameters!=self._parameter_values:
                self.parameters[0].fill_(alpha)
                self.parameters[1].fill_(pressure_scale)
                self._parameter_values = parameters
            b = tuple(v.contiguous() for v in b)
            pressure = None if pressure is None else pressure.contiguous()
            # Compile/capture using defined buffers, then restore initialization
            # because warmup and capture execute and overwrite both ping-pong arrays.
            if self.backend=='graph' and count not in self.graphs:
                self.kernels.initialize(b,pressure,self.diagonal,self.rhs,self.first,self.parameters,self.grid)
                self._capture(count)
            self.kernels.initialize(b,pressure,self.diagonal,self.rhs,self.first,self.parameters,self.grid)
            if self.backend=='graph':
                self.graphs.move_to_end(count)
                self.graphs[count].replay()
                result = self.second if count%2 else self.first
            else:
                result = self._block(count)
            self.solves += 1
            self.sweeps += count
            return tuple(v.clone() for v in result) if owned else result

    def summary(self):
        return dict(backend=self.backend,solves=self.solves,sweeps=self.sweeps,
                    graph_builds=self.graph_builds,cached_graphs=len(self.graphs),
                    buffer_bytes=sum(v.numel()*v.element_size() for group in
                        (self.rhs,self.first,self.second) for v in group))
