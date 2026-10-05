"""Owned ping-pong buffers for unchanged BE Jacobi sweeps.

Borrowed velocity scratch is consumed by the projection before the next solve.
Public projected velocities and pressure remain separately owned. Workspaces
are local to a flow solver and are not intended for concurrent solves.
"""
import torch
from .open_boundary import velocity_laplacian
from .execution import tensor_kernel


class BEHelmholtzWorkspace:
    def __init__(self, flow, *, fused=False, graph=False):
        self.flow = flow
        self.u = tuple(torch.empty_like(d) for d in flow.diagonals)
        self.next = tuple(torch.empty_like(d) for d in flow.diagonals)
        self.rhs = tuple(torch.empty_like(d) for d in flow.diagonals)
        self.kernel = tensor_kernel(self._advance, self.u[0].device) if fused else self._advance
        self.graph_requested = graph and self.u[0].is_cuda
        self.graph = None

    def load(self, rhs):
        for b, u, value in zip(self.rhs, self.u, rhs):
            b.copy_(value); u.copy_(value)
        return self.u, self.rhs

    def _advance(self):
        f = self.flow
        source, target = self.u, self.next
        for _ in range(f.options.check_every):
            for c, (out, u, b, d) in enumerate(zip(target, source, self.rhs, f.diagonals)):
                out.copy_(u+(b-u+f.dt*f.mu/f.rho*velocity_laplacian(u, c, f.grid.spacing))/d)
            source, target = target, source
        if f.options.check_every % 2:
            for out, value in zip(self.u, source):
                out.copy_(value)
        return self.u

    def capture(self):
        from .graph_capture import capture_initialization
        with capture_initialization():
            saved = tuple(u.clone() for u in self.u)
            stream = torch.cuda.Stream(device=self.u[0].device)
            stream.wait_stream(torch.cuda.current_stream(self.u[0].device))
            with torch.cuda.stream(stream):
                self.kernel(); self.kernel()
            torch.cuda.current_stream(self.u[0].device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.kernel()
            for u, old in zip(self.u, saved):
                u.copy_(old)

    def advance(self):
        if self.graph_requested and self.graph is None:
            self.capture()
        self.kernel() if self.graph is None else self.graph.replay()
        return self.u
