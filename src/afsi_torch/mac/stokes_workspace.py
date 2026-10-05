"""Owned Stokes scratch, with unchanged no-slip residuals and true norms.

Only intermediate residuals are borrowed. Returned flow states never alias
these buffers. Like the existing pressure/mass workspaces, this object is
not intended for concurrent solves.
"""
import torch
from .grid import gradient,divergence
from .execution import tensor_kernel


def _reduce_start(partial,count):
    totals=partial.sum(0)
    return torch.stack((torch.sqrt(totals[0]),totals[1]/count,partial[:,2].amin()>0))


def _reduce_norms(partial):
    return torch.sqrt(partial.sum(0))


def _pressure_input_reduce(partial,count):
    sums=partial[:,:3].sum(0)/count
    return torch.cat((sums,partial[:,3:].amin(0)))


def _center_pressure(rhs,p,metrics):
    rhs.sub_(metrics[0]);p.sub_(metrics[2])


def _pressure_packet(inputs,norms):
    return torch.stack((inputs[3],inputs[4],inputs[0],inputs[1],norms[0],norms[1]))


class PressureStartMetrics:
    """Block input and true residual reductions around the original centering."""
    def __init__(self,solver):
        self.solver=solver
        sample=solver.diagonals[0]
        count=(sample.numel()+255)//256
        self.inputs=sample.new_empty((count,5));self.norms=sample.new_empty((count,2))
        self.spacing=sample.new_tensor([h*h for h in solver.spacings[0]])
        self.reduce_inputs=tensor_kernel(_pressure_input_reduce,sample.device)
        self.center=tensor_kernel(_center_pressure,sample.device)
        self.reduce_norms=tensor_kernel(_reduce_norms,sample.device)
        self.packet=tensor_kernel(_pressure_packet,sample.device)

    def measure(self,rhs,p):
        if not rhs.is_cuda or not rhs.is_contiguous() or not p.is_contiguous():
            return self.solver.__class__._start_metrics(self.solver,rhs,p)
        from ._triton_stokes import pressure_input_partial,pressure_norm_partial
        pressure_input_partial(rhs,p,self.inputs)
        inputs=self.reduce_inputs(self.inputs,rhs.numel())
        self.center(rhs,p,inputs)
        pressure_norm_partial(rhs,p,self.norms,self.spacing)
        return self.packet(inputs,self.reduce_norms(self.norms))


class StokesWorkspace:
    def __init__(self,flow):
        self.flow=flow
        sample=flow.pressure_solver.diagonals[0]
        self.residual=torch.empty_like(sample)
        self.partial=sample.new_empty(((sample.numel()+255)//256,2))
        self.start_partial=sample.new_empty((len(self.partial),3))
        self.coefficients=sample.new_empty(9)
        self._coefficient_key=None
        self.update_coefficients()
        self.reduce_start=tensor_kernel(_reduce_start,sample.device)
        self.reduce_norms=tensor_kernel(_reduce_norms,sample.device)

    def update_coefficients(self):
        f=self.flow
        # Match eager CUDA tensor / Python-scalar arithmetic: form the
        # reciprocal before casting it to the tensor dtype. Storing rounded
        # FP32 h and dividing in Triton adds an avoidable rounding/approximate
        # division difference, amplified when directional divergences cancel.
        key=(f.alpha,f.dt/f.rho,f.rho/f.dt,*(1/h for h in f.grid.spacing),
             *(1/(h*h) for h in f.grid.spacing))
        if key!=self._coefficient_key:
            self.coefficients.copy_(self.coefficients.new_tensor(key))
            self._coefficient_key=key

    def start(self,b,p):
        """One scalar packet: RHS norm, pressure mean, pressure finite flag."""
        if p.is_cuda and all(v.is_contiguous() for v in (*b,p)):
            from ._triton_stokes import start_partial
            start_partial(b,p,self.start_partial,self.flow.grid.shape)
            return self.reduce_start(self.start_partial,p.numel())
        return torch.stack((torch.sqrt(sum(v.square().sum() for v in b)),
                            p.mean(),torch.isfinite(p).all()))

    def measure(self,u,p,b):
        """Compute D u once for both its norm and the next Schur RHS."""
        f=self.flow
        self.update_coefficients()
        if p.is_cuda and all(v.is_contiguous() for v in (*u,p,*b)):
            from ._triton_stokes import residual_partial
            residual_partial(u,p,b,self.residual,self.partial,f.grid.shape,
                             self.coefficients)
            return self.reduce_norms(self.partial),self.residual
        gp=gradient(p,f.grid.spacing)
        momentum=tuple(v-f.alpha*l+f.dt/f.rho*g-r
                       for v,l,g,r in zip(u,f._lap(u),gp,b))
        div=divergence(u,f.grid.spacing)
        metrics=torch.stack((torch.sqrt(sum(v.square().sum() for v in momentum)),
                             torch.linalg.vector_norm(div)))
        self.residual.copy_(-f.rho/f.dt*div)
        return metrics,self.residual

    def summary(self):
        return dict(backend='blocked-triton' if self.residual.is_cuda else 'buffered-cpu',
            allocated_bytes=sum(v.numel()*v.element_size() for v in
                                (self.residual,self.partial,self.start_partial,self.coefficients)),
            divergence_reused=True)
