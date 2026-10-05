"""CN--AB2 incompressible flow and midpoint quadrature FE/IB coupling.

Griffith--Luo (2017), Appendix A, equations 71--81. The Stokes solve retains
the existing no-slip velocity stencil and solves its actual pressure Schur
equation: H and G do NOT commute at these walls. All large arrays stay on the
tensor device; Neumann multigrid is the Schur preconditioner.
"""
from dataclasses import dataclass, asdict
from math import ceil, isfinite, log
import torch
from .flow import MACFlow, MACFlowResult, MACTransportGuardError
from .grid import zero_normal, slab, gradient, divergence, velocity_laplacian, convection
from .ppm import convection_ppm
from .coupling import MACIBStepper, MACState
from ..transport import cnab_policy, COURANT_LIMIT


@dataclass(frozen=True)
class CNABOptions:
    advection: str = 'ppm'
    helmholtz_rtol: float = 1e-13
    max_helmholtz_iterations: int = 512
    max_stokes_iterations: int = 40
    helmholtz_backend: str = 'auto'

    def __post_init__(self):
        if self.advection not in ('ppm','centered'):
            raise ValueError('CNAB advection must be ppm or centered')
        if self.helmholtz_backend not in ('auto','torch','triton','graph'):
            raise ValueError('CN Helmholtz backend must be auto, torch, triton or graph')
        if not isfinite(self.helmholtz_rtol) or not 0 < self.helmholtz_rtol < 1e-8:
            raise ValueError('CNAB Helmholtz relative tolerance must be in (0,1e-8)')
        for value in (self.max_helmholtz_iterations,self.max_stokes_iterations):
            if type(value) is not int or value < 1:
                raise ValueError('positive CNAB iteration limits required')


def blend(a,b,wa=.5,wb=.5):
    return tuple(wa*x+wb*y for x,y in zip(a,b))


def extrapolated_advection(current, previous, dt, previous_dt=None):
    """AB2 interval average for unequal steps; constant steps retain 1.5/-0.5."""
    previous_dt = dt if previous_dt is None else previous_dt
    if not isfinite(previous_dt) or previous_dt <= 0:
        raise ValueError('positive previous AB2 time step required')
    ratio = dt/(2*previous_dt)
    return blend(current, previous, 1+ratio, -ratio)


class CNABTransportGuardError(MACTransportGuardError):
    def __init__(self,numbers):
        self.diagnostics = dict(cnab_policy(),**numbers,triggered=['courant'])
        ValueError.__init__(self,
            f'CNAB explicit convection CFL={numbers["courant"]:.9g} > {COURANT_LIMIT}; '
            'elastic/IB restrictions also remain')


class MACCNABFlow(MACFlow):
    def __init__(self,*args,cnab_options=None,**kwargs):
        # CN viscosity has no explicit diffusion veto. Convection remains explicit.
        kwargs['implicit_transport'] = True
        super().__init__(*args,**kwargs)
        self.cnab_options = CNABOptions() if cnab_options is None else cnab_options
        self.stokes_calls = 0
        self.alpha = self.dt*self.mu/(2*self.rho)
        self._adv = convection_ppm if self.cnab_options.advection=='ppm' else convection
        self._helmholtz_diagonal = []
        for c,u in enumerate(self.grid.zeros(device=self.pressure_solver.diagonals[0].device,
                                              dtype=self.pressure_solver.diagonals[0].dtype)):
            diagonal = torch.full_like(u,1+2*self.alpha*sum(1/h**2 for h in self.grid.spacing))
            for axis,h in enumerate(self.grid.spacing):
                if axis != c:
                    diagonal[slab(axis,0,1)] += self.alpha/h**2
                    diagonal[slab(axis,-1,None)] += self.alpha/h**2
            diagonal[slab(c,0,1)] = 1.
            diagonal[slab(c,-1,None)] = 1.
            self._helmholtz_diagonal.append(diagonal)
        # Jacobi is a fixed linear polynomial, with a conservative infinity-norm
        # contraction bound. No direction-dependent stopping inside Schur actions.
        s = 2*self.alpha*sum(1/h**2 for h in self.grid.spacing)
        q = s/(1+s)
        self.helmholtz_iterations = max(1,ceil(log(self.cnab_options.helmholtz_rtol/(1+s))/log(q))) if q else 1
        if self.helmholtz_iterations > self.cnab_options.max_helmholtz_iterations:
            raise ValueError('CN Helmholtz iteration budget too small for this dt/viscosity/grid')
        self._lap = lambda u:tuple(velocity_laplacian(v,c,self.grid.spacing) for c,v in enumerate(u))
        self._metrics = self._stokes_metrics
        self._pressure_right = lambda b,p:tuple(r-self.dt/self.rho*g for r,g in
                                               zip(b,gradient(p,self.grid.spacing)))
        requested = self.cnab_options.helmholtz_backend
        device = self.pressure_solver.diagonals[0].device
        # Native same-checkpoint timings found the custom graph sweep slower.
        # Keep it explicitly selectable; automatic execution uses compiled torch.
        self.helmholtz_backend = 'torch' if requested=='auto' else requested
        self._helmholtz_workspace = None
        if self.helmholtz_backend!='torch':
            if device.type!='cuda':
                raise ValueError('Triton/graph CN Helmholtz requires CUDA')
            from .helmholtz import HelmholtzWorkspace
            self._helmholtz_workspace = HelmholtzWorkspace(self.grid,self._helmholtz_diagonal,
                                                         backend=self.helmholtz_backend)
        self.helmholtz_calls = 0
        self.helmholtz_sweeps = 0
        if self.execution_backend=='fused':
            from .execution import tensor_kernel
            device = self.pressure_solver.diagonals[0].device
            self._adv = tensor_kernel(self._adv,device)
            self._lap = tensor_kernel(self._lap,device)
            self._helmholtz_sweep = tensor_kernel(self._helmholtz_sweep,device)
            self._metrics = tensor_kernel(self._metrics,device)
            self._right = tensor_kernel(self._right,device)
            self._pressure_right = tensor_kernel(self._pressure_right,device)

    def advection(self,velocity):
        return self._adv(velocity,self.grid.spacing)

    def set_time_step(self,dt):
        """Update bounded scalar/diagonal CN data; retain the pressure graph/workspace."""
        if not isfinite(dt) or dt<=0:
            raise ValueError('positive finite CN time step required')
        if dt==self.dt:
            return
        alpha=dt*self.mu/(2*self.rho)
        s=2*alpha*sum(1/h**2 for h in self.grid.spacing)
        q=s/(1+s)
        count=max(1,ceil(log(self.cnab_options.helmholtz_rtol/(1+s))/log(q))) if q else 1
        if count>self.cnab_options.max_helmholtz_iterations:
            raise ValueError('CN Helmholtz iteration budget too small')
        self.dt,self.alpha,self.helmholtz_iterations=float(dt),alpha,count
        self.viscous_number=dt*self.mu/self.rho*sum(1/h**2 for h in self.grid.spacing)
        for c,diagonal in enumerate(self._helmholtz_diagonal):
            diagonal.fill_(1+s)
            for axis,h in enumerate(self.grid.spacing):
                if axis!=c:
                    diagonal[slab(axis,0,1)] += alpha/h**2
                    diagonal[slab(axis,-1,None)] += alpha/h**2
            diagonal[slab(c,0,1)]=1.
            diagonal[slab(c,-1,None)]=1.

    def _helmholtz_sweep(self,u,b):
        Lu = self._lap(u)
        return zero_normal(tuple(v+(r-v+self.alpha*l)/d for v,r,l,d in
                           zip(u,b,Lu,self._helmholtz_diagonal)))

    def _solve_helmholtz(self,b,pressure=None,*,owned=True):
        self.helmholtz_calls += 1
        self.helmholtz_sweeps += self.helmholtz_iterations
        if self._helmholtz_workspace is not None:
            return self._helmholtz_workspace.solve(b,alpha=self.alpha,count=self.helmholtz_iterations,
                pressure=pressure,pressure_scale=self.dt/self.rho,owned=owned)
        if pressure is not None:
            b = self._pressure_right(b,pressure)
        u = tuple(r/d for r,d in zip(zero_normal(b),self._helmholtz_diagonal))
        for _ in range(self.helmholtz_iterations):
            u = self._helmholtz_sweep(u,b)
        return u

    def helmholtz(self,b):
        """Owned velocity result, valid after later solves and time-step changes."""
        return self._solve_helmholtz(b)

    def helmholtz_summary(self):
        return dict(backend=self.helmholtz_backend,calls=self.helmholtz_calls,
            sweeps=self.helmholtz_sweeps,workspace=None if self._helmholtz_workspace is None
            else self._helmholtz_workspace.summary())

    def _right(self,old,advection,density):
        return zero_normal(tuple(u+self.alpha*l+self.dt*(-a+f/self.rho)
                           for u,l,a,f in zip(old,self._lap(old),advection,density)))

    def _stokes_metrics(self,u,p,b):
        gp = gradient(p,self.grid.spacing)
        momentum = tuple(v-self.alpha*l+self.dt/self.rho*g-r
                         for v,l,g,r in zip(u,self._lap(u),gp,b))
        # Boundary-normal rows are prescribed; G is zero there.
        return torch.stack((torch.sqrt(sum(v.square().sum() for v in momentum)),
                            torch.linalg.vector_norm(divergence(u,self.grid.spacing))))

    @torch.no_grad()
    def stokes(self,b,initial=None):
        """Solve H u + dt/rho G p=b, D u=0 with true residual acceptance."""
        self.stokes_calls += 1
        b = zero_normal(b)
        opt = self.pressure_solver.options
        rhs_norm = torch.sqrt(sum(v.square().sum() for v in b)).item()
        momentum_tol = max(opt.atol,opt.rtol*rhs_norm)
        p = self.pressure_solver.diagonals[0].new_zeros(self.grid.shape) if initial is None else initial.clone()
        if p.shape != self.grid.shape or not torch.isfinite(p).all():
            raise ValueError('invalid initial CN Stokes pressure')
        p -= p.mean()
        cycles = 0
        poisson_solves = 0
        # Scale the continuity tolerance exactly like the old Poisson RHS check.
        z = self._solve_helmholtz(b,owned=False)
        continuity_rhs = -self.rho/self.dt*divergence(z,self.grid.spacing)
        pressure_tol = max(opt.atol,opt.rtol*torch.linalg.vector_norm(continuity_rhs).item())
        div_tol = self.dt/self.rho*pressure_tol
        # Roundoff floor is bounded by the norm of the discrete D operator.
        floor = 100*torch.finfo(p.dtype).eps*max(rhs_norm,torch.finfo(p.dtype).tiny)*sum(1/h for h in self.grid.spacing)
        div_tol = max(div_tol,floor)
        del z,continuity_rhs
        for iteration in range(self.cnab_options.max_stokes_iterations+1):
            u = self._solve_helmholtz(b,p,owned=False)
            momentum,div = self._metrics(u,p,b).tolist()
            if not isfinite(momentum+div):
                raise RuntimeError('nonfinite CN Stokes residual')
            if momentum <= momentum_tol and div <= div_tol:
                if self._helmholtz_workspace is not None:
                    u = tuple(v.clone() for v in u)
                return MACFlowResult(u,p,dict(pressure=dict(cycles=cycles,backend=self.pressure_solver.backend,
                    poisson_solves=poisson_solves,schur_iterations=iteration,residual_norm=self.rho/self.dt*div,
                    tolerance=self.rho/self.dt*div_tol,meaning='CN half-time pressure; true Schur residual'),
                    stokes=dict(momentum_residual=momentum,momentum_tolerance=momentum_tol,
                    divergence_norm=div,divergence_tolerance=div_tol,
                    helmholtz_iterations=self.helmholtz_iterations,helmholtz_backend=self.helmholtz_backend)))
            if iteration == self.cnab_options.max_stokes_iterations:
                break
            residual = -self.rho/self.dt*divergence(u,self.grid.spacing)
            # H enforces exactly zero boundary-normal velocity. The discrete
            # D sum telescopes to zero; remove only its floating-point mean.
            # Near convergence, cancellation of large velocities can exceed
            # the generic Poisson compatibility screen's absolute threshold.
            residual -= residual.mean()
            # Approximate Schur inverse A0^-1 + alpha I. Unlike commuting
            # factorization, every correction is followed by an actual H solve.
            delta,info = self.pressure_solver.solve(residual)
            cycles += info['cycles']; poisson_solves += 1
            p += delta+self.alpha*residual
            p -= p.mean()
        error = RuntimeError(f'CN Stokes failed: momentum {momentum:.6g}/{momentum_tol:.6g}, divergence {div:.6g}/{div_tol:.6g}')
        error.diagnostics = dict(stage='CN Stokes',momentum_residual=momentum,momentum_tolerance=momentum_tol,
                                 divergence_norm=div,divergence_tolerance=div_tol,schur_iterations=iteration)
        raise error

    @torch.no_grad()
    def linear_response(self,b):
        """Homogeneous CN Stokes action, normalized for nested Krylov solves."""
        b = zero_normal(b)
        scale = torch.sqrt(sum(v.square().sum() for v in b)).item()
        if not isfinite(scale):
            raise ValueError('nonfinite CN linear-response RHS')
        if scale==0:
            return tuple(torch.zeros_like(v) for v in b)
        response = self.stokes(tuple(v/scale for v in b))
        return tuple(scale*v for v in response.velocity)

    def check_transport(self,velocity,density):
        finite,C,Re,A = self._step_checks(velocity,density).tolist()
        if not finite:
            raise ValueError('finite CNAB velocity and force density required')
        if C > COURANT_LIMIT:
            raise CNABTransportGuardError(dict(courant=C,cell_reynolds=Re,advection_diffusion_number=A,
                component_max_abs_velocity=[v.abs().max().item() for v in velocity],
                dt=self.dt,spacing=self.grid.spacing,viscous_number=self.viscous_number))
        return dict(courant=C,cell_reynolds=Re,advection_diffusion_number=A,viscous_number=self.viscous_number)

    @torch.no_grad()
    def advance(self,velocity,density,advection,pressure_initial=None):
        self.grid.check_velocity(velocity); self.grid.check_velocity(density)
        self.grid.check_velocity(advection)
        numbers = self.check_transport(velocity,density)
        result = self.stokes(self._right(velocity,advection,density),pressure_initial)
        return MACFlowResult(result.velocity,result.pressure,dict(result.diagnostics,**numbers))


class MidpointMACIBStepper(MACIBStepper):
    """Explicit midpoint structural force with CN--AB2 fluid, not Newton FSI."""

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.optimized:
            from .execution import tensor_kernel
            self._force_checks = tensor_kernel(self._force_checks,self.solid_execution.model.mesh.X.device)

    def _force_checks(self,flags,force,points):
        return torch.stack((flags.all(),self.transfer._support_flags(points),torch.isfinite(force).all()))

    def _force_geometry(self,x,time):
        if self.optimized:
            # Evaluate geometry once; aggregate validity/support/force checks
            # into a single scalar read, using the existing compiled kernels.
            force,geometry = self.solid_execution.force_with_geometry(x,time)
            points = (self.transfer.validation_points(x) if hasattr(self.transfer,'validation_points') else
                      self.transfer.evaluate(x).reshape(-1,3))
            valid,support,finite = self._force_checks(geometry[-1],force,points).tolist()
            if not valid:
                raise ValueError('invalid midpoint/end-point solid geometry')
            if not support:
                raise ValueError('MAC IB support reaches a wall or is nonfinite; enlarge/refine the fluid box')
            if not finite:
                raise ValueError('nonfinite midpoint/end-point solid force')
            self.solid_execution.remember_geometry(x,geometry)
        else:
            self.validate(x)
            self.transfer.check_support(self.transfer.interaction_points(x))
            force = self.force(x,time)
            if not torch.isfinite(force).all():
                raise ValueError('nonfinite midpoint/end-point solid force')
        return force

    @torch.no_grad()
    def step(self,state,*,diagnostics=True):
        dt = self.flow.dt
        if type(state.step) is not int or state.step < 0 or not isfinite(state.time) or abs(state.time-state.step*dt)>1e-12:
            raise ValueError('inconsistent CNAB state clock')
        self.validate(state.x)
        stencil_n = self.transfer.prepare(state.x)
        U_n,old_mass = self.transfer.interpolate(state.velocity,stencil_n)
        half_time,target = state.time+.5*dt,(state.step+1)*dt
        current_adv = self.flow.advection(state.velocity)
        startup = state.previous_advection is None
        startup_info = None
        if startup:
            # Appendix A.2: old force -> provisional CN Stokes solve, then
            # average provisional x/u to evaluate the corrected half-time RHS.
            force_n = self._force_geometry(state.x,state.time)
            density_n,_ = self.transfer.spread(force_n,stencil_n)
            provisional = self.flow.advance(state.velocity,density_n,current_adv,state.pressure)
            x_half = state.x+.5*dt*U_n
            half_adv = self.flow.advection(blend(state.velocity,provisional.velocity))
            startup_info = provisional.diagnostics
        else:
            self.flow.grid.check_velocity(state.previous_advection)
            x_half = state.x+.5*dt*U_n
            half_adv = blend(current_adv,state.previous_advection,1.5,-.5)
        half_force = self._force_geometry(x_half,half_time)
        stencil_half = self.transfer.prepare(x_half)
        density,spread_info = self.transfer.spread(half_force,stencil_half)
        flow = self.flow.advance(state.velocity,density,half_adv,state.pressure)
        average = blend(state.velocity,flow.velocity)
        U_half,interpolation_info = self.transfer.interpolate(average,stencil_half)
        x_new = state.x+dt*U_half
        fraction = ((x_new-state.x).abs()/x_new.new_tensor(self.flow.grid.spacing)).max().item()
        if not isfinite(fraction) or fraction > self.max_displacement:
            raise ValueError('CNAB solid displacement exceeds grid-based limit; reduce dt')
        force_new = self._force_geometry(x_new,target)
        new = MACState(state.step+1,target,x_new,flow.velocity,flow.pressure,force_new,target,
                       tuple(a.detach().clone() for a in current_adv))
        # No driver history is committed before acceptance; failed trials keep
        # state/history intact and can be saved or retried reproducibly.
        info = dict(flow=flow.diagnostics,force_mass=asdict(spread_info),velocity_mass=asdict(interpolation_info),
            predictor_mass=asdict(old_mass),max_grid_displacement=fraction,used_force_time_s=half_time,
            next_force_time_s=target,time_integrator='CN-AB2/midpoint-IB',startup_predictor_corrector=startup,
            startup_flow=startup_info,ib_geometry='predicted midpoint',force_sampling='midpoint')
        if diagnostics:
            solid_power = (U_half*half_force).sum()
            fluid_power = self.flow.grid.volume*sum((u*f).sum() for u,f in zip(average,density))
            info.update(solid_power=solid_power.item(),fluid_power=fluid_power.item(),
                power_error=abs((solid_power-fluid_power).item()),
                divergence_l2=(self.flow.grid.volume*divergence(flow.velocity,self.flow.grid.spacing).square().sum()).sqrt().item())
        return new,info
