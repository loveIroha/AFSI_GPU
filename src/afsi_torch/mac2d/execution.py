"""Build reference or optimized execution of the same valve discretization."""
from .grid import ChannelGrid
from .flow import ChannelFlow
from .transfer import TriangleTransfer
from .coupling import ValveStepper


def build_driver(solid,settings,device):
    backend=settings.get('execution_backend','optimized')
    if backend not in ('reference','optimized'):
        raise ValueError('execution backend must be reference or optimized')
    pressure=settings.get('pressure_backend','auto')
    if pressure not in ('auto','reference','workspace','graph'):
        raise ValueError('invalid pressure backend')
    optimized=backend=='optimized'
    from ..mac.multigrid import MGOptions
    from ..fluid.solvers import SolverOptions
    grid=ChannelGrid((settings['nx'],settings['ny']),tuple(settings.get('fluid_lengths',(8.,solid.config.height))))
    flow=ChannelFlow(grid,dt=settings['dt'],rho=settings['rho'],mu=settings['mu'],device=device,
        fused=settings['fused'],optimized=optimized,pressure_backend=None if pressure=='auto' else pressure,
        options=MGOptions(**settings.get('pressure_solver',{})),inlet_config=settings.get('inlet'))
    transfer=TriangleTransfer(grid,solid.geometry,mass_backend=settings['mass_backend'],
        warm_start=settings['warm_start'],fused=settings['fused'],optimized=optimized,
        options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None)
    return ValveStepper(flow,transfer,solid,optimized=optimized,fused=settings['fused'])
