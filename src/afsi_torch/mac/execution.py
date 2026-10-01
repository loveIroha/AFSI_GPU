"""Optional execution optimizations, independent of spatial/time discretization."""
import torch


def tensor_kernel(function, device):
    # Full graphs make accidental host checks/graph breaks fail visibly. No
    # CUDA graph output reuse: returned state tensors belong to the caller.
    if torch.device(device).type == 'cuda':
        return torch.compile(function, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    return function


def build_driver(model, settings, device):
    from .flow import MACFlow
    from .transfer import FETransfer
    from .coupling import MACIBStepper
    from ..solid import prepare_p2
    backend = settings.get('execution_backend', 'torch')
    solid_backend = settings.get('solid_backend', 'reference')
    mass_backend = settings.get('mass_backend', 'pcg')
    coupling_backend=settings.get('coupling_backend','reference')
    if backend not in ('torch', 'fused'):
        raise ValueError('execution backend must be torch or fused')
    if solid_backend not in ('reference','pointwise'):
        raise ValueError('solid backend must be reference or pointwise')
    if mass_backend not in ('pcg','graph'):
        raise ValueError('mass backend must be pcg or graph')
    if coupling_backend not in ('reference','optimized'):
        raise ValueError('coupling backend must be reference or optimized')
    if coupling_backend=='optimized' and backend!='fused':
        raise ValueError('optimized coupling backend requires fused execution')
    if backend!='fused' and (solid_backend!='reference' or mass_backend!='pcg'):
        raise ValueError('pointwise solid and graph mass backends require execution_backend=fused')
    from ..config import lv_grid
    from .multigrid import MGOptions
    from ..fluid.solvers import SolverOptions
    from .implicit import MACCouplingOptions, ImplicitMACIBStepper
    from ..config import _decode
    coupling = _decode(MACCouplingOptions, settings.get('coupling', {}))
    grid = lv_grid(settings)
    flow = MACFlow(grid, dt=settings['dt'], rho=settings['rho'], mu=settings['mu'],
                   device=device, pressure_backend=settings.get('pressure_backend','torch'),
                   execution_backend=backend, options=MGOptions(**settings.get('pressure_solver',{})),
                   implicit_transport=coupling.scheme == 'implicit-newton')
    degree = settings['interaction_degree']
    if model.mesh.cells.shape[1] == 4:
        from ..p1 import prepare_p1
        prepare_geometry = prepare_p1
    else:
        prepare_geometry = prepare_p2
    geometry = model.geometry if degree is None else prepare_geometry(model.mesh.X,model.mesh.cells,degree=degree)
    if backend == 'fused':
        from .compact_transfer import CompactFETransfer
        from .solid_execution import SolidExecution
        transfer = CompactFETransfer(grid,geometry,warm_start=settings.get('warm_start',False),
                                     mass_backend=mass_backend,
                                     options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None)
        if hasattr(model, 'execution_factory'):
            if solid_backend != 'reference':
                raise ValueError('custom solid model requires its own execution factory')
            solid = model.execution_factory()
        elif solid_backend=='pointwise':
            from .solid_pointwise import PointwiseSolidExecution
            solid = PointwiseSolidExecution(model)
        else:
            solid = SolidExecution(model)
    else:
        transfer = FETransfer(grid,geometry,warm_start=settings.get('warm_start',False),
                              options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None)
        solid = model
    if coupling.scheme == 'implicit-newton':
        from ..real_lv import RealLVSolid
        if not isinstance(model, RealLVSolid):
            raise ValueError('implicit-newton currently requires the P1 H-O real-LV model')
        return ImplicitMACIBStepper(flow,transfer,model,coupling,solid if backend=='fused' else None)
    return MACIBStepper(flow,transfer,solid.force,solid.validate,
        optimized=coupling_backend=='optimized',solid_execution=solid if backend=='fused' else None)
