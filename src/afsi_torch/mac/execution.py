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
    if coupling.scheme in ('implicit-newton','cnab-semiimplicit'):
        from ..solids.contracts import require_tangent
        require_tangent(model)
    grid = lv_grid(settings)
    from .rk3 import MACRK3Flow
    from .cnab import MACCNABFlow, MidpointMACIBStepper
    cnab = coupling.scheme in ('cnab-midpoint','cnab-semiimplicit')
    flow_class = MACCNABFlow if cnab else MACRK3Flow if coupling.scheme == 'explicit-rk3' else MACFlow
    flow_options = dict(cnab_options=coupling.cnab) if cnab else {}
    flow = flow_class(grid, dt=settings['dt'], rho=settings['rho'], mu=settings['mu'],
                   device=device, pressure_backend=settings.get('pressure_backend','torch'),
                   execution_backend=backend, options=MGOptions(**settings.get('pressure_solver',{})),
                   implicit_transport=coupling.scheme == 'implicit-newton',**flow_options)
    degree = settings['interaction_degree']
    if model.mesh.cells.shape[1] == 4:
        from ..p1 import prepare_p1
        prepare_geometry = prepare_p1
    else:
        prepare_geometry = prepare_p2
    geometry = model.geometry if degree is None else prepare_geometry(model.mesh.X,model.mesh.cells,degree=degree)
    from .adaptive_transfer import InteractionQuadratureOptions, AdaptiveP1Transfer
    quadrature = _decode(InteractionQuadratureOptions,settings.get('interaction_quadrature',{}))
    adaptive = quadrature.mode=='adaptive'
    if adaptive and (not cnab or model.mesh.cells.shape[1]!=4):
        raise ValueError('adaptive interaction requires P1 CNAB coupling')
    if backend == 'fused':
        from .compact_transfer import CompactFETransfer
        transfer_class = AdaptiveP1Transfer if adaptive else CompactFETransfer
        transfer_options = dict(quadrature_options=quadrature,fused=True) if adaptive else {}
        transfer = transfer_class(grid,geometry,warm_start=settings.get('warm_start',False),
                                     mass_backend=mass_backend,
                                     options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None,**transfer_options)
        from ..solids.contracts import make_execution
        solid = make_execution(model,solid_backend,optimized=coupling_backend=='optimized')
    else:
        transfer_class = AdaptiveP1Transfer if adaptive else FETransfer
        transfer_options = dict(quadrature_options=quadrature,fused=False,mass_backend='pcg') if adaptive else {}
        transfer = transfer_class(grid,geometry,warm_start=settings.get('warm_start',False),
                              options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None,**transfer_options)
        solid = model
    if coupling.scheme in ('implicit-newton','cnab-semiimplicit'):
        if coupling.scheme == 'cnab-semiimplicit':
            from .semiimplicit import SemiImplicitMACIBStepper
            return SemiImplicitMACIBStepper(flow,transfer,model,coupling,solid if backend=='fused' else None,
                                           optimized=coupling_backend=='optimized')
        return ImplicitMACIBStepper(flow,transfer,model,coupling,solid if backend=='fused' else None)
    stepper = MidpointMACIBStepper if coupling.scheme == 'cnab-midpoint' else MACIBStepper
    return stepper(flow,transfer,solid.force,solid.validate,
        optimized=coupling_backend=='optimized',solid_execution=solid if backend=='fused' else None)
