"""Explicit settings for demo_337's active contraction script, in CGS units.

The generated geometry follows its companion benchmark recipe. It is NOT
asserted identical to the external mesh/fiber files used by the AFSI run.
"""
from dataclasses import dataclass
from math import isfinite
import torch
from .geometry import LVConfig, generate_lv
from .lv_model import LVSolid

SOURCE = 'https://github.com/loveIroha/afsi/blob/main/afsic/demo/demo_337/fsi_paralell_fibers_contraction.py'


@dataclass(frozen=True)
class AFSI337Loads:
    pressure: float = 150000.  # dyn/cm^2
    tension: float = 600000.
    ramp_time: float = 1.5

    def __post_init__(self):
        if (not all(isfinite(v) for v in (self.pressure, self.tension, self.ramp_time)) or
                min(self.pressure, self.tension) < 0 or self.ramp_time <= 0):
            raise ValueError('finite nonnegative loads and positive ramp time required')

    def at(self, time):
        if not isfinite(time) or time < 0:
            raise ValueError('finite nonnegative load time required')
        scale = min(time/self.ramp_time, 1.)
        return self.pressure*scale, self.tension*scale


def geometry_config(mesh_size=.1):
    return LVConfig(inner_axes=(.7, .7, 1.7), outer_axes=(1., 1., 2.),
                    base_height=.5, center=(3.5, 2.5, 2.5), long_axis='x', mesh_size=mesh_size)


def default_quadrature(device='cpu'):
    """Basix 0.10 default degree-4 simplex rules, without a Basix runtime dependency.

    Native input can instead carry the rules exported by its installed Basix.
    Explicit point tables matter for the exponential constitutive law.
    """
    q = [[0,.5,.5],[.5,0,.5],[.5,.5,0],[.5,0,0],[0,.5,0],[0,0,.5],
         [.6984197043243866,.1005267652252045,.1005267652252045],
         [.1005267652252045,.1005267652252045,.1005267652252045],
         [.1005267652252045,.1005267652252045,.6984197043243866],
         [.1005267652252045,.6984197043243866,.1005267652252045],
         [.0568813795204234,.3143728734931922,.3143728734931922],
         [.3143728734931922,.3143728734931922,.3143728734931922],
         [.3143728734931922,.3143728734931922,.0568813795204234],
         [.3143728734931922,.0568813795204234,.3143728734931922]]
    w = [.003174603174603167]*6+[.014764970790496783]*4+[.022139791114265117]*4
    s = [[.816847572980459,.091576213509771],[.091576213509771,.816847572980459],
         [.091576213509771,.091576213509771],[.10810301816807,.445948490915965],
         [.445948490915965,.10810301816807],[.445948490915965,.445948490915965]]
    sw = [.054975871827661]*3+[.1116907948390055]*3
    cast = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    return (cast(q), cast(w)), (cast(s), cast(sw))


def generated_model(*, mesh_size=.1, device='cpu', geometry=None, parameters=None, loads=None, beta=5e5,
                    basal_constraint='spring'):
    from .geometry.ellipsoid_fibers import laplace_ellipsoid_fibers
    mesh = generate_lv(geometry_config(mesh_size) if geometry is None else geometry, device=device)
    fibers, info = laplace_ellipsoid_fibers(mesh)
    vq, sq = default_quadrature(device)
    return LVSolid(mesh, loads=AFSI337Loads() if loads is None else loads, parameters=parameters,
                   beta=beta, basal_constraint=basal_constraint, fibers=fibers, volume_quadrature=vq,
                   surface_quadrature=sq, fiber_metadata=dict(
                       mode='generated', recipe='P1 Laplace -> nodal P2 ellipsoidal helix',
                       endo_angle_degrees=90., epi_angle_degrees=-90., laplace=info,
                       external_fields_matched=False))


def alignment(model, settings):
    native = model.fiber_metadata.get('mode') == 'native'
    return dict(source=SOURCE, demo_blob='283b23f5155dbc57043edd2aa7280d61c3c8e985',
        protocol='simultaneous pressure/tension linear ramp to 1.5 s, then hold; horizon 2 s',
        reference_dt_s=5e-5, reference_fluid_cells=32, reference_fluid_box_cm=[0.,5.],
        fluid_mesh_matched=(tuple(settings.get('fluid_shape',(settings['fluid_cells'],)*3)) == (32,)*3
                            and tuple(settings.get('fluid_lengths',(settings['box_length'],)*3)) == (5.,)*3
                            and tuple(settings['origin']) == (0.,0.,0.)),
        dt_matched=settings['dt'] == 5e-5,
        solid_source='exported AFSI mesh and P2 fields' if native else 'generated companion benchmark geometry',
        solid_mesh_and_fibers_matched=native,
        fiber_details=model.fiber_metadata,
        quadrature_points=dict(tetrahedron=len(model.geometry.weights[0]), triangle=len(model.endo.quadrature_weights)),
        boundary_conditions=f'outer fluid no-slip; pressure gauge origin; BASE {model.basal_constraint_mode} penalty beta={model.beta:g}; EPI free',
        basal_constraint_matched=(model.basal_constraint_mode=='spring' and model.beta==5e5),
        differences=([] if model.basal_constraint_mode=='spring' and model.beta==5e5 else [
            'BASE constraint differs from native AFSI demo_337 three-direction spring beta=500000.'])+([] if native else [
            'Original XDMF/f0/s0/cdm are external; generated solid connectivity/size not verified identical.',
            'Generated fibers evaluate Laplace-based ellipsoid recipe at P2 nodes; external projected coefficients may differ.']),
        solver_difference='PyTorch CSR Jacobi-PCG replaces native PETSc linear solvers',
        diagnostic_definition='AFSI volume is wall integral of detF; cavity volume is an additional output')
