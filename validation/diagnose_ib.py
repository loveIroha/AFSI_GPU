"""Frozen-LV factorial probes: kernel width, FE load mapping and projection.

Diagnostic alternatives only. The production IB/Chorin/coupling are unchanged.
Same geometry, force and zero initial fluid velocity in every probe; this is
not a trajectory, timestep convergence study or a physiological flow solution.
"""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from math import isfinite
from pathlib import Path
import platform
import numpy as np
import torch
from afsi_torch import ib
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
from afsi_torch.fluid.solvers import pcg, SolverOptions
from afsi_torch.geometry import LVConfig, generate_lv, cavity_volume
from afsi_torch.lv_model import LVSolid
from afsi_torch.preload import load_preload
from afsi_torch.units import CGS_UNITS


def scaled_stencil(x, grid, dilation):
    """Diagnostic phi((x-X)/epsilon)*h/epsilon on each axis, epsilon=m*h.

    Integer m only: sum over m shifted sublattices preserves the Peskin
    zeroth/first moments. No weight renormalization and no clipped support.
    m=1 delegates exactly to the production kernel. Cost grows as (4*m)^3.
    """
    if not isinstance(dilation, int) or isinstance(dilation, bool) or dilation < 1:
        raise ValueError('positive integer kernel dilation required')
    if dilation == 1:
        return ib.prepare_stencil(x, grid)
    ib.prepare_stencil(x, grid)  # includes field/device-independent validation
    scaled = (x-x.new_tensor(grid.origin))/x.new_tensor(grid.spacing)
    m = dilation
    base = torch.floor(scaled-2*m+1).to(torch.int64)
    if (base < 0).any() or (base+4*m > x.new_tensor(grid.shape)).any():
        raise ValueError('fixed-width IB support leaves fluid grid')
    a = torch.arange(4*m, device=x.device)
    nodes = base[:, None, :]+torch.cartesian_prod(a, a, a)[None, :, :]
    w = (ib.peskin4((scaled[:, None, :]-nodes.to(x.dtype))/m)/m).prod(-1)
    nx, ny, _ = grid.shape
    return ib.IBStencil(grid, nodes[..., 0]+nx*(nodes[..., 1]+ny*nodes[..., 2]), w)


class BoxMassInverse:
    """Exact free-velocity mass inverse for this tensor-product Q2 box.

    Zero outer Dirichlet values. Three small 1D Cholesky solves on the active
    device; no global dense matrix or CPU solve. Used only as a reference.
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.factors = []
        X = mesh.velocity_coordinates
        template = X.new_tensor([[4., 2., -1.], [2., 16., 2.], [-1., 2., 4.]])/30
        for n, h in zip(mesh.counts, mesh.cell_sizes):
            M = X.new_zeros((2*n+1, 2*n+1))
            for e in range(n):
                M[2*e:2*e+3, 2*e:2*e+3] += h*template
            self.factors.append(torch.linalg.cholesky(M[1:-1, 1:-1]))

    def __call__(self, rhs):
        nx, ny, nz = self.mesh.velocity_grid.shape
        b = rhs.reshape(nz, ny, nx, 3)[1:-1, 1:-1, 1:-1].clone()
        for axis, L in zip((2, 1, 0), self.factors):
            v = b.movedim(axis, 0)
            shape = v.shape
            b = torch.cholesky_solve(v.reshape(shape[0], -1), L).reshape(shape).movedim(0, axis)
        result = torch.zeros_like(rhs).reshape(nz, ny, nx, 3)
        result[1:-1, 1:-1, 1:-1] = b
        return result.reshape(-1, 3)


@torch.no_grad()
def discrete_projection(flow, star, mass_inverse):
    """Reference M-orthogonal projection onto D u=0 with zero outer velocity.

    S = D M_free^-1 D.T, S lambda = D star,
    u = star - M_free^-1 D.T lambda. Q1 constraint, NOT pointwise div u=0.
    lambda is a projection multiplier, not reported as physical pressure.
    """
    op = flow.op
    if star[flow.velocity_fixed].abs().max().item() > 1e-12:
        raise ValueError('reference projection requires zero boundary velocity')
    action = lambda p: op.divergence(mass_inverse(op.divergence_transpose(p)))
    rhs = op.divergence(star)
    multiplier, info = pcg(action, rhs, flow.diagonals['pressure_stiffness'],
        fixed=flow.pressure_fixed, options=SolverOptions(rtol=1e-12, atol=1e-14, max_iterations=2000))
    velocity = star-mass_inverse(op.divergence_transpose(multiplier))
    # Check the omitted gauge equation as well; never hide incompatibility.
    residual = torch.linalg.vector_norm(op.divergence(velocity)).item()
    if residual > 20*len(rhs)**.5*info.tolerance:
        raise RuntimeError('reference projection full divergence residual failed')
    return velocity, dict(solve=asdict(info), full_divergence_dual_norm=residual)


def relative_norm(a, b):
    denominator = torch.linalg.vector_norm(b).item()
    return None if denominator <= 1e-14 else (torch.linalg.vector_norm(a-b).item()/denominator)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


@torch.no_grad()
def run(device='cpu', output='results/ib_diagnosis', levels=(6, 12, 18), dt=2.5e-5,
        preload=None, pressure_increment=.02):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if (len(levels) < 2 or levels[0] < 6 or any(not isinstance(n, int) or n % levels[0] for n in levels)
            or list(levels) != sorted(set(levels))):
        raise ValueError('use increasing integer multiples of a base cell count >= 6')
    if preload is not None and (not isfinite(pressure_increment) or pressure_increment<=0):
        raise ValueError('positive finite pressure increment required for preload probe')
    # ChorinSolver validates dt before any case is accepted.
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    if preload is None:
        model = LVSolid(generate_lv(LVConfig(mesh_size=1.2), device=device))
        x = model.mesh.X
        force = model.force(x, .001).detach()
        provenance = None
        initial_force = None
    else:
        model,x,tolerance,provenance = load_preload(preload,device=device)
        initial_force = model.force(x,0.).detach()
        original_loads=model.loads
        try:
            model.loads=replace(original_loads,pressure_mmhg=original_loads.pressure_mmhg+pressure_increment)
            force=(model.force(x,0.)-initial_force).detach()
        finally:
            model.loads=original_loads
        if torch.linalg.vector_norm(initial_force).item()>tolerance:
            raise ValueError('preload is not balanced under its baseline load')
    with torch.enable_grad():
        volume_gradient = torch.func.grad(lambda y: cavity_volume(y, model.cavity))(x).detach()
    root = Path(__file__).resolve().parents[1]
    code = hashlib.sha256()
    for path in [Path(__file__), *sorted((root/'src'/'afsi_torch').rglob('*.py'))]:
        code.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes())
    report = dict(metadata=dict(device=str(device), torch=torch.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        units=CGS_UNITS, levels=levels, dt_s=dt, load_time_s=.001 if preload is None else None,
        solid_mesh_size_cm=model.mesh.config.mesh_size,
        solid_nodes=len(x), solid_cells=len(model.mesh.cells), gmsh=model.mesh.gmsh_version,
        loads=asdict(model.loads), material=asdict(model.parameters), beta=model.beta,
        rho=1., mu=1., source_sha256=code.hexdigest(), fixed_kernel_epsilon_cm=6/levels[0],
        production_algorithm_changed=False, frozen_geometry=True, initial_velocity='zero',
        domain_lengths_cm=[12.]*3, domain_origin_cm=[-6.,-6.,-8.]),
        cases={}, comparisons={}, completed=False, full_cycle_ready=False,
        limitations=['frozen one-step response, not the coupled trajectory',
          'fixed physical kernel and dual load and Schur projection are diagnostic alternatives',
          'weak Q1 divergence constraint does not imply pointwise incompressibility',
          'nodal norms compare the identical solid mesh; they are not volume-weighted L2 norms',
          'no single-factor attribution of the full trajectory error or convergence proof'])
    if preload is not None:
        report['metadata']['preload']=dict(provenance=provenance,
            baseline_pressure_mmhg=original_loads.pressure_mmhg,
            pressure_increment_mmhg=pressure_increment,
            initial_force_norm_dyn=torch.linalg.vector_norm(initial_force).item(),
            incremental_force_norm_dyn=torch.linalg.vector_norm(force).item(),
            probe_force='target pressure force minus balanced baseline force at fixed preloaded x',
            reference_rebased=False)
    np.savez_compressed(folder/'solid_probe.npz', X=model.mesh.X.cpu().numpy(),
                        x_preload=x.cpu().numpy(),cells=model.mesh.cells.cpu().numpy(),
                        force=force.cpu().numpy(), volume_gradient=volume_gradient.cpu().numpy())
    arrays = {}
    dump(folder/'diagnosis.json', report)
    for n in levels:
        print(f'Frozen LV probe: fluid {n}^3', flush=True)
        mesh = create_box((n,)*3, (12.,)*3, (-6.,-6.,-8.), device=device)
        op = prepare_operators(mesh)
        flow = ChorinSolver(op, dt=dt)
        inverse = BoxMassInverse(mesh)
        grid = mesh.velocity_grid
        for mode in ('native', 'fixed'):
            m = 1 if mode == 'native' else n//levels[0]
            stencil = scaled_stencil(x, grid, m)
            partition = (stencil.weights.sum(-1)-1).abs().max().item()
            first = (ib.interpolate(mesh.velocity_coordinates, stencil)-x).abs().max().item()
            if partition > 1e-12 or first > 1e-11:
                raise RuntimeError('kernel moment identity failed')
            dual = ib.spread_load(force, stencil)
            density_rhs = op.density_load(dual/grid.cell_volume)
            for path in ('density', 'dual'):
                name = f'{mode}_{path}_{n}'
                print('  '+name, flush=True)
                try:
                    flow_result = flow.step(torch.zeros_like(mesh.velocity_coordinates),
                        **({'density':dual/grid.cell_volume} if path == 'density' else {'nodal_load':dual}))
                    star = flow_result.tentative_velocity
                    exact, exact_info = discrete_projection(flow, star, inverse)
                    star_div = torch.linalg.vector_norm(op.divergence(star)).item()
                    velocities = dict(tentative=star, chorin=flow_result.velocity, schur=exact)
                    record = dict(completed=True, cells_per_axis=n, kernel=mode, load_path=path,
                        epsilon_cm=m*grid.spacing[0], kernel_support_radius_cm=2*m*grid.spacing[0],
                        partition_error=partition, affine_interpolation_error_cm=first,
                        fe_density_vs_dual_load_relative_norm=relative_norm(density_rhs, dual),
                        solver_options=asdict(flow.options), chorin=flow_result.diagnostics,
                        schur=exact_info, responses={})
                    save = {}
                    for projection, velocity in velocities.items():
                        U = ib.interpolate(velocity, stencil)
                        arrays[(mode,path,n,projection)] = U.cpu()
                        save[projection+'_solid_velocity'] = U.cpu().numpy()
                        weak = torch.linalg.vector_norm(op.divergence(velocity)).item()
                        power = (U*force).sum().item()
                        lattice_power = (velocity*dual).sum().item()
                        fe_power = (velocity*density_rhs).sum().item()
                        record['responses'][projection] = dict(
                            solid_velocity_nodal_norm_cm_per_s=torch.linalg.vector_norm(U).item(),
                            max_solid_speed_cm_per_s=torch.linalg.vector_norm(U, dim=-1).max().item(),
                            cavity_rate_ml_per_s=(U*volume_gradient).sum().item(),
                            weak_divergence_relative_to_tentative=None if star_div < 1e-14 else weak/star_div,
                            divergence_l2=flow.divergence_l2(velocity),
                            solid_power_erg_per_s=power, lattice_power_error_erg_per_s=abs(power-lattice_power),
                            density_path_fe_minus_solid_power_erg_per_s=fe_power-power)
                    record['chorin_vs_schur_solid_velocity_relative_norm'] = relative_norm(
                        arrays[(mode,path,n,'chorin')], arrays[(mode,path,n,'schur')])
                    np.savez_compressed(folder/(name+'.npz'), **save)
                    report['cases'][name] = record
                except (ValueError, RuntimeError) as exc:
                    report['cases'][name] = dict(completed=False, error_type=type(exc).__name__, message=str(exc))
                    dump(folder/'diagnosis.json', report)
                    raise
                dump(folder/'diagnosis.json', report)
    for mode in ('native', 'fixed'):
        for path in ('density', 'dual'):
            for projection in ('tentative', 'chorin', 'schur'):
                name = f'{mode}_{path}_{projection}'
                report['comparisons'][name] = [dict(coarse=a, fine=b,
                    solid_velocity_difference_norm_cm_per_s=torch.linalg.vector_norm(
                        arrays[(mode,path,a,projection)]-arrays[(mode,path,b,projection)]).item(),
                    fine_solid_velocity_norm_cm_per_s=torch.linalg.vector_norm(arrays[(mode,path,b,projection)]).item(),
                    solid_velocity_relative_difference=relative_norm(arrays[(mode,path,a,projection)], arrays[(mode,path,b,projection)]))
                    for a,b in zip(levels,levels[1:])]
    report['completed'] = True
    dump(folder/'diagnosis.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/ib_diagnosis')
    parser.add_argument('--levels', type=int, nargs='+', default=[6,12,18])
    parser.add_argument('--dt', type=float, default=2.5e-5)
    parser.add_argument('--preload', default=None)
    parser.add_argument('--pressure-increment-mmhg', type=float, default=.02)
    args = parser.parse_args()
    result = run(args.device, args.output, args.levels, args.dt,args.preload,args.pressure_increment_mmhg)
    print(json.dumps(result, indent=2, allow_nan=False))
