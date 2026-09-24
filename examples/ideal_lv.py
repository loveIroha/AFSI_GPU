"""Generate a centimetre-based ideal LV and verify its solid force evaluation.

Coordinates are prescribed for verification; this is not a static equilibrium
solve, a fluid solve, or a cardiac cycle. No input mesh/fiber files are read.
"""
import argparse
import json
import torch
from afsi_torch import solid, boundary as bd
from afsi_torch.geometry import (LVConfig, generate_lv, ENDO, EPI, BASE,
    rule_based_fibers, signed_cell_volumes, prepare_cavity, cavity_volume)
from afsi_torch.geometry.output import write_lv
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from afsi_torch.units import MMHG_TO_DYN_PER_CM2, CGS_UNITS


def run(device='cpu', mesh_size=1.2, output='results/ideal_lv'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    config = LVConfig(mesh_size=mesh_size)
    mesh = generate_lv(config, device=device)
    X = mesh.X
    geometry = solid.prepare_p2(X, mesh.cells)
    fibers = rule_based_fibers(X, config)
    fields = prepare_reference_fields(geometry, fibers.fiber, fibers.sheet, 1000.)
    parameters = GuccioneParameters()  # C/kappa in dyn/cm^2 in this CGS example.
    endo = bd.prepare_surface(X, mesh.surface(ENDO))
    base = bd.prepare_surface(X, mesh.surface(BASE))
    cavity = prepare_cavity(X, mesh.surface(ENDO))
    relative = X-X.new_tensor(config.center)
    x = X+relative*X.new_tensor([.02, .02, -.01])
    x[:, 2] += .01*config.base_height
    solid.validate_deformation(x, geometry)
    bd.validate_surface(x, endo)
    bd.validate_surface(x, base)
    p = 8*MMHG_TO_DYN_PER_CM2
    beta = 5e5  # dyn/cm^3, same numerical beta as afsi demo_337.
    bulk = solid.guccione_force(x, geometry, fields, parameters)
    pressure = bd.pressure_force(x, endo, p)
    spring = bd.spring_force(x, base, beta)
    force = bulk+pressure+spring
    energy = lambda y: (solid.guccione_energy(y, geometry, fields, parameters)
                         +bd.spring_energy(y, base, beta))
    conservative_force = -torch.func.grad(energy)(x)
    torch.testing.assert_close(conservative_force, bulk+spring, atol=2e-6, rtol=1e-9)
    ref_volume = cavity_volume(X, cavity)
    current_volume = cavity_volume(x, cavity)
    torch.testing.assert_close(current_volume, ref_volume*(1.02**2*.99), atol=1e-10, rtol=1e-11)
    cell_volumes = signed_cell_volumes(X, mesh.cells)
    if not (cell_volumes > 0).all() or not torch.isfinite(force).all():
        raise RuntimeError('invalid reference cells or nonfinite solid force')
    report = dict(status='passed', device=str(device), torch=torch.__version__,
        gmsh=mesh.gmsh_version, units=CGS_UNITS, mesh_size_cm=mesh_size,
        cells=len(mesh.cells), p1_vertices=mesh.vertex_count, p2_nodes=len(X),
        boundary_faces={name: len(mesh.surface(tag)) for name, tag in [('ENDO', ENDO), ('EPI', EPI), ('BASE', BASE)]},
        minimum_cell_volume_cm3=cell_volumes.min().item(),
        wall_volume_cm3=cell_volumes.sum().item(), analytic_wall_volume_cm3=config.wall_volume,
        cavity_volume_ml=ref_volume.item(), analytic_cavity_volume_ml=config.cavity_volume,
        cavity_geometry_relative_error=abs(ref_volume.item()/config.cavity_volume-1),
        prescribed_cavity_volume_ml=current_volume.item(),
        pressure_mmhg=8., pressure_dyn_per_cm2=p, tension_dyn_per_cm2=1000., beta_dyn_per_cm3=beta,
        minimum_detF=torch.linalg.det(solid.deformation_gradient(x, geometry)).min().item(),
        conservative_force_gradient_error_dyn=(conservative_force-bulk-spring).abs().max().item(),
        total_force_norm_dyn=torch.linalg.vector_norm(force).item(),
        minimum_quadrature_fiber_sheet_cross_norm=torch.linalg.vector_norm(fields.normal, dim=-1).min().item(),
        apical_regularization_nodes=int((fibers.apical_weight > 0).sum()),
        equilibrium_solved=False, fluid_solver=False, time_stepping=False)
    if output:
        write_lv(output, mesh, fibers, current=x, force=force, report=report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mesh-size', type=float, default=1.2, help='target tetrahedron edge scale in cm')
    parser.add_argument('--output', default='results/ideal_lv')
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.mesh_size, args.output), indent=2))
