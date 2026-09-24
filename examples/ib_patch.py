"""Solid Guccione/active/boundary forces -> IB -> imposed velocity transfer.

This is an instantaneous coupling check. The velocity is prescribed, not a
Navier-Stokes solution; no physical coupled time step is claimed.
"""
import argparse
import json
import torch
from afsi_torch import boundary as bd, solid, ib
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from afsi_torch.tetrahedron import reference_nodes


def run(device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    X = reference_nodes(device=device)
    cells = torch.arange(10, device=device).reshape(1, 10)
    geometry = solid.prepare_p2(X, cells)
    fields = prepare_reference_fields(geometry, [1., 0., 0.], [0., 1., 0.], 1000.)
    parameters = GuccioneParameters()
    faces = bd.extract_boundary(X, cells)
    base = bd.prepare_surface(X, faces[(X[faces[:, :3], 2].abs() < 1e-12).all(1)])
    loaded = bd.prepare_surface(X, faces[(X[faces[:, :3]].sum(-1) > .9).all(1)])
    x = X+.03*X.square()
    x[:, 2] += .02*X[:, 0].square()
    solid.validate_deformation(x, geometry)
    bd.validate_surface(x, loaded)
    bd.validate_surface(x, base)
    force = (solid.guccione_force(x, geometry, fields, parameters)
             +bd.pressure_force(x, loaded, 1200.)+bd.spring_force(x, base, 5000.))
    grid = ib.UniformGrid((13, 13, 13), (.2, .2, .2), (-.6, -.6, -.6))
    fluid_nodes = grid.coordinates(device=device)
    stencil = ib.prepare_stencil(x, grid)
    u = .1*fluid_nodes.sin()  # Imposed test field, no fluid solve.
    velocity = ib.interpolate(u, stencil)
    density = ib.spread_density(force, stencil)
    power_solid = (force*velocity).sum()
    power_fluid = (u*density).sum()*grid.cell_volume
    force_error = density.sum(0)*grid.cell_volume-force.sum(0)
    torque_error = (torch.linalg.cross(fluid_nodes, density).sum(0)*grid.cell_volume
                    -torch.linalg.cross(x, force).sum(0))
    torch.testing.assert_close(power_fluid, power_solid, atol=1e-8, rtol=1e-12)
    torch.testing.assert_close(force_error, torch.zeros_like(force_error), atol=1e-8, rtol=0)
    torch.testing.assert_close(torque_error, torch.zeros_like(torque_error), atol=1e-8, rtol=0)
    return dict(status='passed', device=str(device), torch=torch.__version__,
                solid_nodes=len(X), fluid_velocity_nodes=grid.node_count, neighbors_per_node=64,
                force_balance_max_abs=force_error.abs().max().item(),
                torque_balance_max_abs=torque_error.abs().max().item(),
                power_error=abs((power_fluid-power_solid).item()),
                power_relative_error=abs((power_fluid-power_solid).item())/max(1., abs(power_solid.item())),
                solid_velocity_max_abs=velocity.abs().max().item(),
                fluid_solver=False, time_stepping=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    print(json.dumps(run(parser.parse_args().device), indent=2))
