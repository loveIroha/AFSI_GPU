"""Manufactured affine velocity/pressure check; no flow solve or time step."""
import argparse
import json
import torch
from afsi_torch.fluid import create_box, prepare_operators


def run(device='cpu'):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    mesh = create_box((3, 2, 2), (1.4, 1.1, .9), (-.3, .2, -.5), device=device)
    op = prepare_operators(mesh)
    X, P = mesh.velocity_coordinates, mesh.pressure_coordinates
    A = X.new_tensor([[.4, .7, -.2], [.3, -.6, .1], [0., .5, .9]])
    b = X.new_tensor([.2, -.4, .6])
    u, p = X@A.T+.2, 1+P@b
    pairs = dict(divergence=(op.divergence(u), A.trace()*op.pressure_mass(torch.ones_like(p))),
                 pressure_gradient=(op.gradient(p), op.velocity_mass(b.expand_as(X))),
                 convection=(op.convection(u), op.velocity_mass(u@A.T)))
    errors = {}
    for name, (actual, expected) in pairs.items():
        torch.testing.assert_close(actual, expected, atol=3e-14, rtol=1e-11)
        errors[name] = (actual-expected).abs().max().item()
    return dict(status='passed', device=str(device), torch=torch.__version__,
                length_unit='cm', cells=len(mesh.velocity_cells), velocity_nodes=len(X),
                pressure_nodes=len(P), quadrature_points_per_cell=len(op.weights),
                max_abs_errors=errors, flow_solved=False, time_stepping=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    print(json.dumps(run(parser.parse_args().device), indent=2))
