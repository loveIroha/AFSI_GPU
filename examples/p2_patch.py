"""Two-cell P2 patch: energy, force and derivative verification on one device."""
import argparse
import json
import torch

from afsi_torch import solid
from afsi_torch.mechanics import determinant3
from afsi_torch.tetrahedron import promote_p1


def run(device="cpu", degree=4):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64, device=device)
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device)
    X, cells = promote_p1(vertices, cells)
    geometry = solid.prepare_p2(X, cells, degree=degree)
    x = X + .08*X.square()
    solid.validate_deformation(x, geometry)
    fn = lambda y: solid.energy(y, geometry, mu=2., lam=5.)
    force_ad = -torch.func.grad(fn)(x)
    force_pk1 = solid.stress_force(x, geometry, mu=2., lam=5.)
    torch.testing.assert_close(force_ad, force_pk1, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(force_pk1.sum(0), torch.zeros(3, device=device, dtype=x.dtype), atol=1e-11, rtol=0)
    direction = torch.sin(x)
    h = 1e-6
    fd = (fn(x+h*direction)-fn(x-h*direction))/(2*h)
    derivative = -(force_ad*direction).sum()
    torch.testing.assert_close(fd, derivative, atol=1e-8, rtol=1e-6)
    return {
        "device": str(device), "torch": torch.__version__, "dtype": str(x.dtype),
        "elements": cells.shape[0], "nodes": X.shape[0],
        "quadrature_points_per_element": geometry.values.shape[0],
        "energy": fn(x).item(),
        "min_det_F_at_quadrature": determinant3(solid.deformation_gradient(x, geometry)).min().item(),
        "max_force_difference": (force_ad-force_pk1).abs().max().item(),
        "net_force_norm": torch.linalg.vector_norm(force_pk1.sum(0)).item(),
        "directional_derivative_error": (fd-derivative).abs().item(),
        "status": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--degree", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.degree), indent=2))
