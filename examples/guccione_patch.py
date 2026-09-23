"""P2 Guccione + prescribed activation, using demo_337 material parameters."""
import argparse
import json
import torch

from afsi_torch import solid
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from afsi_torch.mechanics import determinant3
from afsi_torch.tetrahedron import promote_p1


def run(device="cpu", degree=4):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64, device=device)
    X, cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device=device))
    geo = solid.prepare_p2(X, cells, degree=degree)
    angle = .6*X[:, 0]
    zero = torch.zeros_like(angle)
    fiber = torch.stack((angle.cos(), angle.sin(), zero), -1)
    sheet = torch.stack((-angle.sin(), angle.cos(), zero), -1)
    fields = prepare_reference_fields(geo, fiber, sheet, 1000.+500.*X[:, 1].square())
    parameters = GuccioneParameters()
    x = X+.04*X.square()
    solid.validate_deformation(x, geo)
    fn = lambda y: solid.guccione_energy(y, geo, fields, parameters)
    residual = torch.func.grad(fn)
    force = solid.guccione_force(x, geo, fields, parameters)
    force_error = (force+residual(x)).abs().max()
    torch.testing.assert_close(force, -residual(x), atol=1e-7, rtol=1e-10)
    torch.testing.assert_close(force.sum(0), X.new_zeros(3), atol=1e-7, rtol=0)
    direction = torch.sin(x)
    _, Kv = torch.func.jvp(residual, (x,), (direction,))
    h = 1e-6
    fd = (residual(x+h*direction)-residual(x-h*direction))/(2*h)
    torch.testing.assert_close(Kv, fd, atol=1e-3, rtol=1e-6)
    return {
        "device": str(device), "torch": torch.__version__, "dtype": str(x.dtype),
        "elements": cells.shape[0], "nodes": X.shape[0],
        "quadrature_points_per_element": geo.values.shape[0],
        "potential_at_fixed_tension": fn(x).item(),
        "min_det_F_at_quadrature": determinant3(solid.deformation_gradient(x, geo)).min().item(),
        "max_force_difference": force_error.item(),
        "net_force_norm": torch.linalg.vector_norm(force.sum(0)).item(),
        "tangent_relative_error": (torch.linalg.vector_norm(Kv-fd)/torch.linalg.vector_norm(Kv)).item(),
        "status": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--degree", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.degree), indent=2))
