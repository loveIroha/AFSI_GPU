"""All three solid weak-force terms, without a displacement/time-step solver."""
import argparse
import json
import torch
from afsi_torch import boundary as bd, solid
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from afsi_torch.tetrahedron import reference_nodes


def run(device="cpu"):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    X = reference_nodes(device=device)
    cells = torch.arange(10, device=device).reshape(1, 10)
    geometry = solid.prepare_p2(X, cells)
    fields = prepare_reference_fields(geometry, [1., 0., 0.], [0., 1., 0.], 1000.)
    parameters = GuccioneParameters()
    faces = bd.extract_boundary(X, cells)
    # Synthetic tags: z=0 is BASE; x+y+z=1 is the loaded patch.
    base = bd.prepare_surface(X, faces[(X[faces[:, :3], 2].abs() < 1e-12).all(-1)])
    loaded = bd.prepare_surface(X, faces[(X[faces[:, :3]].sum(-1) > .9).all(-1)])
    pressure = bd.prepare_coefficient(loaded, 1200.)
    beta = bd.prepare_coefficient(base, 5000., nonnegative=True)
    x = X+.03*X.square()
    x[:, 2] += .02*X[:, 0].square()
    solid.validate_deformation(x, geometry)
    bd.validate_surface(x, loaded)
    bd.validate_surface(x, base)

    def force(y):
        return (solid.guccione_force(y, geometry, fields, parameters)
                +bd.pressure_force(y, loaded, pressure)+bd.spring_force(y, base, beta))

    bulk = solid.guccione_force(x, geometry, fields, parameters)
    pressure_g = bd.pressure_force(x, loaded, pressure)
    spring_g = bd.spring_force(x, base, beta)
    torch.testing.assert_close(bulk.sum(0), X.new_zeros(3), atol=1e-8, rtol=0)
    spring_ad = -torch.func.grad(lambda y: bd.spring_energy(y, base, beta))(x)
    torch.testing.assert_close(spring_g, spring_ad, atol=1e-9, rtol=1e-10)
    direction = torch.sin(torch.arange(x.numel(), dtype=x.dtype, device=device)).reshape_as(x)
    residual = lambda y: -force(y)  # Full residual, including follower pressure.
    _, Kv = torch.func.jvp(residual, (x,), (direction,))
    h = 1e-6
    fd = (residual(x+h*direction)-residual(x-h*direction))/(2*h)
    torch.testing.assert_close(Kv, fd, atol=1e-3, rtol=1e-6)
    return {
        "device": str(device), "torch": torch.__version__,
        "elements": 1, "nodes": 10, "boundary_faces": faces.shape[0],
        "loaded_faces": loaded.faces.shape[0], "base_faces": base.faces.shape[0],
        "surface_quadrature_points": loaded.values.shape[0],
        "pressure_resultant": pressure_g.sum(0).detach().cpu().tolist(),
        "spring_resultant": spring_g.sum(0).detach().cpu().tolist(),
        "total_force_norm": torch.linalg.vector_norm(force(x)).item(),
        "spring_gradient_error": (spring_g-spring_ad).abs().max().item(),
        "full_tangent_relative_error": (torch.linalg.vector_norm(Kv-fd)/torch.linalg.vector_norm(Kv)).item(),
        "status": "passed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    print(json.dumps(run(parser.parse_args().device), indent=2))
