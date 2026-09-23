"""Optional independent CPU reference: python validation/compare_basix.py.

Requires fenics-basix, not the unrelated package named basix on PyPI.
This compares local basis/gradients and volume forces, not DOLFINx global DOFs.
"""
import json
import numpy as np
import torch
import basix

from afsi_torch import solid
from afsi_torch.quadrature import tetrahedron_rule
from afsi_torch.tetrahedron import promote_p1, reference_nodes, tabulate


def compare():
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.tetrahedron,
                                   2, basix.LagrangeVariant.equispaced)
    nodes = reference_nodes().numpy()
    distance = np.linalg.norm(nodes[:, None, :] - element.points[None, :, :], axis=-1)
    permutation = distance.argmin(axis=1)
    np.testing.assert_allclose(distance[np.arange(10), permutation], 0, atol=1e-14)
    assert len(np.unique(permutation)) == 10
    points, weights = tetrahedron_rule(4)
    table = element.tabulate(1, points.numpy())
    ref_N = table[0, :, :, 0][:, permutation]
    ref_dN = np.stack([table[basix.index(*derivative), :, :, 0][:, permutation]
                       for derivative in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]], axis=-1)
    N, dN = tabulate(points)
    np.testing.assert_allclose(N.numpy(), ref_N, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(dN.numpy(), ref_dN, atol=1e-13, rtol=1e-12)

    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)
    X, cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    geometry = solid.prepare_p2(X, cells, degree=4)
    x = X + .08*X.square()
    ref_force, ref_energy = np.zeros_like(X.numpy()), 0.
    # Intentionally simple NumPy cell loop as an independent correctness oracle.
    for cell in cells.numpy():
        vertices = X.numpy()[cell[:4]]
        D = (vertices[1:] - vertices[:1]).T
        gradients = ref_dN @ np.linalg.inv(D)
        w = weights.numpy()*abs(np.linalg.det(D))
        F = np.einsum("ai,qaJ->qiJ", x.numpy()[cell], gradients)
        logJ = np.log(np.linalg.det(F))
        W = (F**2).sum(axis=(-2, -1))-3-2*logJ+2.5*logJ**2
        invT = np.linalg.inv(F).transpose(0, 2, 1)
        P = 2*(F-invT)+5*logJ[:, None, None]*invT
        local = -np.einsum("q,qiJ,qaJ->ai", w, P, gradients)
        np.add.at(ref_force, cell, local)
        ref_energy += np.dot(w, W)
    actual_force = solid.stress_force(x, geometry, 2., 5.).numpy()
    actual_energy = float(solid.energy(x, geometry, 2., 5.))
    np.testing.assert_allclose(actual_force, ref_force, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(actual_energy, ref_energy, atol=1e-13, rtol=1e-11)
    return {
        "basix": basix.__version__, "local_to_basix_columns": permutation.tolist(),
        "max_basis_error": float(np.max(np.abs(N.numpy()-ref_N))),
        "max_basis_gradient_error": float(np.max(np.abs(dN.numpy()-ref_dN))),
        "energy_error": abs(actual_energy-ref_energy),
        "max_force_error": float(np.max(np.abs(actual_force-ref_force))),
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
