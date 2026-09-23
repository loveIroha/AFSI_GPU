"""Independent volume-Basix/cofactor reference for P2 boundary forces.

Reference uses full tetrahedron basis on each facet and det(F)*inv(F).T*N0,
not the triangle basis or deformed tangent cross-products of the Torch kernel.
Topology and quadrature points are shared; no DOLFINx solver is executed.
"""
import json
import numpy as np
import torch
import basix
from afsi_torch import boundary as bd, triangle
from afsi_torch.tetrahedron import promote_p1, reference_nodes


def compare():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)
    X, cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    faces = bd.extract_boundary(X, cells)
    surface = bd.prepare_surface(X, faces, degree=6)
    x = X+.04*X.square()
    x[:, 2] += .03*X[:, 0].square()
    p_nodes = 1000.+200.*X[:, 0].square()
    beta_nodes = 5000.+1000.*X[:, 1].square()
    pressure = bd.prepare_coefficient(surface, p_nodes)
    beta = bd.prepare_coefficient(surface, beta_nodes, nonnegative=True)
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.tetrahedron,
                                   2, basix.LagrangeVariant.equispaced)
    xi_nodes = reference_nodes().numpy()
    distance = np.linalg.norm(xi_nodes[:, None, :]-element.points[None, :, :], axis=-1)
    permutation = distance.argmin(1)
    np.testing.assert_allclose(distance[np.arange(10), permutation], 0, atol=1e-14)
    assert len(np.unique(permutation)) == 10
    q, qw = triangle.quadrature(6)
    q, qw = q.numpy(), qw.numpy()
    Xn, xn, cn = X.numpy(), x.numpy(), cells.numpy()
    ref_p, ref_s = np.zeros_like(Xn), np.zeros_like(Xn)
    ref_spring_energy = 0.
    for face in faces.numpy():
        owners = [cell for cell in cn if np.isin(face[:3], cell[:4]).all()]
        assert len(owners) == 1
        cell = owners[0]
        local_vertices = [int(np.where(cell == v)[0][0]) for v in face[:3]]
        a, b, c = xi_nodes[local_vertices]
        qxi = a+q[:, :1]*(b-a)+q[:, 1:]*(c-a)
        table = element.tabulate(1, qxi)
        N = table[0, :, :, 0][:, permutation]
        dN = np.stack([table[basix.index(*d), :, :, 0][:, permutation]
                       for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]], -1)
        D = (Xn[cell[1:4]]-Xn[cell[0]]).T
        gradients = dN @ np.linalg.inv(D)
        F = np.einsum("ai,qaJ->qiJ", xn[cell], gradients)
        a0 = np.cross(Xn[face[1]]-Xn[face[0]], Xn[face[2]]-Xn[face[0]])
        cof = np.linalg.det(F)[:, None, None]*np.linalg.inv(F).transpose(0, 2, 1)
        p, stiffness = N @ p_nodes.numpy()[cell], N @ beta_nodes.numpy()[cell]
        traction = -p[:, None]*np.einsum("qiJ,J->qi", cof, a0)
        np.add.at(ref_p, cell, np.einsum("q,qa,qi->ai", qw, N, traction))
        u = N @ (xn[cell]-Xn[cell])
        dA0 = qw*np.linalg.norm(a0)
        np.add.at(ref_s, cell, -np.einsum("q,qa,q,qi->ai", dA0, N, stiffness, u))
        ref_spring_energy += .5*np.sum(dA0*stiffness*np.sum(u*u, axis=1))
    p_actual = bd.pressure_force(x, surface, pressure).numpy()
    s_actual = bd.spring_force(x, surface, beta).numpy()
    energy_actual = float(bd.spring_energy(x, surface, beta))
    np.testing.assert_allclose(p_actual, ref_p, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(s_actual, ref_s, atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(energy_actual, ref_spring_energy, atol=1e-10, rtol=1e-11)
    return {
        "basix": basix.__version__, "reference": "tetrahedron basis + cofactor(F)*N0",
        "max_pressure_force_error": float(np.abs(p_actual-ref_p).max()),
        "max_spring_force_error": float(np.abs(s_actual-ref_s).max()),
        "spring_energy_error": abs(energy_actual-ref_spring_energy),
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
