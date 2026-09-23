"""Independent Basix + NumPy complex-step constitutive/FEM reference.

No PyTorch derivatives or analytic PK1 formula are used in the oracle. Same
quadrature points isolate formula/assembly errors from quadrature-rule errors.
This does NOT execute the complete original afsi or DOLFINx solver.
"""
import json
import numpy as np
import torch
import basix

from afsi_torch import materials, solid
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.quadrature import tetrahedron_rule
from afsi_torch.tetrahedron import promote_p1, reference_nodes


def density(F, f, s, tension, p):
    """Scalar contraction form, independent of the Torch matrix weighting form."""
    n = np.cross(s, f)
    E = .5*(F.T @ F-np.eye(3))
    eff, ess, enn = f @ E @ f, s @ E @ s, n @ E @ n
    efs, efn, esn = f @ E @ s, f @ E @ n, s @ E @ n
    Q = p.bf*eff**2+p.bt*(ess**2+enn**2+2*esn**2)+2*p.bfs*(efs**2+efn**2)
    W = .5*p.C*np.expm1(Q)+p.kappa*(np.linalg.det(F)-1)**2
    return W+.5*tension*((F@f) @ (F@f)-f@f)


def complex_step_pk1(F, f, s, tension, p):
    P = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            perturbed = F.astype(complex)
            perturbed[i, j] += 1e-30j
            P[i, j] = density(perturbed, f, s, tension, p).imag/1e-30
    return P


def compare():
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)
    X, cells = promote_p1(vertices, torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    geo = solid.prepare_p2(X, cells)
    angle = .6*X[:, 0]
    zero = torch.zeros_like(angle)
    f_nodes = torch.stack((angle.cos(), angle.sin(), zero), -1)
    s_nodes = torch.stack((-angle.sin(), angle.cos(), zero), -1)
    t_nodes = 1000.+500.*X[:, 1].square()
    fields = prepare_reference_fields(geo, f_nodes, s_nodes, t_nodes)
    p = materials.GuccioneParameters()
    x = X+.04*X.square()
    points, weights = tetrahedron_rule(4)
    element = basix.create_element(basix.ElementFamily.P, basix.CellType.tetrahedron,
                                   2, basix.LagrangeVariant.equispaced)
    distance = np.linalg.norm(reference_nodes().numpy()[:, None, :]-element.points[None, :, :], axis=-1)
    permutation = distance.argmin(axis=1)
    np.testing.assert_allclose(distance[np.arange(10), permutation], 0, atol=1e-14)
    assert len(np.unique(permutation)) == 10
    table = element.tabulate(1, points.numpy())
    N = table[0, :, :, 0][:, permutation]
    dN = np.stack([table[basix.index(*d), :, :, 0][:, permutation]
                   for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]], -1)
    ref_force, ref_potential, stresses = np.zeros_like(X.numpy()), 0., []
    for cell in cells.numpy():
        vertices = X.numpy()[cell[:4]]
        D = (vertices[1:]-vertices[:1]).T
        gradients = dN @ np.linalg.inv(D)
        w = abs(np.linalg.det(D))*weights.numpy()
        Fs = np.einsum("ai,qaJ->qiJ", x.numpy()[cell], gradients)
        fs, ss, ts = N @ f_nodes.numpy()[cell], N @ s_nodes.numpy()[cell], N @ t_nodes.numpy()[cell]
        P = np.stack([complex_step_pk1(F, f, s, t, p) for F, f, s, t in zip(Fs, fs, ss, ts)])
        W = np.array([density(F, f, s, t, p) for F, f, s, t in zip(Fs, fs, ss, ts)])
        stresses.append(P)
        np.add.at(ref_force, cell, -np.einsum("q,qiJ,qaJ->ai", w, P, gradients))
        ref_potential += w @ W
    F = solid.deformation_gradient(x, geo)
    P = materials.guccione_pk1(F, fields.fiber, fields.sheet, fields.normal, p)
    P += materials.active_pk1(F, fields.fiber, fields.tension)
    ref_P = np.stack(stresses)
    actual_force = solid.guccione_force(x, geo, fields, p).numpy()
    actual_potential = float(solid.guccione_energy(x, geo, fields, p))
    np.testing.assert_allclose(P.numpy(), ref_P, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(actual_force, ref_force, atol=1e-7, rtol=1e-10)
    np.testing.assert_allclose(actual_potential, ref_potential, atol=1e-9, rtol=1e-11)
    return {
        "basix": basix.__version__, "reference": "NumPy complex-step + Basix",
        "max_pk1_error": float(np.abs(P.numpy()-ref_P).max()),
        "potential_error": abs(actual_potential-ref_potential),
        "max_force_error": float(np.abs(actual_force-ref_force).max()),
        "status": "passed",
    }


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
