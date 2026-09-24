"""Adapter checks; these do NOT claim an actual DOLFINx run."""
import numpy as np
import pytest
import torch
from afsi_torch import solid, boundary as bd, triangle
from afsi_torch.quadrature import tetrahedron_rule, prepare_rule
from afsi_torch.tetrahedron import reference_nodes
from validation.reference_io import match_points
from validation.compare_dolfinx import select_tagged_faces


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA device unavailable'))])
def device(request):
    return request.param


def test_external_volume_rule_and_dof_permutation(device):
    X = reference_nodes(device=device)
    permutation = torch.tensor([7, 0, 9, 3, 1, 8, 2, 6, 4, 5], device=device)
    inverse = permutation.argsort()
    cells = inverse[torch.arange(10, device=device)].reshape(1, 10)
    q, w = tetrahedron_rule(6)
    # Reversed point order must keep weights paired, irrespective of global DOFs.
    geo = solid.prepare_p2(X[permutation], cells, quadrature=(q.flip(0), w.flip(0)))
    x = X+.04*X.square()
    base = solid.prepare_p2(X, torch.arange(10, device=device).reshape(1, 10), degree=6)
    force = solid.stress_force(x[permutation], geo, 2., 10.)[inverse]
    torch.testing.assert_close(force, solid.stress_force(x, base, 2., 10.), atol=1e-12, rtol=1e-11)
    F = solid.deformation_gradient(x[permutation], geo)
    expected = torch.diag_embed(1+.08*q.flip(0).to(device)).unsqueeze(0)
    torch.testing.assert_close(F, expected, atol=1e-13, rtol=1e-12)


def test_external_surface_rule_and_polynomial_resultant(device):
    X = reference_nodes(device=device)
    faces = bd.extract_boundary(X, torch.arange(10, device=device).reshape(1, 10))
    bottom = faces[(X[faces[:, :3], 2] == 0).all(1)]
    q, w = triangle.quadrature(6)
    surface = bd.prepare_surface(X, bottom, quadrature=(q.flip(0), w.flip(0)))
    p = bd.prepare_coefficient(surface, 1+X[:, 0]**2)
    force = bd.pressure_force(X, surface, p)
    torch.testing.assert_close(force.sum(0), X.new_tensor([0., 0., 7/12]), atol=1e-13, rtol=1e-12)
    # Caller mutations after preparation cannot corrupt the integration weights.
    w.zero_()
    torch.testing.assert_close(bd.pressure_force(X, surface, p), force)


@pytest.mark.parametrize('bad', ['outside', 'nan', 'normalization', 'shape', 'empty'])
def test_external_rule_rejects_invalid_data(bad):
    q, w = tetrahedron_rule(4)
    if bad == 'outside':
        q[0, 0] = 2
    elif bad == 'nan':
        w[0] = float('nan')
    elif bad == 'normalization':
        w *= 2
    elif bad == 'shape':
        q = q[:, :2]
    else:
        q, w = q[:0], w[:0]
    with pytest.raises(ValueError):
        prepare_rule((q, w), 3, torch.zeros(1, dtype=torch.float64))


def test_coordinate_mapping_rejects_ambiguity_and_missing_nodes():
    X = reference_nodes().numpy()
    order = np.array([7, 0, 9, 3, 1, 8, 2, 6, 4, 5])
    np.testing.assert_array_equal(match_points(X, X[order]), np.argsort(order))
    with pytest.raises(ValueError, match='ambiguous'):
        match_points(X, np.concatenate((X, X[:1])))
    with pytest.raises(ValueError, match='missing'):
        match_points(X, X[1:])
    with pytest.raises(ValueError, match='one-to-one'):
        match_points(X[[0, 0]], X)


def test_facet_tags_survive_ordering_and_reject_interior():
    X = reference_nodes()
    faces = bd.extract_boundary(X, torch.arange(10).reshape(1, 10)).numpy()
    vertices = faces[[2, 0], :3][:, [2, 0, 1]]
    selected = select_tagged_faces(faces, vertices, np.array([2, 1]))
    np.testing.assert_array_equal(selected[1], faces[[0]])
    np.testing.assert_array_equal(selected[2], faces[[2]])
    with pytest.raises(ValueError):
        select_tagged_faces(faces, np.array([[0, 1, 9], vertices[1]]), np.array([2, 1]))
    with pytest.raises(ValueError):
        select_tagged_faces(faces, vertices[[0, 0]], np.array([2, 1]))
