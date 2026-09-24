"""Small NumPy-only full-grid IB oracle, no production stencil reuse."""
from math import sqrt
import numpy as np


def phi(r):
    r = abs(float(r))
    if r >= 2:
        return 0.
    if r >= 1:
        return (5-2*r-sqrt(-7+12*r-4*r*r))/8
    return (3-2*r+sqrt(1+4*r-4*r*r))/8


def weights(positions, nodes, spacing):
    H = np.empty((len(positions), len(nodes)))
    for a, x in enumerate(positions):
        for i, y in enumerate(nodes):
            d = (x-y)/spacing
            H[a, i] = phi(d[0])*phi(d[1])*phi(d[2])
    np.testing.assert_allclose(H.sum(1), 1., atol=2e-13, rtol=0)
    return H
