"""Monotone parabolic reconstruction for conservative MAC momentum fluxes.

Uniform-grid fourth-order interface interpolation and the original PPM cell
monotonicity constraints. This is a method-of-lines reconstruction: AB2 (or
the startup midpoint rule) supplies the time quadrature. It does not reproduce
IBAMR's characteristic tracing/PPM implementation bit for bit.
"""
import torch
from .grid import slab


def parabolic_states(q, axis):
    """Left/right states at n+1 interfaces; odd reflection at closed walls."""
    # Two ghost cells on each side; the grid has at least four cells.
    pad = torch.cat((-q[slab(axis, 0, 2)].flip(axis), q,
                     -q[slab(axis, -2, None)].flip(axis)), axis)
    a, b = pad[slab(axis, 1, -2)], pad[slab(axis, 2, -1)]
    face = (7/12)*(a+b)-(1/12)*(pad[slab(axis, None, -3)]+pad[slab(axis, 3, None)])
    face = torch.maximum(torch.minimum(a,b), torch.minimum(torch.maximum(a,b),face))
    left, right = face[slab(axis, None, -1)], face[slab(axis, 1, None)]
    extremum = (right-q)*(q-left) <= 0
    left, right = torch.where(extremum,q,left), torch.where(extremum,q,right)
    delta = right-left
    curvature = 6*(q-.5*(left+right))
    # Conditions use the same original parabola, not sequentially updated faces.
    left_new = torch.where(delta*curvature > delta.square(),3*q-2*right,left)
    right_new = torch.where(delta*curvature < -delta.square(),3*q-2*left,right)
    # Odd ghosts supply the two physical wall interfaces. Flux there is zero.
    from_left = torch.cat((-left_new[slab(axis,0,1)], right_new),axis)
    from_right = torch.cat((left_new,-right_new[slab(axis,-1,None)]),axis)
    return from_left, from_right


def convection_ppm(velocity, spacing):
    """Upwind parabolic momentum fluxes on staggered control volumes."""
    result = []
    for c,u in enumerate(velocity):
        out = torch.zeros_like(u)
        for axis,h in enumerate(spacing):
            ql,qr = parabolic_states(u,axis)
            if axis == c:
                speed = .5*(u[slab(c,1,None)]+u[slab(c,None,-1)])
                ql,qr = ql[slab(c,1,-1)], qr[slab(c,1,-1)]
                flux = speed*torch.where(speed>=0,ql,qr)
                out[slab(c,1,-1)] += torch.diff(flux,dim=c)/h
            else:
                other = velocity[axis]
                padded = torch.cat((-other[slab(c,0,1)],other,-other[slab(c,-1,None)]),c)
                speed = .5*(padded[slab(c,1,None)]+padded[slab(c,None,-1)])
                flux = speed*torch.where(speed>=0,ql,qr)
                out += torch.diff(flux,dim=axis)/h
        out[slab(c,0,1)] = 0.
        out[slab(c,-1,None)] = 0.
        result.append(out)
    return tuple(result)
