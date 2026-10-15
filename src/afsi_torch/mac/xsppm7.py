"""IBAMR xsPPM7 reconstruction and staggered advective derivative in PyTorch.

Algorithm: Rider, Greenough & Kamm (2007); IBAMR godunov_xsPPM7_extrapolate3d,
WENO5_interp, monotonize and navier_stokes_reset_adv_velocity3d. The spatial
extrapolator has NO characteristic time tracing; CN--AB2 supplies time accuracy.
Closed-box ghost values are odd reflections about the actual wall location.
See docs/GRIFFITH_LUO_2017.md and third_party/IBAMR_COPYRIGHT.

Portions derived from IBAMR, Copyright (c) 2014-2026 by the IBAMR developers.
Distributed under the 3-clause BSD license reproduced in that notice.
"""
import torch
from .grid import slab, zero_normal


def _median(a,b,c):
    return torch.maximum(torch.minimum(a,b),torch.minimum(torch.maximum(a,b),c))


def _monotonize(qm,q,qp,left,right):
    lo,hi=_median(q,left,qm),_median(q,right,qp)
    return _median(q,lo,3*q-2*hi),_median(q,hi,3*q-2*lo)


def _weno(q):
    a,b,c,d,e=q
    f=((11*c-7*b+2*a)/6,(2*d+5*c-b)/6,(-e+5*d+2*c)/6)
    smooth=((13/12)*(c-2*b+a).square()+.25*(3*c-4*b+a).square(),
            (13/12)*(d-2*c+b).square()+.25*(d-b).square(),
            (13/12)*(e-2*d+c).square()+.25*(e-4*d+3*c).square())
    # FP64 matches IBAMR's 1e-40 regularizer; retain a positive FP32 value.
    eps=max(1e-40,torch.finfo(c.dtype).tiny)
    weights=tuple(w/(s+eps) for w,s in zip((.1,.6,.3),smooth))
    total=sum(weights)
    omega=tuple(w/total for w in weights)
    mapped=tuple(o*(w+w*w-3*w*o+o.square())/(w*w+o*(1-2*w))
                 for o,w in zip(omega,(.1,.6,.3)))
    total=sum(mapped)
    return sum(w/total*v for w,v in zip(mapped,f))


def _odd_pad(q,axis,component,width):
    # Normal velocity has boundary nodes; tangential velocity has half-cell
    # locations. Their reflection indices must differ by one.
    left=q[slab(axis,1,width+1)] if axis==component else q[slab(axis,0,width)]
    right=q[slab(axis,-width-1,-1)] if axis==component else q[slab(axis,-width,None)]
    return torch.cat((-left.flip(axis),q,-right.flip(axis)),axis)


def xsppm7_states(q,axis,component):
    """One left/right trace at every control-volume face, including walls."""
    p=_odd_pad(q,axis,component,4)
    n=q.shape[axis]
    # Reconstruct physical cells and one ghost on each side. Seven-point
    # interpolation therefore reads four ghost layers.
    s=lambda k:p[slab(axis,3+k,3+k+n+2)]
    a,b,c,d,e,f,g=(s(k) for k in range(-3,4))
    left=(-3*g+25*f-101*e+319*d+214*c-38*b+4*a)/420
    right=(-3*a+25*b-101*c+319*d+214*e-38*f+4*g)/420
    lo,hi=_monotonize(c,d,e,left,right)
    troubled=((lo-left).square()>=1e-12)|((hi-right).square()>=1e-12)
    wl,wr=_weno((f,e,d,c,b)),_weno((b,c,d,e,f))
    extremum=((lo-d).square()<=1e-12)|((hi-d).square()<=1e-12)
    def slope(qm,q0,qp):
        dl,dr=q0-qm,qp-q0
        center=.5*(qp-qm)
        limited=torch.minimum(center.abs(),torch.minimum(2*dl.abs(),2*dr.abs()))*center.sign()
        return torch.where(dl*dr>1e-12,limited,torch.zeros_like(center))
    dm,dc,dp=slope(b,c,d),slope(c,d,e),slope(d,e,f)
    fl=_median(.5*(c+d)-(dc-dm)/6,wl,left)
    fr=_median(.5*(d+e)-(dp-dc)/6,wr,right)
    wl=torch.where(extremum,_median(d,wl,left),wl)
    wr=torch.where(extremum,_median(d,wr,right),wr)
    ml,mr=_monotonize(c,d,e,torch.where(extremum,wl,fl),torch.where(extremum,wr,fr))
    left=torch.where(troubled,_median(wl,ml,left),left)
    right=torch.where(troubled,_median(wr,mr,right),right)
    return right[slab(axis,None,-1)],left[slab(axis,1,None)]


def convection_xsppm7(velocity,spacing):
    """PPM upwind faces -> reset advecting velocity -> advective u.grad(u)."""
    velocity=zero_normal(velocity)
    faces=[]
    for c,u in enumerate(velocity):
        row=[]
        for axis in range(3):
            other=_odd_pad(velocity[axis],c,axis,1)
            speed=.5*(other[slab(c,1,None)]+other[slab(c,None,-1)])
            left,right=xsppm7_states(u,axis,c)
            sign=torch.where(speed.abs()>1e-8,speed.sign(),torch.zeros_like(speed))
            row.append(.5*(left+right)+.5*sign*(left-right))
        faces.append(row)
    result=[]
    for c in range(3):
        value=torch.zeros_like(velocity[c])
        for axis,h in enumerate(spacing):
            # IBAMR resets U_adv[c][axis] from U_half[axis][c], avoiding an
            # inconsistent centered advector after upwind reconstruction.
            speed=faces[axis][c]
            value += .5*(speed[slab(axis,1,None)]+speed[slab(axis,None,-1)])*torch.diff(faces[c][axis],dim=axis)/h
        result.append(value)
    return zero_normal(tuple(result))
