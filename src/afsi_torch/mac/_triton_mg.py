"""Fused CUDA stencils on contiguous PyTorch pressure tensors.

Homogeneous Neumann A=-D G, weighted Jacobi, 2x cell restriction, and
cell-centered trilinear prolongation (align_corners=False). No atomics or
changes to solver tolerance. Imported only by the optional CUDA backend.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _apply(P, i, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
           HX2: tl.constexpr, HY2: tl.constexpr, HZ2: tl.constexpr):
    active = i < NX*NY*NZ
    x = i//(NY*NZ)
    y = (i//NZ)%NY
    z = i%NZ
    p = tl.load(P+i, mask=active, other=0.)
    xp = tl.load(P+i+NY*NZ, mask=active & (x+1 < NX), other=0.)
    xm = tl.load(P+i-NY*NZ, mask=active & (x > 0), other=0.)
    yp = tl.load(P+i+NZ, mask=active & (y+1 < NY), other=0.)
    ym = tl.load(P+i-NZ, mask=active & (y > 0), other=0.)
    zp = tl.load(P+i+1, mask=active & (z+1 < NZ), other=0.)
    zm = tl.load(P+i-1, mask=active & (z > 0), other=0.)
    # Same per-axis accumulation order as grid.negative_laplacian.
    a = -tl.where(x+1 < NX, (xp-p)/HX2, 0.)
    a += tl.where(x > 0, (p-xm)/HX2, 0.)
    a -= tl.where(y+1 < NY, (yp-p)/HY2, 0.)
    a += tl.where(y > 0, (p-ym)/HY2, 0.)
    a -= tl.where(z+1 < NZ, (zp-p)/HZ2, 0.)
    a += tl.where(z > 0, (p-zm)/HZ2, 0.)
    return p, a


@triton.jit
def _stencil(P, RHS, DIAG, OUT, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
             HX2: tl.constexpr, HY2: tl.constexpr, HZ2: tl.constexpr,
             SMOOTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    active = i < NX*NY*NZ
    p, a = _apply(P, i, NX, NY, NZ, HX2, HY2, HZ2)
    r = tl.load(RHS+i, mask=active, other=0.)-a
    if SMOOTH:
        d = tl.load(DIAG+i, mask=active, other=1.)
        value = p+(2./3.)*r/d
    else:
        value = r
    tl.store(OUT+i, value, mask=active)


@triton.jit
def _restrict(FINE, OUT, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
              BLOCK: tl.constexpr):
    # NX,NY,NZ are the coarse dimensions; every coarse cell has eight children.
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    active = i < NX*NY*NZ
    x, y, z = i//(NY*NZ), (i//NZ)%NY, i%NZ
    j = ((2*x)*(2*NY)+2*y)*(2*NZ)+2*z
    value = tl.load(FINE+j, mask=active, other=0.)
    value += tl.load(FINE+j+1, mask=active, other=0.)
    value += tl.load(FINE+j+2*NZ, mask=active, other=0.)
    value += tl.load(FINE+j+2*NZ+1, mask=active, other=0.)
    value += tl.load(FINE+j+4*NY*NZ, mask=active, other=0.)
    value += tl.load(FINE+j+4*NY*NZ+1, mask=active, other=0.)
    value += tl.load(FINE+j+4*NY*NZ+2*NZ, mask=active, other=0.)
    value += tl.load(FINE+j+4*NY*NZ+2*NZ+1, mask=active, other=0.)
    tl.store(OUT+i, value*.125, mask=active)


@triton.jit
def _prolong_add(COARSE, FINE, NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                 BLOCK: tl.constexpr):
    # Fine coordinate maps to i/2-1/4 on the cell-centered coarse lattice.
    # Clamp both endpoints to reproduce PyTorch's edge replication exactly.
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    active = i < NX*NY*NZ
    x, y, z = i//(NY*NZ), (i//NZ)%NY, i%NZ
    ax, ay, az = x//2-1+x%2, y//2-1+y%2, z//2-1+z%2
    x0, x1 = tl.maximum(ax, 0), tl.minimum(ax+1, NX//2-1)
    y0, y1 = tl.maximum(ay, 0), tl.minimum(ay+1, NY//2-1)
    z0, z1 = tl.maximum(az, 0), tl.minimum(az+1, NZ//2-1)
    wx = tl.where(x%2 == 0, .75, .25)
    wy = tl.where(y%2 == 0, .75, .25)
    wz = tl.where(z%2 == 0, .75, .25)
    c000 = tl.load(COARSE+(x0*(NY//2)+y0)*(NZ//2)+z0, mask=active, other=0.)
    c001 = tl.load(COARSE+(x0*(NY//2)+y0)*(NZ//2)+z1, mask=active, other=0.)
    c010 = tl.load(COARSE+(x0*(NY//2)+y1)*(NZ//2)+z0, mask=active, other=0.)
    c011 = tl.load(COARSE+(x0*(NY//2)+y1)*(NZ//2)+z1, mask=active, other=0.)
    c100 = tl.load(COARSE+(x1*(NY//2)+y0)*(NZ//2)+z0, mask=active, other=0.)
    c101 = tl.load(COARSE+(x1*(NY//2)+y0)*(NZ//2)+z1, mask=active, other=0.)
    c110 = tl.load(COARSE+(x1*(NY//2)+y1)*(NZ//2)+z0, mask=active, other=0.)
    c111 = tl.load(COARSE+(x1*(NY//2)+y1)*(NZ//2)+z1, mask=active, other=0.)
    c00 = c000*(1.-wz)+c001*wz
    c01 = c010*(1.-wz)+c011*wz
    c10 = c100*(1.-wz)+c101*wz
    c11 = c110*(1.-wz)+c111*wz
    correction = ((c00*(1.-wy)+c01*wy)*(1.-wx)+(c10*(1.-wy)+c11*wy)*wx)
    value = tl.load(FINE+i, mask=active, other=0.)+correction
    tl.store(FINE+i, value, mask=active)


class TritonKernels:
    name = 'triton'
    block = 256

    def smooth(self, p, rhs, diagonal, out, spacing):
        with torch.cuda.device(p.device):
            _stencil[(triton.cdiv(p.numel(), self.block),)](
                p, rhs, diagonal, out, *p.shape, *(h*h for h in spacing),
                SMOOTH=True, BLOCK=self.block, enable_fp_fusion=False)

    def residual(self, p, rhs, out, spacing):
        with torch.cuda.device(p.device):
            _stencil[(triton.cdiv(p.numel(), self.block),)](
                p, rhs, p, out, *p.shape, *(h*h for h in spacing),
                SMOOTH=False, BLOCK=self.block, enable_fp_fusion=False)

    def restrict(self, fine, out):
        with torch.cuda.device(fine.device):
            _restrict[(triton.cdiv(out.numel(), self.block),)](
                fine, out, *out.shape, BLOCK=self.block, enable_fp_fusion=False)

    def prolong_add(self, coarse, fine):
        with torch.cuda.device(fine.device):
            _prolong_add[(triton.cdiv(fine.numel(), self.block),)](
                coarse, fine, *fine.shape, BLOCK=self.block, enable_fp_fusion=False)
