"""Allocation-free CN velocity Jacobi kernels, including odd wall ghosts."""
import triton
import triton.language as tl


@triton.jit
def _initialize(B, P, D, R, U, PARAM,
                NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
                C: tl.constexpr, H: tl.constexpr, PRESSURE: tl.constexpr,
                BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < NX*NY*NZ
    x, y, z = i//(NY*NZ), (i//NZ)%NY, i%NZ
    coord = x if C == 0 else y if C == 1 else z
    length = NX if C == 0 else NY if C == 1 else NZ
    interior = valid & (coord > 0) & (coord < length-1)
    rhs = tl.load(B+i, valid, other=0.)
    if PRESSURE:
        # Pressure cells on either side of an interior component-normal face.
        if C == 0:
            j, stride = ((x-1)*NY+y)*NZ+z, NY*NZ
        elif C == 1:
            j, stride = (x*(NY-1)+y-1)*NZ+z, NZ
        else:
            j, stride = (x*NY+y)*(NZ-1)+z-1, 1
        left = tl.load(P+j, interior, other=0.)
        right = tl.load(P+j+stride, interior, other=0.)
        rhs -= tl.load(PARAM+1)*((right-left)/H)
    rhs = tl.where(interior, rhs, 0.)
    diag = tl.load(D+i, valid, other=1.)
    tl.store(R+i, rhs, valid)
    tl.store(U+i, rhs/diag, valid)


@triton.jit
def _sweep(U, R, D, OUT, PARAM,
           NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
           C: tl.constexpr, HX2: tl.constexpr, HY2: tl.constexpr,
           HZ2: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < NX*NY*NZ
    x, y, z = i//(NY*NZ), (i//NZ)%NY, i%NZ
    u = tl.load(U+i, valid, other=0.)
    xm = tl.load(U+i-NY*NZ, valid & (x > 0), other=0.)
    xp = tl.load(U+i+NY*NZ, valid & (x+1 < NX), other=0.)
    ym = tl.load(U+i-NZ, valid & (y > 0), other=0.)
    yp = tl.load(U+i+NZ, valid & (y+1 < NY), other=0.)
    zm = tl.load(U+i-1, valid & (z > 0), other=0.)
    zp = tl.load(U+i+1, valid & (z+1 < NZ), other=0.)
    lap = (tl.where(x > 0, xm, -u)-2*u+tl.where(x+1 < NX, xp, -u))/HX2
    lap += (tl.where(y > 0, ym, -u)-2*u+tl.where(y+1 < NY, yp, -u))/HY2
    lap += (tl.where(z > 0, zm, -u)-2*u+tl.where(z+1 < NZ, zp, -u))/HZ2
    r = tl.load(R+i, valid, other=0.)
    diag = tl.load(D+i, valid, other=1.)
    value = u+(r-u+tl.load(PARAM)*lap)/diag
    coord = x if C == 0 else y if C == 1 else z
    length = NX if C == 0 else NY if C == 1 else NZ
    value = tl.where((coord > 0) & (coord < length-1), value, 0.)
    tl.store(OUT+i, value, valid)


class HelmholtzKernels:
    block = 256

    def initialize(self, b, p, diagonal, rhs, u, parameters, grid):
        for c in range(3):
            _initialize[(triton.cdiv(b[c].numel(), self.block),)](
                b[c], parameters if p is None else p, diagonal[c], rhs[c], u[c], parameters,
                *b[c].shape, c, grid.spacing[c], p is not None, self.block,
                enable_fp_fusion=False)

    def sweep(self, u, rhs, diagonal, out, parameters, grid):
        for c in range(3):
            _sweep[(triton.cdiv(u[c].numel(), self.block),)](
                u[c], rhs[c], diagonal[c], out[c], parameters,
                *u[c].shape, c, *(h*h for h in grid.spacing), self.block,
                enable_fp_fusion=False)
