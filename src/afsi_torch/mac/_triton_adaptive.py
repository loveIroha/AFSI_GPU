"""Fused affine-P1 quadrature transfers and cell-local spread reduction."""
import torch
import triton
import triton.language as tl
from ._triton_ib import _links
from .adaptive_cell import support_extent


@triton.jit
def _force(CELLS,N,W,F,e,q,C:tl.constexpr,NQ:tl.constexpr):
    i0 = tl.load(CELLS+4*e)
    i1 = tl.load(CELLS+4*e+1)
    i2 = tl.load(CELLS+4*e+2)
    i3 = tl.load(CELLS+4*e+3)
    f0,f1 = tl.load(F+3*i0+C),tl.load(F+3*i1+C)
    f2,f3 = tl.load(F+3*i2+C),tl.load(F+3*i3+C)
    n0 = tl.load(N+4*q,mask=q<NQ,other=0.)
    n1 = tl.load(N+4*q+1,mask=q<NQ,other=0.)
    n2 = tl.load(N+4*q+2,mask=q<NQ,other=0.)
    n3 = tl.load(N+4*q+3,mask=q<NQ,other=0.)
    weight = tl.load(W+e*NQ+q,mask=q<NQ,other=0.)
    return ((n0*f0+n1*f1)+n2*f2+n3*f3)*weight


@triton.jit
def _spread_points(BASE,PHI,CELLS,N,W,F,OUT,NE:tl.constexpr,NQ:tl.constexpr,
                   NY:tl.constexpr,NZ:tl.constexpr,C:tl.constexpr,VOLUME:tl.constexpr,BP:tl.constexpr):
    p = tl.program_id(0)*BP+tl.arange(0,BP)
    e,q = p//NQ,p%NQ
    # Padded points must not access cells outside this group.
    safe_e = tl.minimum(e,NE-1)
    f = _force(CELLS,N,W,F,safe_e,q,C,NQ)
    ids,weights = _links(BASE,PHI,p[:,None],tl.arange(0,64)[None,:],NY,NZ,NE*NQ)
    tl.atomic_add(OUT+ids,weights*f[:,None]/VOLUME,mask=p[:,None]<NE*NQ,sem='relaxed')


@triton.jit
def _spread_cells(BASE,PHI,CELLS,N,W,F,OUT,NQ:tl.constexpr,Q:tl.constexpr,
                  NY:tl.constexpr,NZ:tl.constexpr,C:tl.constexpr,VOLUME:tl.constexpr,
                  EXTENT:tl.constexpr,BN:tl.constexpr):
    e = tl.program_id(0)
    q = tl.arange(0,Q)
    p = e*NQ+q
    bx = tl.load(BASE+3*p,mask=q<NQ,other=2147483647).to(tl.int32)
    by = tl.load(BASE+3*p+1,mask=q<NQ,other=2147483647).to(tl.int32)
    bz = tl.load(BASE+3*p+2,mask=q<NQ,other=2147483647).to(tl.int32)
    k = tl.program_id(1)*BN+tl.arange(0,BN)
    gx = tl.min(bx,0)+k//(EXTENT*EXTENT)
    gy = tl.min(by,0)+(k//EXTENT)%EXTENT
    gz = tl.min(bz,0)+k%EXTENT
    ox,oy,oz = gx[:,None]-bx[None,:],gy[:,None]-by[None,:],gz[:,None]-bz[None,:]
    active = ((k[:,None]<EXTENT**3)&(q[None,:]<NQ)&
              (ox>=0)&(ox<4)&(oy>=0)&(oy<4)&(oz>=0)&(oz<4))
    # Clamp masked offsets as well so padded pointer arithmetic stays bounded.
    wx = tl.load(PHI+12*p[None,:]+tl.minimum(tl.maximum(ox,0),3),mask=active,other=0.)
    wy = tl.load(PHI+12*p[None,:]+4+tl.minimum(tl.maximum(oy,0),3),mask=active,other=0.)
    wz = tl.load(PHI+12*p[None,:]+8+tl.minimum(tl.maximum(oz,0),3),mask=active,other=0.)
    f = _force(CELLS,N,W,F,e,q,C,NQ)
    local = tl.sum(((wx*wy)*wz)*f[None,:],1)/VOLUME
    ids = (gx*NY+gy)*NZ+gz
    mask = (k<EXTENT**3)&(tl.sum(active.to(tl.int32),1)>0)
    tl.atomic_add(OUT+ids,local,mask=mask,sem='relaxed')


@triton.jit
def _gather_assemble(BASE,PHI,CELLS,N,W,U,RHS,NQ:tl.constexpr,Q:tl.constexpr,
                     NY:tl.constexpr,NZ:tl.constexpr,C:tl.constexpr):
    e = tl.program_id(0)
    # Split dense rules into small quadrature tiles; large Qx64 blocks can
    # spill into more shared memory than an RTX 4090 thread block permits.
    q = tl.program_id(1)*Q+tl.arange(0,Q)
    p = e*NQ+tl.minimum(q,NQ-1)
    ids,weights = _links(BASE,PHI,p[:,None],tl.arange(0,64)[None,:],NY,NZ,2147483647)
    u = tl.load(U+ids,mask=q[:,None]<NQ,other=0.)
    gathered = tl.sum(u*weights,1)
    w = tl.load(W+e*NQ+q,mask=q<NQ,other=0.)
    for a in tl.static_range(4):
        shape = tl.load(N+4*q+a,mask=q<NQ,other=0.)
        local = tl.sum((gathered*w)*shape,0)
        node = tl.load(CELLS+4*e+a)
        tl.atomic_add(RHS+3*node+C,local,sem='relaxed')


def spread(transfer, coefficient, stencil, *, reduced):
    result = transfer.grid.zeros(device=coefficient.device,dtype=coefficient.dtype)
    offset = 0
    with torch.cuda.device(coefficient.device):
        for group in stencil.rule.groups:
            ne,nq = group.weights.shape
            count = ne*nq
            extent = support_extent(group,transfer.quadrature_options.point_density)
            q = triton.next_power_of_2(nq)
            for c,out in enumerate(result):
                args = (stencil.base[c,offset:offset+count],stencil.phi[c,offset:offset+count],
                        group.cells,group.values,group.weights,coefficient,out)
                if reduced and q<=256 and extent<=8:
                    bn = max(8,min(64,2048//q))
                    _spread_cells[(ne,triton.cdiv(extent**3,bn))](*args,nq,q,*transfer.grid.face_shape(c)[1:],
                        c,transfer.grid.volume,extent,bn,num_warps=4,enable_fp_fusion=False)
                else:
                    _spread_points[(triton.cdiv(count,4),)](*args,ne,nq,*transfer.grid.face_shape(c)[1:],
                        c,transfer.grid.volume,4,enable_fp_fusion=False)
            offset += count
    return result


def assemble_velocity(transfer, velocity, stencil):
    rhs = torch.zeros_like(transfer.diagonal)
    offset = 0
    with torch.cuda.device(rhs.device):
        for group in stencil.rule.groups:
            ne,nq = group.weights.shape
            count = ne*nq
            q = triton.next_power_of_2(nq)
            if q>256:
                # Preserve full high-order quadrature without enormous thread blocks.
                from .compact_transfer import CompactStencil
                part = CompactStencil(stencil.base[:,offset:offset+count],stencil.phi[:,offset:offset+count])
                gathered = transfer.gather_grid(velocity,part)
                rhs.add_(transfer._assemble_kernel(gathered,group.values,group.cells,group.weights,transfer.diagonal))
            else:
                for c,u in enumerate(velocity):
                    tile = min(32,q)
                    _gather_assemble[(ne,triton.cdiv(nq,tile))](stencil.base[c,offset:offset+count],stencil.phi[c,offset:offset+count],
                        group.cells,group.values,group.weights,u.contiguous(),rhs,nq,tile,
                        *transfer.grid.face_shape(c)[1:],c,num_warps=4,enable_fp_fusion=False)
            offset += count
    return rhs
