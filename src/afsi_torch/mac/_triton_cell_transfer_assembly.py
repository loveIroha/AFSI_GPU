"""Exact cell-resident quadrature contraction into bounded IB hash tables.

One program owns one cell. Quadrature bases, all four delta weights per axis,
FE shape values and volume weights are loaded outside the lattice-tile loop.
The same current integrand is reduced in the same quadrature order as the
site backend; only its scheduling and reuse of input data change.
"""
import triton
import triton.language as tl
from ._triton_transfer_assembly import _hash_add, _cached_add


@triton.jit
def _four(i, p0, p1, p2, p3):
    return tl.where(i==0,p0,tl.where(i==1,p1,tl.where(i==2,p2,tl.where(i==3,p3,0.))))


@triton.jit(do_not_specialize=['P','OFFSET'])
def _cell_entries(BASE,PHI,N,W,CELLS,LOW,WIDTH,PREFIX,KEY,VALUE,FLAG,
                  P,Q:tl.constexpr,C:tl.constexpr,SHARED:tl.constexpr,OFFSET,
                  NY:tl.constexpr,NZ:tl.constexpr,NF:tl.constexpr,
                  B:tl.constexpr,BQ:tl.constexpr,HASH:tl.constexpr,CAPACITY:tl.constexpr,
                  CACHED:tl.constexpr,MKEY,MVALUE,MCOUNT,MCAP:tl.constexpr):
    cell = tl.program_id(0).to(tl.int64)
    q = tl.arange(0,BQ)
    point = OFFSET+cell*Q+q
    lx = (0 if C==0 else 1) if SHARED else C
    ly = (0 if C==1 else 1) if SHARED else C
    lz = (0 if C==2 else 1) if SHARED else C
    bx = tl.load(BASE+(lx*P+point)*3,mask=q<Q,other=0)
    by = tl.load(BASE+(ly*P+point)*3+1,mask=q<Q,other=0)
    bz = tl.load(BASE+(lz*P+point)*3+2,mask=q<Q,other=0)
    # Each Q-vector is loaded once for the entire cell, then broadcast across
    # lattice sites. No site-by-Q global loads or cell-prefix binary searches.
    tx = ((lx*P+point)*3)*4
    ty = ((ly*P+point)*3+1)*4
    tz = ((lz*P+point)*3+2)*4
    x0 = tl.load(PHI+tx,mask=q<Q,other=0)
    x1 = tl.load(PHI+tx+1,mask=q<Q,other=0)
    x2 = tl.load(PHI+tx+2,mask=q<Q,other=0)
    x3 = tl.load(PHI+tx+3,mask=q<Q,other=0)
    y0 = tl.load(PHI+ty,mask=q<Q,other=0)
    y1 = tl.load(PHI+ty+1,mask=q<Q,other=0)
    y2 = tl.load(PHI+ty+2,mask=q<Q,other=0)
    y3 = tl.load(PHI+ty+3,mask=q<Q,other=0)
    z0 = tl.load(PHI+tz,mask=q<Q,other=0)
    z1 = tl.load(PHI+tz+1,mask=q<Q,other=0)
    z2 = tl.load(PHI+tz+2,mask=q<Q,other=0)
    z3 = tl.load(PHI+tz+3,mask=q<Q,other=0)
    weight = tl.load(W+cell*Q+q,mask=q<Q,other=0)
    n0 = tl.load(N+q*4,mask=q<Q,other=0)
    n1 = tl.load(N+q*4+1,mask=q<Q,other=0)
    n2 = tl.load(N+q*4+2,mask=q<Q,other=0)
    n3 = tl.load(N+q*4+3,mask=q<Q,other=0)
    node0 = tl.load(CELLS+cell*4)
    node1 = tl.load(CELLS+cell*4+1)
    node2 = tl.load(CELLS+cell*4+2)
    node3 = tl.load(CELLS+cell*4+3)
    lowx = tl.load(LOW+cell*3)
    lowy = tl.load(LOW+cell*3+1)
    lowz = tl.load(LOW+cell*3+2)
    sx = tl.load(WIDTH+cell*3)
    sy = tl.load(WIDTH+cell*3+1)
    sz = tl.load(WIDTH+cell*3+2)
    size = sx*sy*sz
    if not HASH:
        start = tl.load(PREFIX+cell)
    lanes = tl.arange(0,B)
    for tile in range(tl.cdiv(size,B)):
        relative = tile*B+lanes
        valid = relative<size
        gx = relative//(sy*sz)+lowx
        gy = (relative//sz)%sy+lowy
        gz = relative%sz+lowz
        px = _four(gx[:,None]-bx[None,:],x0[None,:],x1[None,:],x2[None,:],x3[None,:])
        py = _four(gy[:,None]-by[None,:],y0[None,:],y1[None,:],y2[None,:],y3[None,:])
        pz = _four(gz[:,None]-bz[None,:],z0[None,:],z1[None,:],z2[None,:],z3[None,:])
        kernel = (px*py)*pz*weight[None,:]
        fluid = (gx*NY+gy)*NZ+gz
        for a in tl.static_range(4):
            if a==0:
                shape,node = n0,node0
            elif a==1:
                shape,node = n1,node1
            elif a==2:
                shape,node = n2,node2
            else:
                shape,node = n3,node3
            value = tl.sum(kernel*shape[None,:],axis=1)
            key = node*NF+fluid
            if CACHED:
                _cached_add(KEY,VALUE,MKEY,MVALUE,MCOUNT,key,value,valid,CAPACITY,MCAP,128)
            elif HASH:
                _hash_add(KEY,VALUE,FLAG,key,value,valid,CAPACITY,128)
            else:
                tl.store(KEY+(start+relative)*4+a,key,mask=valid)
                tl.store(VALUE+(start+relative)*4+a,value,mask=valid)


def cell_entries(stencil,group,component,offset,low,width,prefix,count,face_shape,workspace,cache=None):
    """One cell per program; no full contribution arrays are materialized."""
    import torch
    q = len(group.values)
    # Bound site-by-Q intermediates while retaining the cell's Q-vectors.
    block = 8 if q<=128 else 4
    warps = 4 if q<=128 else 8
    cached = cache is not None
    with torch.cuda.device(group.weights.device):
        _cell_entries[(len(group.cells),)](stencil.base,stencil.phi,group.values,
            group.weights,group.cells,low,width,prefix,workspace.keys,workspace.values,workspace.flag,
            stencil.base.shape[1],q,component,getattr(stencil,'layout',None)=='shared',offset,
            face_shape[1],face_shape[2],face_shape[0]*face_shape[1]*face_shape[2],
            block,triton.next_power_of_2(q),True,workspace.capacity,cached,
            cache.missing_keys if cached else workspace.keys,
            cache.missing_values if cached else workspace.values,
            cache.missing_count if cached else workspace.flag,
            cache.missing_capacity if cached else 0,num_warps=warps,enable_fp_fusion=False)
