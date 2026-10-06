"""Parallel cell/quadrature contraction for the exact frozen IB matrix.

One program integrates a tile of cell/lattice pairs over all Gaussian
points. It emits four FE-vertex contributions per lattice site, rather
than materializing 64*4 contributions for every interaction point.
"""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['START','COUNT','P','E','OFFSET','SEARCH'])
def _entries(BASE,PHI,N,W,CELLS,LOW,WIDTH,PREFIX,KEY,VALUE,
             START,COUNT,P,Q:tl.constexpr,E,C:tl.constexpr,SHARED:tl.constexpr,OFFSET,
             NY:tl.constexpr,NZ:tl.constexpr,NF:tl.constexpr,
             SEARCH,B:tl.constexpr,BQ:tl.constexpr):
    local = tl.program_id(0)*B+tl.arange(0,B)
    site = START+local
    valid = local<COUNT
    left = tl.full((B,),0,tl.int64); right = tl.full((B,),0,tl.int64)+E
    for _ in range(SEARCH):
        middle = (left+right)//2
        end = tl.load(PREFIX+middle+1,mask=middle<E,other=9223372036854775807)
        advance = (end<=site)&(middle<E)
        left = tl.where(advance,middle+1,left)
        right = tl.where(advance,right,middle)
    cell = tl.minimum(left,E-1)
    relative = site-tl.load(PREFIX+cell)
    sy = tl.load(WIDTH+cell*3+1); sz = tl.load(WIDTH+cell*3+2)
    gx = relative//(sy*sz)+tl.load(LOW+cell*3)
    gy = (relative//sz)%sy+tl.load(LOW+cell*3+1)
    gz = relative%sz+tl.load(LOW+cell*3+2)
    q = tl.arange(0,BQ)
    point = OFFSET+cell[:,None]*Q+q[None,:]
    lattice_x = (0 if C==0 else 1) if SHARED else C
    lattice_y = (0 if C==1 else 1) if SHARED else C
    lattice_z = (0 if C==2 else 1) if SHARED else C
    ix = gx[:,None]-tl.load(BASE+(lattice_x*P+point)*3,mask=q[None,:]<Q,other=0)
    iy = gy[:,None]-tl.load(BASE+(lattice_y*P+point)*3+1,mask=q[None,:]<Q,other=0)
    iz = gz[:,None]-tl.load(BASE+(lattice_z*P+point)*3+2,mask=q[None,:]<Q,other=0)
    mask = valid[:,None]&(q[None,:]<Q)
    px = tl.load(PHI+((lattice_x*P+point)*3)*4+ix,mask=mask&(ix>=0)&(ix<4),other=0)
    py = tl.load(PHI+((lattice_y*P+point)*3+1)*4+iy,mask=mask&(iy>=0)&(iy<4),other=0)
    pz = tl.load(PHI+((lattice_z*P+point)*3+2)*4+iz,mask=mask&(iz>=0)&(iz<4),other=0)
    weight = tl.load(W+cell[:,None]*Q+q[None,:],mask=mask,other=0)
    kernel = (px*py)*pz*weight
    fluid = (gx*NY+gy)*NZ+gz
    for a in tl.static_range(4):
        shape = tl.load(N+q*4+a,mask=q<Q,other=0)
        value = tl.sum(kernel*shape[None,:],axis=1)
        node = tl.load(CELLS+cell*4+a)
        tl.store(KEY+local*4+a,node*NF+fluid,mask=valid)
        tl.store(VALUE+local*4+a,value,mask=valid)


def entries(stencil,group,component,offset,low,width,prefix,start,count,face_shape):
    import torch
    keys = torch.empty(count*4,device=group.weights.device,dtype=torch.int64)
    values = torch.empty(count*4,device=group.weights.device,dtype=group.weights.dtype)
    q = group.values.shape[0]
    # Keep FP64 quadrature reductions in a bounded register tile.
    block = 16 if q<=128 else 8
    with torch.cuda.device(group.weights.device):
        _entries[(triton.cdiv(count,block),)](stencil.base,stencil.phi,group.values,
            group.weights,group.cells,low,width,prefix,keys,values,
            start,count,stencil.base.shape[1],q,len(group.cells),component,
            getattr(stencil,'layout',None)=='shared',offset,face_shape[1],face_shape[2],
            face_shape[0]*face_shape[1]*face_shape[2],(len(group.cells)+1).bit_length()+1,
            block,triton.next_power_of_2(q),num_warps=4,enable_fp_fusion=False)
    return keys,values
