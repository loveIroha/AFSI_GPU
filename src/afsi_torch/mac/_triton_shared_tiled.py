"""Cell-owned vector gather and tile-local reduction of unchanged IB links.

Keep every quadrature point and Peskin neighbor. Gather assembles twelve nodal
entries per cell; spread sorts a small tile's links and combines equal grid IDs
before global FP64 atomics. No point-sized intermediate arrays are allocated.
"""
import torch
import triton
import triton.language as tl
from ._triton_ib import _links
from ._triton_adaptive import _force


@triton.jit
def _segment(k0,v0,k1,v1):
    return k1,tl.where(k0==k1,v0+v1,v1)


@triton.jit(do_not_specialize=['TOTAL','OFFSET'])
def _spread_reduced(BASE,PHI,CELLS,N,W,F,OUT,TOTAL,OFFSET,
                    NQ:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,C:tl.constexpr,
                    VOLUME:tl.constexpr,FACE_COUNT:tl.constexpr,P:tl.constexpr):
    e = tl.program_id(0)
    q = tl.program_id(1)*P+tl.arange(0,P)
    safe = tl.minimum(q,NQ-1)
    force = _force(CELLS,N,W,F,e,safe,C,NQ)
    ids,weight = _links(BASE,PHI,(OFFSET+e*NQ+safe)[:,None],
        tl.arange(0,64)[None,:],NY,NZ,TOTAL,True,C)
    count:tl.constexpr = P*64
    index = tl.arange(0,count)
    # Append the original lane index to obtain stable key/value sorting using
    # integer sort plus register gather. Padded links sort beyond the grid.
    if (FACE_COUNT+1)*count < 2147483648:
        ids = ids.to(tl.int32)
    keys = tl.reshape(tl.where(q[:,None]<NQ,ids,FACE_COUNT),(count,))*count+index
    values = tl.reshape(tl.where(q[:,None]<NQ,weight*force[:,None]/VOLUME,0.),(count,))
    keys = tl.sort(keys,descending=False)
    grid_ids = keys//count
    values = tl.gather(values,(keys%count).to(tl.int32),0)
    _,sums = tl.associative_scan((grid_ids,values),0,_segment)
    following = tl.gather(grid_ids,tl.minimum(index+1,count-1),0)
    emit = (grid_ids<FACE_COUNT)&((index==count-1)|(grid_ids!=following))
    tl.atomic_add(OUT+grid_ids,sums,mask=emit,sem='relaxed')


@triton.jit
def _gather_component(U,bx,by,bz,wx,wy,wz,valid,
                      NY:tl.constexpr,NZ:tl.constexpr,Q:tl.constexpr):
    axis = tl.arange(0,4)
    ids = (((bx[:,None,None,None]+axis[None,:,None,None])*NY+
            by[:,None,None,None]+axis[None,None,:,None])*NZ+
            bz[:,None,None,None]+axis[None,None,None,:])
    weight = (wx[:,:,None,None]*wy[:,None,:,None])*wz[:,None,None,:]
    velocity = tl.load(U+ids,valid[:,None,None,None],other=0.)
    return tl.sum(tl.reshape(velocity*weight,(Q,64)),1)


@triton.jit(do_not_specialize=['TOTAL','OFFSET'])
def _gather_cell(BASE,PHI,CELLS,N,W,U0,U1,U2,RHS,TOTAL,OFFSET,
                 NQ:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,Q:tl.constexpr):
    e = tl.program_id(0)
    node_axis = tl.arange(0,4)
    nodes = tl.load(CELLS+4*e+node_axis)
    acc0 = tl.full((4,),0.,RHS.dtype.element_ty)
    acc1 = tl.full((4,),0.,RHS.dtype.element_ty)
    acc2 = tl.full((4,),0.,RHS.dtype.element_ty)
    # Bounded live tensors even for the positive high-order fallback rules.
    for block in range(tl.cdiv(NQ,Q)):
        q = block*Q+tl.arange(0,Q)
        valid = q<NQ
        point = OFFSET+e*NQ+tl.minimum(q,NQ-1)
        bx0 = tl.load(BASE+3*point)
        by0 = tl.load(BASE+3*point+1)
        bz0 = tl.load(BASE+3*point+2)
        bx1 = tl.load(BASE+3*TOTAL+3*point)
        by1 = tl.load(BASE+3*TOTAL+3*point+1)
        bz1 = tl.load(BASE+3*TOTAL+3*point+2)
        ax = node_axis[None,:]
        wx0 = tl.load(PHI+12*point[:,None]+ax)
        wy0 = tl.load(PHI+12*point[:,None]+4+ax)
        wz0 = tl.load(PHI+12*point[:,None]+8+ax)
        wx1 = tl.load(PHI+12*TOTAL+12*point[:,None]+ax)
        wy1 = tl.load(PHI+12*TOTAL+12*point[:,None]+4+ax)
        wz1 = tl.load(PHI+12*TOTAL+12*point[:,None]+8+ax)
        weight = tl.load(W+e*NQ+q,valid,other=0.)
        shape = tl.load(N+4*q[:,None]+ax,valid[:,None],other=0.)
        g0 = _gather_component(U0,bx0,by1,bz1,wx0,wy1,wz1,valid,NY,NZ,Q)
        acc0 += tl.sum((g0*weight)[:,None]*shape,0)
        g1 = _gather_component(U1,bx1,by0,bz1,wx1,wy0,wz1,valid,NY+1,NZ,Q)
        acc1 += tl.sum((g1*weight)[:,None]*shape,0)
        g2 = _gather_component(U2,bx1,by1,bz0,wx1,wy1,wz0,valid,NY,NZ+1,Q)
        acc2 += tl.sum((g2*weight)[:,None]*shape,0)
    tl.atomic_add(RHS+3*nodes,acc0,sem='relaxed')
    tl.atomic_add(RHS+3*nodes+1,acc1,sem='relaxed')
    tl.atomic_add(RHS+3*nodes+2,acc2,sem='relaxed')


def spread(transfer,coefficient,stencil):
    result = transfer.grid.zeros(device=coefficient.device,dtype=coefficient.dtype)
    offset = 0
    with torch.cuda.device(coefficient.device):
        for group in stencil.rule.groups:
            ne,nq = group.weights.shape
            for c,out in enumerate(result):
                _spread_reduced[(ne,triton.cdiv(nq,4))](
                    stencil.base,stencil.phi,group.cells,group.values,group.weights,
                    coefficient,out,stencil.rule.point_count,offset,nq,*out.shape[1:],
                    c,transfer.grid.volume,out.numel(),4,num_warps=4,enable_fp_fusion=False)
            offset += ne*nq
    return result


def assemble_velocity(transfer,velocity,stencil):
    rhs = torch.zeros_like(transfer.diagonal)
    offset = 0
    with torch.cuda.device(rhs.device):
        for group in stencil.rule.groups:
            ne,nq = group.weights.shape
            _gather_cell[(ne,)](stencil.base,stencil.phi,group.cells,group.values,group.weights,
                *(u.contiguous() for u in velocity),rhs,stencil.rule.point_count,offset,
                nq,*transfer.grid.shape[1:],min(16,triton.next_power_of_2(nq)),
                num_warps=4,enable_fp_fusion=False)
            offset += ne*nq
    return rhs
