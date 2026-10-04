"""Shared-table P1 FE/IB fusion without point-force/point-velocity arrays.

All quadrature points and 64 Peskin neighbors are retained. Changing adaptive
group sizes/offsets are runtime values, not recompilation specialization keys.
No global scratch is reused: outputs and frozen stencils remain caller-owned.
"""
import torch
import triton
import triton.language as tl
from ._triton_ib import _links
from ._triton_adaptive import _force


@triton.jit(do_not_specialize=['NE','TOTAL','OFFSET'])
def _spread_shared(BASE,PHI,CELLS,N,W,F,OUT,NE,TOTAL,OFFSET,NQ:tl.constexpr,
                   NY:tl.constexpr,NZ:tl.constexpr,C:tl.constexpr,VOLUME:tl.constexpr,BP:tl.constexpr):
    local=tl.program_id(0)*BP+tl.arange(0,BP)
    active=local<NE*NQ
    safe=tl.minimum(local,NE*NQ-1)
    e,q=safe//NQ,safe%NQ
    force=_force(CELLS,N,W,F,e,q,C,NQ)
    ids,weight=_links(BASE,PHI,(OFFSET+safe)[:,None],tl.arange(0,64)[None,:],NY,NZ,TOTAL,True,C)
    tl.atomic_add(OUT+ids,weight*force[:,None]/VOLUME,mask=active[:,None],sem='relaxed')


@triton.jit(do_not_specialize=['TOTAL','OFFSET'])
def _gather_shared(BASE,PHI,CELLS,N,W,U,RHS,TOTAL,OFFSET,NQ:tl.constexpr,
                   NY:tl.constexpr,NZ:tl.constexpr,C:tl.constexpr,Q:tl.constexpr):
    e=tl.program_id(0)
    q=tl.program_id(1)*Q+tl.arange(0,Q)
    point=OFFSET+e*NQ+tl.minimum(q,NQ-1)
    ids,weight=_links(BASE,PHI,point[:,None],tl.arange(0,64)[None,:],NY,NZ,TOTAL,True,C)
    velocity=tl.load(U+ids,mask=q[:,None]<NQ,other=0.)
    gathered=tl.sum(velocity*weight,1)
    w=tl.load(W+e*NQ+q,mask=q<NQ,other=0.)
    weighted=gathered*w
    for a in tl.static_range(4):
        shape=tl.load(N+4*q+a,mask=q<NQ,other=0.)
        node=tl.load(CELLS+4*e+a)
        tl.atomic_add(RHS+3*node+C,tl.sum(weighted*shape,0),sem='relaxed')


def spread(transfer,coefficient,stencil):
    result=transfer.grid.zeros(device=coefficient.device,dtype=coefficient.dtype)
    offset=0
    with torch.cuda.device(coefficient.device):
        for group in stencil.rule.groups:
            ne,nq=group.weights.shape
            for c,out in enumerate(result):
                _spread_shared[(triton.cdiv(ne*nq,4),)](
                    stencil.base,stencil.phi,group.cells,group.values,group.weights,coefficient,out,
                    ne,stencil.rule.point_count,offset,nq,*transfer.grid.face_shape(c)[1:],
                    c,transfer.grid.volume,4,num_warps=4,enable_fp_fusion=False)
            offset+=ne*nq
    return result


def assemble_velocity(transfer,velocity,stencil):
    rhs=torch.zeros_like(transfer.diagonal)
    offset=0
    with torch.cuda.device(rhs.device):
        for group in stencil.rule.groups:
            ne,nq=group.weights.shape
            # Fixed small tiles also cover high-order fallback rules without a
            # huge thread block or a full point-velocity intermediate.
            tile=min(16,triton.next_power_of_2(nq))
            for c,u in enumerate(velocity):
                _gather_shared[(ne,triton.cdiv(nq,tile))](
                    stencil.base,stencil.phi,group.cells,group.values,group.weights,u.contiguous(),rhs,
                    stencil.rule.point_count,offset,nq,*transfer.grid.face_shape(c)[1:],
                    c,tile,num_warps=4,enable_fp_fusion=False)
            offset+=ne*nq
    return rhs
