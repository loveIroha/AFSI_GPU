"""Affine P1 coordinates and shared Peskin tables in one CUDA pass.

No point cloud or broadcast distance arrays are materialized. Each point/axis
lane generates the face and center tables. FP64 round-to-nearest sqrt/division
and disabled FMA preserve precision; algebraic kernel symmetry shares a radical.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['NE','TOTAL','OFFSET'])
def _prepare_group(X,CELLS,N,ORIGIN,SPACING,LIMITS,BASE,PHI,INVALID,
                   NE,NQ:tl.constexpr,TOTAL,OFFSET,B:tl.constexpr):
    lane=tl.program_id(0)*B+tl.arange(0,B)
    p,axis=lane//3,lane%3
    active=p<NE*NQ
    e,q=p//NQ,p%NQ
    position=tl.full((B,),0.,tl.float64)
    for a in tl.static_range(4):
        node=tl.load(CELLS+4*e+a,mask=active,other=0)
        shape=tl.load(N+4*q+a,mask=active,other=0.)
        value=tl.load(X+3*node+axis,mask=active,other=0.)
        position=position+shape*value
    origin=tl.load(ORIGIN+axis)
    spacing=tl.load(SPACING+axis)
    limit=tl.load(LIMITS+axis)
    physical=(position-origin)/spacing
    valid=(physical>=2.)&(physical<limit)
    # NaN and infinity also fail the range screen. Nothing is spread/gathered
    # before the caller checks this flag. Padded lanes cannot fail the screen.
    if tl.sum((active&~valid).to(tl.int32),0)>0:
        tl.atomic_or(INVALID,1,sem='relaxed')
    for lattice in tl.static_range(2):
        start=origin if lattice==0 else origin+.5*spacing
        scaled=(position-start)/spacing
        scaled=tl.where(valid,scaled,0.)
        lower=tl.floor(scaled-1.).to(tl.int64)
        f=scaled-lower.to(tl.float64)-1.
        # For offsets 0,1,2,3: |r|=1+f,f,1-f,2-f. All four
        # radicals equal sqrt(1+4*f-4*f*f), f in [0,1].
        radical=tl.sqrt(1.+4.*f-4.*(f*f))
        index=(lattice*TOTAL+OFFSET+p)*3+axis
        tl.store(BASE+index,lower,mask=active)
        # Power-of-two scaling is exactly equivalent to division by eight.
        tl.store(PHI+4*index,(3.-2.*f-radical)*.125,mask=active)
        tl.store(PHI+4*index+1,(3.-2.*f+radical)*.125,mask=active)
        tl.store(PHI+4*index+2,(1.+2.*f+radical)*.125,mask=active)
        tl.store(PHI+4*index+3,(1.+2.*f-radical)*.125,mask=active)


def prepare(transfer,x,rule):
    """Return owned tables; later prepares may not overwrite frozen IB geometry."""
    x=x.contiguous()
    total=rule.point_count
    base=torch.empty((2,total,3),device=x.device,dtype=torch.int64)
    phi=x.new_empty((2,total,3,4))
    invalid=torch.zeros((),device=x.device,dtype=torch.int32)
    offset=0
    with torch.cuda.device(x.device):
        for group in rule.groups:
            ne,nq=group.cells.shape[0],group.values.shape[0]
            _prepare_group[(triton.cdiv(ne*nq*3,128),)](
                x,group.cells,group.values,transfer.origin,transfer.spacing,
                transfer.limits,base,phi,invalid,ne,nq,total,offset,128,
                num_warps=4,enable_fp_fusion=False)
            offset+=ne*nq
    if invalid.item():
        raise ValueError('MAC IB support reaches a wall or is nonfinite; enlarge/refine the fluid box')
    return base,phi
