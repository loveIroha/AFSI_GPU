"""Hierarchical exact boolean support reduction; no single CTA scans the mesh."""
from contextlib import nullcontext
import torch
import triton
import triton.language as tl


@triton.jit
def _support_partial(X,O,H,L,OUT,N:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    active=i<N*3
    axis=i%3
    point=tl.load(X+(i//3)*S0+axis*S1,active,other=0.)
    origin=tl.load(O+axis)
    spacing=tl.load(H+axis)
    limit=tl.load(L+axis)
    if point.dtype==tl.float32:
        scaled=tl.div_rn(point-origin,spacing)
    else:
        scaled=(point-origin)/spacing
    valid=(tl.abs(point)<float('inf')) & (scaled>=2.) & (scaled<limit)
    tl.store(OUT+tl.program_id(0),tl.min(tl.where(active,valid,True).to(tl.int32),0))


@triton.jit
def _all_partial(IN,OUT,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    valid=tl.load(IN+i,i<N,other=1)
    tl.store(OUT+tl.program_id(0),tl.min(valid,0))


def support_flags(points,origin,spacing,limits):
    """Return an owned scalar boolean, with every point and strict bound tested."""
    n=len(points)
    if not n:return torch.ones((),device=points.device,dtype=torch.bool)
    context=torch.cuda.device(points.device) if points.is_cuda else nullcontext()
    with context:
        count=triton.cdiv(3*n,1024)
        partial=torch.empty(count,device=points.device,dtype=torch.int32)
        _support_partial[(count,)](points,origin,spacing,limits,partial,n,*points.stride(),1024,
                                  num_warps=4,enable_fp_fusion=False)
        while count>1:
            blocks=triton.cdiv(count,1024)
            reduced=torch.empty(blocks,device=points.device,dtype=torch.int32)
            _all_partial[(blocks,)](partial,reduced,count,min(1024,triton.next_power_of_2(count)),num_warps=4)
            partial,count=reduced,blocks
        return partial[0].to(torch.bool)
