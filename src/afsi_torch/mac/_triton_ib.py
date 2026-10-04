"""Four-point Peskin MAC transfers without materializing 64-link arrays."""
import torch
import triton
import triton.language as tl


@triton.jit
def _links(BASE,PHI,p,k,NY:tl.constexpr,NZ:tl.constexpr,NP,
           SHARED:tl.constexpr=False,C:tl.constexpr=0):
    ox,oy,oz = k//16,(k//4)%4,k%4
    sx:tl.constexpr = int(C!=0) if SHARED else 0
    sy:tl.constexpr = int(C!=1) if SHARED else 0
    sz:tl.constexpr = int(C!=2) if SHARED else 0
    bx = tl.load(BASE+sx*NP*3+3*p,mask=p<NP,other=0)
    by = tl.load(BASE+sy*NP*3+3*p+1,mask=p<NP,other=0)
    bz = tl.load(BASE+sz*NP*3+3*p+2,mask=p<NP,other=0)
    wx = tl.load(PHI+sx*NP*12+12*p+ox,mask=p<NP,other=0.)
    wy = tl.load(PHI+sy*NP*12+12*p+4+oy,mask=p<NP,other=0.)
    wz = tl.load(PHI+sz*NP*12+12*p+8+oz,mask=p<NP,other=0.)
    return ((bx+ox)*NY+by+oy)*NZ+bz+oz,(wx*wy)*wz


@triton.jit
def _gather(BASE,PHI,U,OUT,NP:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,
            C:tl.constexpr,BP:tl.constexpr,SHARED:tl.constexpr=False):
    p = tl.program_id(0)*BP+tl.arange(0,BP)
    ids,w = _links(BASE,PHI,p[:,None],tl.arange(0,64)[None,:],NY,NZ,NP,SHARED,C)
    u = tl.load(U+ids,mask=p[:,None]<NP,other=0.)
    result = tl.sum(u*w,axis=1)
    tl.store(OUT+3*p+C,result,mask=p<NP)


@triton.jit
def _spread(BASE,PHI,F,OUT,NP:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,
            C:tl.constexpr,VOLUME:tl.constexpr,BP:tl.constexpr,SHARED:tl.constexpr=False):
    p = tl.program_id(0)*BP+tl.arange(0,BP)
    ids,w = _links(BASE,PHI,p[:,None],tl.arange(0,64)[None,:],NY,NZ,NP,SHARED,C)
    f = tl.load(F+3*p+C,mask=p<NP,other=0.)
    value = w*f[:,None]/VOLUME
    tl.atomic_add(OUT+ids,value,mask=p[:,None]<NP,sem='relaxed')


def gather(grid,velocity,stencil):
    count=stencil.base.shape[1]
    shared=getattr(stencil,'layout',None)=='shared'
    out=stencil.phi.new_empty((count,3))
    with torch.cuda.device(out.device):
        for c,u in enumerate(velocity):
            _gather[(triton.cdiv(count,4),)](stencil.base if shared else stencil.base[c],
                stencil.phi if shared else stencil.phi[c],u,out,
                count,*grid.face_shape(c)[1:],c,4,SHARED=shared,enable_fp_fusion=False)
    return out


def spread(grid,force,stencil):
    count=stencil.base.shape[1]
    shared=getattr(stencil,'layout',None)=='shared'
    result=grid.zeros(device=force.device,dtype=force.dtype)
    with torch.cuda.device(force.device):
        for c,out in enumerate(result):
            _spread[(triton.cdiv(count,4),)](stencil.base if shared else stencil.base[c],
                stencil.phi if shared else stencil.phi[c],force,out,
                count,*grid.face_shape(c)[1:],c,grid.volume,4,SHARED=shared,enable_fp_fusion=False)
    return result
