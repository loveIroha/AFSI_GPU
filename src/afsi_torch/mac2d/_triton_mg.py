"""Contiguous FP64 channel stencils; no atomics or changed MG operations."""
import torch
import triton
import triton.language as tl


@triton.jit
def _apply(P,i,NX:tl.constexpr,NY:tl.constexpr,HX2:tl.constexpr,HY2:tl.constexpr):
    active=(i>=0)&(i<NX*NY)
    x,y=i//NY,i%NY
    p=tl.load(P+i,active,other=0.)
    xp=tl.load(P+i+NY,active&(x+1<NX),other=0.)
    xm=tl.load(P+i-NY,active&(x>0),other=0.)
    yp=tl.load(P+i+1,active&(y+1<NY),other=0.)
    ym=tl.load(P+i-1,active&(y>0),other=0.)
    a=-tl.where(x+1<NX,(xp-p)/HX2,0.)
    a+=tl.where(x>0,(p-xm)/HX2,0.)
    a-=tl.where(y+1<NY,(yp-p)/HY2,0.)
    a+=tl.where(y>0,(p-ym)/HY2,0.)
    a+=tl.where(x==NX-1,2*p/HX2,0.)
    return p,a


@triton.jit
def _stencil(P,RHS,DIAG,OUT,NX:tl.constexpr,NY:tl.constexpr,
             HX2:tl.constexpr,HY2:tl.constexpr,SMOOTH:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    p,a=_apply(P,i,NX,NY,HX2,HY2)
    r=tl.load(RHS+i,i<NX*NY,other=0.)-a
    if SMOOTH:
        d=tl.load(DIAG+i,i<NX*NY,other=1.)
        r=p+(2./3.)*r/d
    tl.store(OUT+i,r,i<NX*NY)


@triton.jit
def _residual_restrict(P,RHS,OUT,NX:tl.constexpr,NY:tl.constexpr,
                       HX2:tl.constexpr,HY2:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    active=i<(NX//2)*(NY//2)
    x,y=i//(NY//2),i%(NY//2)
    j=2*x*NY+2*y
    _,a00=_apply(P,j,NX,NY,HX2,HY2)
    _,a01=_apply(P,j+1,NX,NY,HX2,HY2)
    _,a10=_apply(P,j+NY,NX,NY,HX2,HY2)
    _,a11=_apply(P,j+NY+1,NX,NY,HX2,HY2)
    r00=tl.load(RHS+j,active,other=0.)-a00
    r01=tl.load(RHS+j+1,active,other=0.)-a01
    r10=tl.load(RHS+j+NY,active,other=0.)-a10
    r11=tl.load(RHS+j+NY+1,active,other=0.)-a11
    tl.store(OUT+i,((r00+r01)+r10+r11)*.25,active)


@triton.jit
def _coarse_value(C,x,y,NX:tl.constexpr,NY:tl.constexpr,active):
    # Neumann ghosts on left/top/bottom; odd Dirichlet ghost on outlet.
    sign=tl.where(x>=NX,-1.,1.)
    xx=tl.minimum(tl.maximum(x,0),NX-1)
    yy=tl.minimum(tl.maximum(y,0),NY-1)
    return sign*tl.load(C+xx*NY+yy,active,other=0.)


@triton.jit
def _prolong_add(C,P,NX:tl.constexpr,NY:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    active=i<NX*NY
    x,y=i//NY,i%NY
    ax,ay=x//2-1+x%2,y//2-1+y%2
    wx,wy=tl.where(x%2==0,.75,.25),tl.where(y%2==0,.75,.25)
    c00=_coarse_value(C,ax,ay,NX//2,NY//2,active)
    c01=_coarse_value(C,ax,ay+1,NX//2,NY//2,active)
    c10=_coarse_value(C,ax+1,ay,NX//2,NY//2,active)
    c11=_coarse_value(C,ax+1,ay+1,NX//2,NY//2,active)
    value=((c00*(1-wy)+c01*wy)*(1-wx)+(c10*(1-wy)+c11*wy)*wx)
    tl.store(P+i,tl.load(P+i,active,other=0.)+value,active)


class TritonKernels:
    name='triton-channel'
    block=256

    def smooth(self,p,rhs,diagonal,out,spacing):
        with torch.cuda.device(p.device):
            _stencil[(triton.cdiv(p.numel(),self.block),)](p,rhs,diagonal,out,*p.shape,
                *(h*h for h in spacing),SMOOTH=True,BLOCK=self.block,enable_fp_fusion=False)

    def residual(self,p,rhs,out,spacing):
        with torch.cuda.device(p.device):
            _stencil[(triton.cdiv(p.numel(),self.block),)](p,rhs,p,out,*p.shape,
                *(h*h for h in spacing),SMOOTH=False,BLOCK=self.block,enable_fp_fusion=False)

    def residual_restrict(self,p,rhs,out,spacing):
        with torch.cuda.device(p.device):
            _residual_restrict[(triton.cdiv(out.numel(),self.block),)](p,rhs,out,*p.shape,
                *(h*h for h in spacing),BLOCK=self.block,enable_fp_fusion=False)

    def prolong_add(self,coarse,fine):
        with torch.cuda.device(fine.device):
            _prolong_add[(triton.cdiv(fine.numel(),self.block),)](coarse,fine,*fine.shape,
                BLOCK=self.block,enable_fp_fusion=False)
