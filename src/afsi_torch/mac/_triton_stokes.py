"""Parallel cell-tiled CN Stokes residuals; prescribed normal walls are zero."""
from contextlib import nullcontext
import torch
import triton
import triton.language as tl


@triton.jit
def _cell_faces(U,V,W,i,j,k,NY:tl.constexpr,NZ:tl.constexpr,ACTIVE):
    u=tl.load(U+i*NY*NZ+j*NZ+k,ACTIVE,other=0.)
    v=tl.load(V+i*(NY+1)*NZ+j*NZ+k,ACTIVE,other=0.)
    w=tl.load(W+i*NY*(NZ+1)+j*(NZ+1)+k,ACTIVE,other=0.)
    return u,v,w


@triton.jit
def _start_partial(U,V,W,P,OUT,NX:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    active=index<NX*NY*NZ
    i=index//(NY*NZ);j=(index//NZ)%NY;k=index%NZ
    u,v,w=_cell_faces(U,V,W,i,j,k,NY,NZ,active)
    p=tl.load(P+index,active,other=0.)
    base=3*tl.program_id(0)
    # The omitted terminal face in each component is a prescribed zero wall.
    tl.store(OUT+base,tl.sum(u*u+v*v+w*w,0))
    tl.store(OUT+base+1,tl.sum(p,0))
    tl.store(OUT+base+2,tl.min(tl.where(active,tl.abs(p)<float('inf'),True).to(tl.int32),0))


@triton.jit
def _component(U,R,P,p,center,i,j,k,ACTIVE,NX:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,
               H0,H1,H2,H20,H21,H22,C:tl.constexpr,ALPHA,SCALE,B:tl.constexpr):
    if C==0:
        n0,n1,n2=NX+1,NY,NZ
        coordinate=i;previous=(i-1)*NY*NZ+j*NZ+k;h=H0
    elif C==1:
        n0,n1,n2=NX,NY+1,NZ
        coordinate=j;previous=i*NY*NZ+(j-1)*NZ+k;h=H1
    else:
        n0,n1,n2=NX,NY,NZ+1
        coordinate=k;previous=i*NY*NZ+j*NZ+k-1;h=H2
    index=i*n1*n2+j*n2+k
    lap=tl.full((B,),0,tl.float64) if center.dtype==tl.float64 else tl.full((B,),0,tl.float32)
    for axis in tl.static_range(3):
        if axis==0:q=i;n=n0;stride=n1*n2;h_squared=H20
        elif axis==1:q=j;n=n1;stride=n2;h_squared=H21
        else:q=k;n=n2;stride=1;h_squared=H22
        left=tl.load(U+index-stride,ACTIVE & (q>0),other=0.)
        right=tl.load(U+index+stride,ACTIVE & (q+1<n),other=0.)
        left=tl.where(q>0,left,-center)
        right=tl.where(q+1<n,right,-center)
        lap=lap+(left-2*center+right)/h_squared
    lap=tl.where(coordinate==0,0.,lap)
    previous_p=tl.load(P+previous,ACTIVE & (coordinate>0),other=0.)
    gp=tl.where(coordinate>0,(p-previous_p)/h,0.)
    rhs=tl.load(R+index,ACTIVE,other=0.)
    return center-ALPHA*lap+SCALE*gp-rhs


@triton.jit
def _residual_partial(U,V,W,P,B0,B1,B2,RESIDUAL,OUT,COEFFICIENTS,
                      NX:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,
                      B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    active=index<NX*NY*NZ
    i=index//(NY*NZ);j=(index//NZ)%NY;k=index%NZ
    u,v,w=_cell_faces(U,V,W,i,j,k,NY,NZ,active)
    p=tl.load(P+index,active,other=0.)
    alpha=tl.load(COEFFICIENTS);scale=tl.load(COEFFICIENTS+1);inverse_scale=tl.load(COEFFICIENTS+2)
    h0=tl.load(COEFFICIENTS+3);h1=tl.load(COEFFICIENTS+4);h2=tl.load(COEFFICIENTS+5)
    h20=tl.load(COEFFICIENTS+6);h21=tl.load(COEFFICIENTS+7);h22=tl.load(COEFFICIENTS+8)
    # Three MAC face lattices, without materializing Gp, Lu or momentum fields.
    m0=_component(U,B0,P,p,u,i,j,k,active,NX,NY,NZ,h0,h1,h2,h20,h21,h22,0,alpha,scale,B)
    m1=_component(V,B1,P,p,v,i,j,k,active,NX,NY,NZ,h0,h1,h2,h20,h21,h22,1,alpha,scale,B)
    m2=_component(W,B2,P,p,w,i,j,k,active,NX,NY,NZ,h0,h1,h2,h20,h21,h22,2,alpha,scale,B)
    right0=tl.load(U+(i+1)*NY*NZ+j*NZ+k,active,other=0.)
    right1=tl.load(V+i*(NY+1)*NZ+(j+1)*NZ+k,active,other=0.)
    right2=tl.load(W+i*NY*(NZ+1)+j*(NZ+1)+k+1,active,other=0.)
    div=(right0-u)/h0+(right1-v)/h1+(right2-w)/h2
    tl.store(RESIDUAL+index,-inverse_scale*div,active)
    base=2*tl.program_id(0)
    tl.store(OUT+base,tl.sum(tl.where(active,m0*m0+m1*m1+m2*m2,0.),0))
    tl.store(OUT+base+1,tl.sum(tl.where(active,div*div,0.),0))


def _context(p):
    return torch.cuda.device(p.device) if p.is_cuda else nullcontext()


def start_partial(b,p,partial,shape):
    with _context(p):
        _start_partial[(len(partial),)](*b,p,partial,*shape,256,
                                       num_warps=4,enable_fp_fusion=False)


def residual_partial(u,p,b,residual,partial,shape,coefficients):
    with _context(p):
        _residual_partial[(len(partial),)](*u,p,*b,residual,partial,coefficients,*shape,
            256,num_warps=4,enable_fp_fusion=False)


@triton.jit
def _pressure_input_partial(RHS,P,OUT,N:tl.constexpr,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B);active=index<N
    rhs=tl.load(RHS+index,active,other=0.);p=tl.load(P+index,active,other=0.)
    base=5*tl.program_id(0)
    tl.store(OUT+base,tl.sum(rhs,0));tl.store(OUT+base+1,tl.sum(tl.abs(rhs),0))
    tl.store(OUT+base+2,tl.sum(p,0))
    tl.store(OUT+base+3,tl.min(tl.where(active,tl.abs(rhs)<float('inf'),True).to(tl.int32),0))
    tl.store(OUT+base+4,tl.min(tl.where(active,tl.abs(p)<float('inf'),True).to(tl.int32),0))


@triton.jit
def _pressure_norm_partial(RHS,P,OUT,H2,NX:tl.constexpr,NY:tl.constexpr,NZ:tl.constexpr,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B);active=index<NX*NY*NZ
    i=index//(NY*NZ);j=(index//NZ)%NY;k=index%NZ
    p=tl.load(P+index,active,other=0.);rhs=tl.load(RHS+index,active,other=0.)
    lap=tl.full((B,),0,tl.float64) if p.dtype==tl.float64 else tl.full((B,),0,tl.float32)
    for axis in tl.static_range(3):
        if axis==0:q=i;n=NX;stride=NY*NZ
        elif axis==1:q=j;n=NY;stride=NZ
        else:q=k;n=NZ;stride=1
        left=tl.load(P+index-stride,active & (q>0),other=0.)
        right=tl.load(P+index+stride,active & (q+1<n),other=0.)
        h2=tl.load(H2+axis)
        lap=lap-tl.where(q+1<n,(right-p)/h2,0.)
        lap=lap+tl.where(q>0,(p-left)/h2,0.)
    residual=rhs-lap
    base=2*tl.program_id(0)
    tl.store(OUT+base,tl.sum(rhs*rhs,0))
    tl.store(OUT+base+1,tl.sum(tl.where(active,residual*residual,0.),0))


def pressure_input_partial(rhs,p,partial):
    with _context(p):
        _pressure_input_partial[(len(partial),)](rhs,p,partial,p.numel(),256,
                                               num_warps=4,enable_fp_fusion=False)


def pressure_norm_partial(rhs,p,partial,spacing_squared):
    with _context(p):
        _pressure_norm_partial[(len(partial),)](rhs,p,partial,spacing_squared,*p.shape,256,
                                              num_warps=4,enable_fp_fusion=False)
