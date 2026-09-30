"""Staggered channel: prescribed inlet, no-slip walls, p=0 outlet."""
from dataclasses import dataclass
from math import isfinite,prod
import torch


@dataclass(frozen=True)
class ChannelGrid:
    shape: tuple = (256,64)
    lengths: tuple = (8.,1.61)

    def __post_init__(self):
        if len(self.shape)!=2 or any(type(n) is not int or n<4 for n in self.shape):
            raise ValueError('two integer cell counts >=4 required')
        if len(self.lengths)!=2 or any(not isfinite(v) or v<=0 for v in self.lengths):
            raise ValueError('two finite positive channel lengths required')

    @property
    def spacing(self):
        return tuple(L/n for L,n in zip(self.lengths,self.shape))

    @property
    def volume(self):
        return prod(self.spacing)  # area per unit out-of-plane thickness

    def face_shape(self,c):
        return tuple(n+(axis==c) for axis,n in enumerate(self.shape))

    def coordinates(self,c=None,*,device='cpu',dtype=torch.float64):
        shape=self.shape if c is None else self.face_shape(c)
        axes=[h*(torch.arange(n,device=device,dtype=dtype)+(0 if c==a else .5))
              for a,(n,h) in enumerate(zip(shape,self.spacing))]
        return torch.stack(torch.meshgrid(*axes,indexing='ij'),-1)

    def zeros(self,*,device='cpu',dtype=torch.float64):
        return tuple(torch.zeros(self.face_shape(c),device=device,dtype=dtype) for c in range(2))

    def check_velocity(self,velocity):
        if len(velocity)!=2:
            raise ValueError('two velocity components required')
        for c,u in enumerate(velocity):
            if u.shape!=self.face_shape(c) or u.dtype!=velocity[0].dtype or u.device!=velocity[0].device:
                raise ValueError('channel velocity shape/device/dtype mismatch')


def divergence(velocity,spacing):
    u,v=velocity
    return torch.diff(u,dim=0)/spacing[0]+torch.diff(v,dim=1)/spacing[1]


def gradient(p,spacing):
    hx,hy=spacing
    gx=p.new_zeros((p.shape[0]+1,p.shape[1]))
    gy=p.new_zeros((p.shape[0],p.shape[1]+1))
    gx[1:-1]=(p[1:]-p[:-1])/hx
    gx[-1]=-2*p[-1]/hx  # zero outlet pressure at a half-cell distance
    gy[:,1:-1]=(p[:,1:]-p[:,:-1])/hy
    return gx,gy


def negative_laplacian(p,spacing):
    hx,hy=spacing
    out=torch.zeros_like(p)
    dx=(p[1:]-p[:-1])/hx**2
    dy=(p[:,1:]-p[:,:-1])/hy**2
    out[:-1]-=dx
    out[1:]+=dx
    out[:,:-1]-=dy
    out[:,1:]+=dy
    out[-1]+=2*p[-1]/hx**2
    return out


def boundary(velocity,inlet):
    u,v=(a.clone() for a in velocity)
    u[0]=inlet
    v[:,0]=0
    v[:,-1]=0
    return u,v


def laplacian(velocity,spacing):
    u,v=velocity
    hx,hy=spacing
    # Odd tangential ghosts at no-slip boundaries. Outlet: zero normal
    # derivative for tentative velocity; projection can change outlet flux.
    ul=torch.cat((u[:1],u[:-1]),0)
    ur=torch.cat((u[1:],u[-2:-1]),0)
    ub=torch.cat((-u[:,:1],u[:,:-1]),1)
    ut=torch.cat((u[:,1:],-u[:,-1:]),1)
    vl=torch.cat((-v[:1],v[:-1]),0)
    vr=torch.cat((v[1:],v[-1:]),0)
    vb=torch.cat((-v[:,:1],v[:,:-1]),1)
    vt=torch.cat((v[:,1:],-v[:,-1:]),1)
    return ((ul-2*u+ur)/hx**2+(ub-2*u+ut)/hy**2,
            (vl-2*v+vr)/hx**2+(vb-2*v+vt)/hy**2)


def convection(velocity,spacing):
    u,v=velocity
    hx,hy=spacing
    ux=.5*(u[1:]+u[:-1])
    du=torch.zeros_like(u)
    du[1:-1]=torch.diff(ux.square(),dim=0)/hx
    du[-1]=(u[-1].square()-ux[-1].square())/hx
    vp=torch.cat((-v[:1],v,v[-1:]),0)
    ve=.5*(vp[1:]+vp[:-1])
    up=torch.cat((-u[:,:1],u,-u[:,-1:]),1)
    ue=.5*(up[:,1:]+up[:,:-1])
    du+=torch.diff(ve*ue,dim=1)/hy
    dv=torch.zeros_like(v)
    vy=.5*(v[:,1:]+v[:,:-1])
    dv[:,1:-1]=torch.diff(vy.square(),dim=1)/hy
    vp=torch.cat((-v[:1],v,v[-1:]),0)
    vc=.5*(vp[1:]+vp[:-1])
    dv+=torch.diff(vc*ue,dim=0)/hx
    return du,dv
