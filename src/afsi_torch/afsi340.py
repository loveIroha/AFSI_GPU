"""Generated two-leaflet AFSI demo_340 geometry and independent FRH weak form.

Lengths and material parameters use cm-g-s, per unit out-of-plane thickness.
P2 triangles use vertices 0,1,2 followed by edges 01,02,12.
"""
from dataclasses import dataclass
from math import isfinite,sqrt
import numpy as np
import torch
from .triangle import tabulate
from .mac.execution import tensor_kernel


@dataclass(frozen=True)
class ValveConfig:
    width: float = .0212
    length: float = .7
    x_right: float = 2.
    height: float = 1.61
    mesh_size: float = .01
    C0: float = 2e5
    C1: float = 1e6
    kappa: float = 4e5
    beta: float = 1e8

    def __post_init__(self):
        if any(not isfinite(v) or v<=0 for v in self.__dict__.values()) or 2*self.length>=self.height:
            raise ValueError('positive finite valve parameters and an open reference gap required')


@dataclass(frozen=True)
class ValveMesh:
    X: torch.Tensor
    cells: torch.Tensor
    cell_tags: torch.Tensor
    roots: torch.Tensor
    root_tags: torch.Tensor
    vertex_count: int
    gmsh_version: str


@dataclass(frozen=True)
class TriangleGeometry:
    cells: torch.Tensor
    values: torch.Tensor
    gradients: torch.Tensor
    weights: torch.Tensor
    node_count: int


def triangle_rule(*,device='cpu',dtype=torch.float64):
    # Six positive points, exact for every polynomial of total degree <=4.
    a,b=.445948490915965,.108103018168070
    c,d=.091576213509771,.816847572980459
    points=torch.tensor([[a,a],[a,b],[b,a],[c,c],[c,d],[d,c]],device=device,dtype=dtype)
    weights=torch.tensor([.111690794839005]*3+[.054975871827661]*3,device=device,dtype=dtype)
    return points,weights


def generate_valve(config=None,*,device='cpu'):
    config=config or ValveConfig()
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError('valve generation requires its own Gmsh session')
    gmsh.initialize([],readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.option.setNumber('General.NumThreads',1)
        gmsh.model.add('afsi340_two_leaflets')
        surfaces=[]
        for y in (0.,config.height-config.length):
            surfaces.append(gmsh.model.occ.addRectangle(config.x_right-config.width,y,0,config.width,config.length))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin',config.mesh_size)
        gmsh.option.setNumber('Mesh.MeshSizeMax',config.mesh_size)
        gmsh.option.setNumber('Mesh.ElementOrder',1)
        gmsh.model.mesh.generate(2)
        connectivity=[]
        tags=[]
        for surface,marker in zip(surfaces,(1,11)):
            types,_,conn=gmsh.model.mesh.getElements(2,surface)
            if list(types)!=[2]:
                raise RuntimeError('P1 triangle Gmsh geometry required')
            triangles=np.asarray(conn[0]).reshape(-1,3)
            connectivity.append(triangles)
            tags.extend([marker]*len(triangles))
        raw=np.concatenate(connectivity)
        used=np.unique(raw)
        node_tags,coordinates,_=gmsh.model.mesh.getNodes()
        order=np.argsort(node_tags)
        positions=np.searchsorted(np.asarray(node_tags)[order],used)
        vertices=np.asarray(coordinates).reshape(-1,3)[order[positions],:2]
        triangles=np.searchsorted(used,raw)
        D=vertices[triangles[:,1:]]-vertices[triangles[:,:1]]
        flip=np.linalg.det(D)<0
        triangles[flip]=triangles[flip][:,[0,2,1]]
        version=gmsh.__version__
    finally:
        gmsh.finalize()
    edges=np.array([[0,1],[0,2],[1,2]])
    pairs=np.sort(triangles[:,edges].reshape(-1,2),axis=1)
    unique,inverse,counts=np.unique(pairs,axis=0,return_inverse=True,return_counts=True)
    n=len(vertices)
    X=np.concatenate((vertices,vertices[unique].mean(axis=1)))
    cells=np.concatenate((triangles,n+inverse.reshape(-1,3)),axis=1)
    roots,markers=[],[]
    for edge_id,(pair,count) in enumerate(zip(unique,counts)):
        if count!=1:
            continue
        y=vertices[pair,1]
        if np.all(np.abs(y)<1e-12):
            roots.append([*pair,n+edge_id]); markers.append(4)
        elif np.all(np.abs(y-config.height)<1e-12):
            roots.append([*pair,n+edge_id]); markers.append(15)
    if set(markers)!={4,15}:
        raise RuntimeError('missing lower/upper leaflet root')
    tensor=lambda a,dtype:torch.as_tensor(a,dtype=dtype,device=device)
    return ValveMesh(tensor(X,torch.float64),tensor(cells,torch.int64),tensor(tags,torch.int64),
        tensor(roots,torch.int64),tensor(markers,torch.int64),n,version)


def prepare_triangle(mesh):
    X,cells=mesh.X,mesh.cells
    points,weights=triangle_rule(device=X.device,dtype=X.dtype)
    N,dN=tabulate(points)
    vertices=X[cells[:,:3]]
    D=(vertices[:,1:]-vertices[:,:1]).transpose(-1,-2)
    determinant=torch.linalg.det(D)
    if not torch.isfinite(determinant).all() or (determinant<=0).any():
        raise ValueError('positive reference triangles required')
    gradients=torch.einsum('qaj,ejk->eqak',dN,torch.linalg.inv(D))
    return TriangleGeometry(cells,N,gradients,determinant[:,None]*weights,len(X))


def determinant(F):
    return F[...,0,0]*F[...,1,1]-F[...,0,1]*F[...,1,0]


def frh_energy(F,fiber,C0=2e5,C1=1e6,kappa=4e5):
    J=determinant(F)
    A=(F@fiber[...,None]).squeeze(-1)
    I1=F.square().sum((-2,-1))/J
    I4=A.square().sum(-1)/J
    # Retain the source's -3 constant even in 2D; it has no effect on stress.
    return .5*C0*(I1-3)+C1*(torch.exp(I4-1)-I4)+.5*kappa*(.5*(J.square()-1)-torch.log(J))


def frh_stress(F,fiber,C0=2e5,C1=1e6,kappa=4e5):
    J=determinant(F)
    cof=torch.stack((F[...,1,1],-F[...,1,0],-F[...,0,1],F[...,0,0]),-1).reshape_as(F)
    invT=cof/J[...,None,None]
    A=F[..., :,0]*fiber[...,0,None]+F[..., :,1]*fiber[...,1,None]
    I1=F.square().sum((-2,-1))/J
    I4=A.square().sum(-1)/J
    P=C0*F/J[...,None,None]-.5*C0*I1[...,None,None]*invT
    P+=C1*torch.expm1(I4-1)[...,None,None]*(2*A[..., :,None]*fiber[...,None,:]/J[...,None,None]-I4[...,None,None]*invT)
    return P+.5*kappa*(J.square()-1)[...,None,None]*invT


class ValveSolid:
    def __init__(self,mesh,config=None,*,fused=True):
        self.mesh,self.config=mesh,config or ValveConfig()
        self.geometry=prepare_triangle(mesh)
        sign=mesh.X.new_ones(len(mesh.cells))
        sign[mesh.cell_tags==11]=-1.
        self.fiber=torch.stack((torch.ones_like(sign),sign),-1)*sqrt(.5)
        self.fiber=self.fiber[:,None].expand(-1,self.geometry.weights.shape[1],-1)
        z,w=np.polynomial.legendre.leggauss(3)
        s=mesh.X.new_tensor((z+1)/2)
        self.root_values=torch.stack(((1-s)*(1-2*s),s*(2*s-1),4*s*(1-s)),-1)
        length=torch.linalg.vector_norm(mesh.X[mesh.roots[:,1]]-mesh.X[mesh.roots[:,0]],dim=-1)
        self.root_weights=length[:,None]*mesh.X.new_tensor(w/2)
        c=self.config
        self.probe_reference=mesh.X.new_tensor([[c.x_right-c.width/2,c.height-c.length+1e-4],
                                              [c.x_right-c.width/2,c.length-1e-4]])
        self.probe_cells,self.probe_values=self._locate(self.probe_reference)
        self._force_kernel=tensor_kernel(self._force,mesh.X.device) if fused else self._force
        self._F_kernel=tensor_kernel(self._F,mesh.X.device) if fused else self._F

    def _F(self,x):
        return torch.einsum('eai,eqaJ->eqiJ',x[self.mesh.cells],self.geometry.gradients)

    def validate(self,x):
        if x.shape!=self.mesh.X.shape or x.dtype!=self.mesh.X.dtype or x.device!=self.mesh.X.device:
            raise ValueError('valve coordinates must match mesh')
        J=determinant(self._F_kernel(x))
        if not torch.isfinite(J).all() or (J<=0).any():
            raise ValueError('nonpositive or nonfinite valve det(F)')

    def _force(self,x):
        g,c=self.geometry,self.config
        P=frh_stress(self._F(x),self.fiber,c.C0,c.C1,c.kappa)
        local=-torch.einsum('eq,eqiJ,eqaJ->eai',g.weights,P,g.gradients)
        result=torch.zeros_like(x).index_add(0,self.mesh.cells.reshape(-1),local.reshape(-1,2))
        root_displacement=x[self.mesh.roots]-self.mesh.X[self.mesh.roots]
        u=torch.einsum('qa,bai->bqi',self.root_values,root_displacement)
        spring=-c.beta*torch.einsum('bq,qa,bqi->bai',self.root_weights,self.root_values,u)
        return result.index_add(0,self.mesh.roots.reshape(-1),spring.reshape(-1,2))

    def force(self,x):
        return self._force_kernel(x)

    def _locate(self,points):
        vertices=self.mesh.X[self.mesh.cells[:,:3]].detach().cpu().numpy()
        D=(vertices[:,1:]-vertices[:,:1]).transpose(0,2,1)
        cells,values=[],[]
        for point in points.detach().cpu().numpy():
            rs=np.linalg.solve(D,(point-vertices[:,0])[...,None])[...,0]
            inside=(rs>=-1e-10).all(-1)&(rs.sum(-1)<=1+1e-10)
            if not inside.any():
                raise ValueError('reference probe is outside the leaflet')
            index=np.flatnonzero(inside)[0]
            cells.append(index)
            N,_=tabulate(points.new_tensor(rs[index:index+1]))
            values.append(N[0])
        return torch.tensor(cells,device=points.device),torch.stack(values)

    def probes(self,x):
        return torch.einsum('pa,pai->pi',self.probe_values,x[self.mesh.cells[self.probe_cells]])

    def diagnostics(self,x):
        F=self._F_kernel(x)
        J=determinant(F)
        c=self.config
        strain=(self.geometry.weights*(frh_energy(F,self.fiber,c.C0,c.C1,c.kappa)+.5*c.C0)).sum()
        u=torch.einsum('qa,bai->bqi',self.root_values,x[self.mesh.roots]-self.mesh.X[self.mesh.roots])
        spring=.5*c.beta*(self.root_weights*u.square().sum(-1)).sum()
        probe=self.probes(x)
        displacement=probe-self.probe_reference
        return dict(solid_area_cm2=(self.geometry.weights*J).sum().item(),minimum_detF=J.min().item(),
            maximum_detF=J.max().item(),strain_energy_per_thickness=strain.item(),spring_energy_per_thickness=spring.item(),
            upper_tip_dx_cm=displacement[0,0].item(),upper_tip_dy_cm=displacement[0,1].item(),
            lower_tip_dx_cm=displacement[1,0].item(),lower_tip_dy_cm=displacement[1,1].item(),
            probe_gap_cm=(probe[0,1]-probe[1,1]).item(),max_displacement_cm=(x-self.mesh.X).norm(dim=-1).max().item())
