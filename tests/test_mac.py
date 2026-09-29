"""MAC manufactured fields, projection, and FE/IB conservation identities."""
from math import pi
import numpy as np
import pytest
import torch
from afsi_torch.mac import MACGrid, MACFlow, GeometricMultigrid, divergence, gradient
from afsi_torch.mac.grid import negative_laplacian, velocity_laplacian, convection, zero_normal
from afsi_torch.mac.transfer import FETransfer
from afsi_torch.mac.coupling import MACIBStepper
from afsi_torch.solid import prepare_p2, validate_deformation
from afsi_torch.tetrahedron import reference_nodes

DEVICES = ['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_separable_stencil_matches_expanded_kernel_on_shifted_anisotropic_grid(device):
    from afsi_torch.ib import peskin4
    X = .6*reference_nodes(device=device)-.2
    geometry = prepare_p2(X, torch.arange(10, device=device).reshape(1,10))
    grid = MACGrid((16,20,24), (4.,6.,8.), (-2.,-3.,-4.))
    transfer = FETransfer(grid, geometry)
    offsets = torch.cartesian_prod(*(torch.arange(4, device=device),)*3)
    # Exercise subcell offsets and curved P2 elements on all staggered lattices.
    for shift in (0., .125, -.31):
        x = X+shift
        x[4:] += .01*torch.sin(3*x[4:])
        stencil = transfer.prepare(x)
        points = transfer.interaction_points(x)
        for c in range(3):
            scaled = (points-points.new_tensor(grid.face_origin(c)))/points.new_tensor(grid.spacing)
            nodes = torch.floor(scaled-1).long()[:,None,:]+offsets[None,:,:]
            shape = grid.face_shape(c)
            ids = (nodes[...,0]*shape[1]+nodes[...,1])*shape[2]+nodes[...,2]
            weights = peskin4(scaled[:,None,:]-nodes.to(X.dtype)).prod(-1)
            torch.testing.assert_close(stencil.indices[c], ids, rtol=0, atol=0)
            torch.testing.assert_close(stencil.weights[c], weights, rtol=1e-14, atol=1e-16)
            torch.testing.assert_close(stencil.weights[c].sum(-1), torch.ones_like(points[:,0]),
                                       rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('device',DEVICES)
def test_neumann_mac_operator_and_summation_by_parts(device):
    grid=MACGrid((8,8,8),(2.,3.,4.))
    xyz=grid.coordinates(device=device)
    p=torch.sin(xyz[...,0]+.7*xyz[...,1]-.3*xyz[...,2])
    # Independent scalar NumPy Neumann Laplacian.
    a=p.cpu().numpy()
    expected=np.zeros_like(a)
    for index in np.ndindex(a.shape):
        for axis,h in enumerate(grid.spacing):
            for side in (-1,1):
                neighbor=list(index)
                neighbor[axis]+=side
                if 0<=neighbor[axis]<a.shape[axis]:
                    expected[index]+=(a[index]-a[tuple(neighbor)])/h**2
    np.testing.assert_allclose(negative_laplacian(p,grid.spacing).cpu(),expected,atol=2e-14)
    torch.testing.assert_close(negative_laplacian(p,grid.spacing),-divergence(gradient(p,grid.spacing),grid.spacing))
    u=zero_normal(tuple(torch.sin(grid.coordinates(c,device=device).sum(-1)) for c in range(3)))
    identity=(p*divergence(u,grid.spacing)).sum()+sum((v*g).sum() for v,g in zip(u,gradient(p,grid.spacing)))
    assert abs(identity.item())<1e-12


@pytest.mark.parametrize('device',DEVICES)
def test_multigrid_manufactured_cosine_and_projection(device):
    grid=MACGrid((16,16,16),(2.,2.,2.))
    xyz=grid.coordinates(device=device)
    exact=torch.cos(pi*xyz[...,0]/2)*torch.cos(2*pi*xyz[...,1]/2)*torch.cos(pi*xyz[...,2]/2)
    rhs=negative_laplacian(exact,grid.spacing)
    mg=GeometricMultigrid(grid,device=device)
    p,info=mg.solve(rhs)
    torch.testing.assert_close(p,exact,rtol=2e-8,atol=2e-9)
    assert info['residual_norm']<=info['tolerance'] and info['cycles']<80
    with pytest.raises(ValueError,match='incompatible'):
        mg.solve(torch.ones_like(rhs))
    flow=MACFlow(grid,dt=1e-4,device=device)
    u=zero_normal(tuple(torch.sin(grid.coordinates(c,device=device).sum(-1)*(c+1)) for c in range(3)))
    result=flow.project(u)
    assert divergence(result.velocity,grid.spacing).abs().max().item()<1e-8
    assert sum(v.square().sum() for v in result.velocity)<=sum(v.square().sum() for v in u)+1e-10
    repeated=flow.project(result.velocity)
    for a,b in zip(repeated.velocity,result.velocity):
        torch.testing.assert_close(a,b,rtol=1e-7,atol=1e-8)


@pytest.mark.parametrize('device',DEVICES)
def test_mac_viscous_eigenmode_and_centered_momentum_flux(device):
    grid=MACGrid((8,8,8),(1.,1.,1.))
    for c in range(3):
        xyz=grid.coordinates(c,device=device)
        u=torch.sin(pi*xyz).prod(-1)
        velocity=list(grid.zeros(device=device))
        velocity[c]=u
        u=zero_normal(velocity)[c]
        eigenvalue=sum(-4*np.sin(pi/(2*n))**2/h**2 for n,h in zip(grid.shape,grid.spacing))
        torch.testing.assert_close(velocity_laplacian(u,c,grid.spacing),eigenvalue*u,rtol=1e-11,atol=2e-12)
    velocity=tuple(grid.coordinates(c,device=device)[...,c] for c in range(3))
    for u,a in zip(velocity,convection(velocity,grid.spacing)):
        torch.testing.assert_close(a[2:-2,2:-2,2:-2],4*u[2:-2,2:-2,2:-2],atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('device',DEVICES)
def test_gradient_body_force_has_zero_projected_velocity(device):
    grid=MACGrid((16,16,16),(2.,2.,2.))
    flow=MACFlow(grid,dt=1e-4,device=device)
    xyz=grid.coordinates(device=device)
    p=torch.cos(pi*xyz[...,0]/2)*torch.cos(pi*xyz[...,2]/2)
    result=flow.step(grid.zeros(device=device),gradient(p,grid.spacing))
    torch.testing.assert_close(result.pressure,p,rtol=1e-8,atol=1e-9)
    assert max(u.abs().max().item() for u in result.velocity)<1e-11
    with pytest.raises(ValueError,match='viscous'):
        MACFlow(grid,dt=1.,device=device)


def transfer_setup(device):
    X=.6*reference_nodes(device=device)-.2
    geometry=prepare_p2(X,torch.arange(10,device=device).reshape(1,10))
    grid=MACGrid((16,16,16),(4.,4.,4.),(-2.,-2.,-2.))
    return X,geometry,grid,FETransfer(grid,geometry)


@pytest.mark.parametrize('device',DEVICES)
def test_mass_transfer_warm_start_preserves_solve_and_can_reset(device):
    X, geometry, grid, cold = transfer_setup(device)
    warm = FETransfer(grid, geometry, warm_start=True)
    with pytest.raises(ValueError, match='warm_start'):
        FETransfer(grid, geometry, warm_start=1)
    x = X.clone()
    x[4:] += .01*torch.sin(3*x[4:])
    stencil = cold.prepare(x)
    velocity = zero_normal(tuple(torch.sin(grid.coordinates(c,device=device).sum(-1))
                                 for c in range(3)))
    for k in range(3):
        force = torch.sin(2*X)+k*.0001*torch.cos(X)
        density_cold, _ = cold.spread(force,stencil)
        density_warm, force_info = warm.spread(force,stencil)
        U_cold, _ = cold.interpolate(velocity,stencil)
        U_warm, velocity_info = warm.interpolate(velocity,stencil)
        for a,b in zip(density_cold,density_warm):
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-8)
        torch.testing.assert_close(U_cold,U_warm,rtol=1e-8,atol=1e-8)
        assert force_info.residual_norm <= force_info.tolerance
        assert velocity_info.residual_norm <= velocity_info.tolerance
    warm.reset_warm_start()
    assert warm._force_coefficient is None and warm._velocity_coefficient is None


@pytest.mark.parametrize('device',DEVICES)
def test_quadrature_transfer_force_torque_power_and_constant_velocity(device):
    X,geo,grid,transfer=transfer_setup(device)
    x=X.clone()
    x[4:]+=.01*torch.sin(3*x[4:])
    stencil=transfer.prepare(x)
    b=torch.sin(torch.arange(30,device=device,dtype=X.dtype).reshape(10,3))
    density,info=transfer.spread(b,stencil)
    np.testing.assert_allclose(np.array([f.sum().item()*grid.volume for f in density]),b.sum(0).cpu(),atol=2e-11)
    # Torque comparison includes curved P2 geometry at the interaction points.
    torque=torch.zeros(3,device=device,dtype=X.dtype)
    for c,f in enumerate(density):
        vector=torch.zeros((*f.shape,3),device=device,dtype=X.dtype)
        vector[...,c]=f
        torque+=torch.linalg.cross(grid.coordinates(c,device=device),vector).sum((0,1,2))*grid.volume
    torch.testing.assert_close(torque,torch.linalg.cross(x,b).sum(0),atol=2e-11,rtol=1e-10)
    velocity=zero_normal(tuple(torch.sin(grid.coordinates(c,device=device).sum(-1)*(c+1)) for c in range(3)))
    U,info_u=transfer.interpolate(velocity,stencil)
    solid_power=(U*b).sum()
    fluid_power=sum((u*f).sum() for u,f in zip(velocity,density))*grid.volume
    torch.testing.assert_close(solid_power,fluid_power,atol=2e-11,rtol=1e-10)
    uniform=tuple(torch.full_like(f,c+1.) for c,f in enumerate(density))
    U,_=transfer.interpolate(uniform,stencil)
    torch.testing.assert_close(U,torch.tensor([1.,2.,3.],device=device).to(X).expand_as(X),atol=1e-10,rtol=1e-10)
    # Consistent mass is essential: P2 row-sum lumping is not positive.
    assert (transfer.mass_action(torch.ones_like(X))[:4]<0).all()
    with pytest.raises(ValueError,match='support'):
        transfer.prepare(x+10)


@pytest.mark.parametrize('device',DEVICES)
def test_mac_fe_coupling_keeps_lag_and_nonzero_motion(device):
    X,geo,grid,transfer=transfer_setup(device)
    flow=MACFlow(grid,dt=1e-4,device=device)
    times=[]
    def force(x,t):
        times.append(t)
        return torch.full_like(x,.001+t)
    driver=MACIBStepper(flow,transfer,force,lambda x:validate_deformation(x,geo))
    state=driver.initialize(X)
    first,_=driver.step(state)
    torch.testing.assert_close(first.x,X,rtol=0,atol=0)
    second,info=driver.step(first)
    assert times==[0.,flow.dt] and second.force_time==flow.dt
    assert (second.x-X).abs().max()>0
    assert info['power_error']<1e-12 and info['divergence_l2']<1e-9
    torch.testing.assert_close(state.x,X,rtol=0,atol=0)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('warm_start',[False,True])
def test_mac_lv_checkpoint_resume_matches_uninterrupted(tmp_path,device,warm_start):
    pytest.importorskip('gmsh')
    from examples.lv_mac import run
    from afsi_torch.mac.checkpoint import load_mac
    common=dict(device=device,mesh_size=.4,fluid_cells=16,log_every=2,checkpoint_every=2,
                warm_start=warm_start)
    whole=run(**common,output=tmp_path/'whole',end_time=.0003)
    run(**common,output=tmp_path/'split',end_time=.00015)
    resumed=run(device=device,resume=tmp_path/'split'/'checkpoint.npz',end_time=.0003,log_every=2)
    assert resumed['accepted_steps']==whole['accepted_steps']==6
    assert resumed['settings']['warm_start'] is warm_start
    _,a,_,_=load_mac(tmp_path/'whole'/'checkpoint.npz',device)
    _,b,_,_=load_mac(tmp_path/'split'/'checkpoint.npz',device)
    for name in ('x','force','pressure'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=1e-8,atol=1e-8)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,rtol=1e-8,atol=1e-10)
    assert whole['last']['divergence_l2']<1e-9


@pytest.mark.parametrize('device',DEVICES)
def test_shared_p2_nodes_and_refined_interaction_quadrature(device):
    from afsi_torch.tetrahedron import promote_p1
    X=torch.tensor([[0.,0.,0.],[.6,0.,0.],[0.,.6,0.],[0.,0.,.6],[.6,.6,.6]],device=device,dtype=torch.float64)-.2
    X,cells=promote_p1(X,torch.tensor([[0,1,2,3],[1,2,3,4]],device=device))
    grid=MACGrid((16,16,16),(4.,4.,4.),(-2.,-2.,-2.))
    b=torch.sin(3*X)
    u=tuple(torch.cos(grid.coordinates(c,device=device).sum(-1)) for c in range(3))
    for degree in (4,6):
        transfer=FETransfer(grid,prepare_p2(X,cells,degree=degree))
        stencil=transfer.prepare(X)
        f,_=transfer.spread(b,stencil)
        U,_=transfer.interpolate(u,stencil)
        torch.testing.assert_close((U*b).sum(),grid.volume*sum((v*g).sum() for v,g in zip(u,f)),atol=2e-11,rtol=2e-10)
