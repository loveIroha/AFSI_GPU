"""2D material, mixed-boundary projection and adjoint FE/IB verification."""
from math import factorial
import pytest
import torch
from afsi_torch.afsi340 import ValveConfig,generate_valve,ValveSolid,triangle_rule,frh_energy,frh_stress
from afsi_torch.mac2d import ChannelGrid,ChannelFlow,divergence,gradient,negative_laplacian
from afsi_torch.mac2d.multigrid import ChannelMultigrid
from afsi_torch.mac2d.transfer import TriangleTransfer
from afsi_torch.mac2d.coupling import ValveStepper

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


def make_solid(device='cpu',fused=True):
    pytest.importorskip('gmsh')
    config=ValveConfig(mesh_size=.03)
    return ValveSolid(generate_valve(config,device=device),config,fused=fused)


def test_degree_four_triangle_rule():
    points,w=triangle_rule()
    for i in range(5):
        for j in range(5-i):
            exact=factorial(i)*factorial(j)/factorial(i+j+2)
            assert abs((w*points[:,0]**i*points[:,1]**j).sum().item()-exact)<2e-15


@pytest.mark.parametrize('device',DEVICES)
def test_frh_matches_energy_derivative_and_objectivity(device):
    F=torch.tensor([[[1.1,.2],[-.1,.94]],[[.97,-.15],[.21,1.08]]],device=device,dtype=torch.float64,requires_grad=True)
    f=F.new_tensor([[2**-.5,2**-.5],[2**-.5,-2**-.5]])
    expected=torch.autograd.grad(frh_energy(F,f).sum(),F)[0]
    actual=frh_stress(F,f)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-8)
    R=F.new_tensor([[.6,-.8],[.8,.6]])
    torch.testing.assert_close(frh_stress(R@F,f),R@actual,rtol=2e-12,atol=2e-8)
    torch.testing.assert_close(frh_energy(R@F,f),frh_energy(F,f),rtol=2e-12,atol=2e-8)
    I=torch.eye(2,device=device,dtype=F.dtype).expand_as(F)
    torch.testing.assert_close(frh_stress(I,f),torch.zeros_like(F),rtol=0,atol=2e-9)


@pytest.mark.parametrize('device',DEVICES)
def test_geometry_roots_and_solid_force_gradient(device):
    solid=make_solid(device)
    m,g=solid.mesh,solid.geometry
    assert set(m.cell_tags.tolist())=={1,11} and set(m.root_tags.tolist())=={4,15}
    assert abs(g.weights.sum().item()-2*.0212*.7)<1e-13
    torch.testing.assert_close(solid.probes(m.X),solid.probe_reference,rtol=0,atol=1e-14)
    x=(m.X+.0001*torch.sin(2*m.X)).requires_grad_()
    F=solid._F(x)
    u=torch.einsum('qa,bai->bqi',solid.root_values,x[m.roots]-m.X[m.roots])
    energy=(g.weights*frh_energy(F,solid.fiber)).sum()+.5*solid.config.beta*(solid.root_weights*u.square().sum(-1)).sum()
    expected=-torch.autograd.grad(energy,x)[0]
    torch.testing.assert_close(solid.force(x.detach()),expected,rtol=2e-9,atol=2e-7)
    with pytest.raises(ValueError,match='det'):
        solid.validate(m.X*0)


@pytest.mark.parametrize('device',DEVICES)
def test_channel_operator_mg_and_open_projection(device):
    grid=ChannelGrid((32,16))
    p=torch.sin(.7*grid.coordinates(device=device).sum(-1))+.4
    A=negative_laplacian(p,grid.spacing)
    torch.testing.assert_close(A,-divergence(gradient(p,grid.spacing),grid.spacing),rtol=2e-13,atol=2e-13)
    mg=ChannelMultigrid(grid,device=device)
    result,info=mg.solve(A)
    assert info['residual_norm']<=info['tolerance']
    torch.testing.assert_close(result,p,rtol=1e-8,atol=1e-8)
    flow=ChannelFlow(grid,device=device)
    vel=grid.zeros(device=device)
    result=flow.step(vel,grid.zeros(device=device),time=0.)
    torch.testing.assert_close(result.velocity[0][0],flow.inlet(0.),rtol=0,atol=0)
    assert result.velocity[1][:,0].count_nonzero()==0 and result.velocity[1][:,-1].count_nonzero()==0
    error=grid.spacing[1]*(result.velocity[0][-1].sum()-result.velocity[0][0].sum()).abs().item()
    assert error<1e-7
    assert divergence(result.velocity,grid.spacing).abs().max()<1e-7
    assert vel[0].count_nonzero()==0 and vel[1].count_nonzero()==0
    with pytest.raises(ValueError,match='stability'):
        flow.step(tuple(torch.full_like(v,100.) for v in vel),grid.zeros(device=device),time=0.)


@pytest.mark.parametrize('device',DEVICES)
def test_interior_transfer_force_torque_power_and_constant(device):
    solid=make_solid(device)
    grid=ChannelGrid((64,32),(8.,4.))
    transfer=TriangleTransfer(grid,solid.geometry,mass_backend='graph')
    x=solid.mesh.X+solid.mesh.X.new_tensor([0.,1.])
    stencil=transfer.prepare(x)
    force=torch.sin(3*x)
    density,fi=transfer.spread(force,stencil)
    velocity=tuple(torch.sin((c+1)*grid.coordinates(c,device=device).sum(-1)) for c in range(2))
    U,vi=transfer.interpolate(velocity,stencil)
    assert fi.residual_norm<=fi.tolerance and vi.residual_norm<=vi.tolerance
    torch.testing.assert_close(torch.stack([v.sum()*grid.volume for v in density]),force.sum(0),rtol=1e-10,atol=2e-9)
    torch.testing.assert_close((U*force).sum(),grid.volume*sum((u*f).sum() for u,f in zip(velocity,density)),rtol=1e-9,atol=2e-9)
    torque=grid.volume*((grid.coordinates(1,device=device)[...,0]*density[1]).sum()-(grid.coordinates(0,device=device)[...,1]*density[0]).sum())
    torch.testing.assert_close(torque,(x[:,0]*force[:,1]-x[:,1]*force[:,0]).sum(),rtol=1e-9,atol=2e-9)
    uniform=tuple(torch.full_like(v,c+1.) for c,v in enumerate(velocity))
    U,_=transfer.interpolate(uniform,stencil)
    torch.testing.assert_close(U,U.new_tensor([1.,2.]).expand_as(U),rtol=1e-9,atol=1e-9)
    with pytest.raises(ValueError,match='support'):
        transfer.prepare(x+10)


@pytest.mark.parametrize('device',DEVICES)
def test_wall_reflection_adjointness_and_no_slip(device):
    solid=make_solid(device)
    grid=ChannelGrid((64,16))
    t=TriangleTransfer(grid,solid.geometry,mass_backend='graph')
    stencil=t.prepare(solid.mesh.X)
    assert any((w<0).any() for w in stencil.weights)
    force=torch.sin(5*solid.mesh.X)
    density,_=t.spread(force,stencil)
    velocity=tuple(torch.sin(grid.coordinates(c,device=device).sum(-1)) for c in range(2))
    velocity[1][:,0]=0
    velocity[1][:,-1]=0
    U,_=t.interpolate(velocity,stencil)
    torch.testing.assert_close((U*force).sum(),grid.volume*sum((u*f).sum() for u,f in zip(velocity,density)),rtol=1e-9,atol=2e-8)


@pytest.mark.parametrize('device',DEVICES)
def test_short_coupled_valve_load(device):
    solid=make_solid(device)
    grid=ChannelGrid((64,16))
    driver=ValveStepper(ChannelFlow(grid,device=device),TriangleTransfer(grid,solid.geometry),solid)
    state=driver.initialize()
    for _ in range(12):
        state,info=driver.step(state,diagnostics=True)
    assert state.step==12 and state.time==12/16000
    assert solid.diagnostics(state.x)['upper_tip_dx_cm']>0
    assert info['power_relative_error']<1e-7
    assert info['flow']['inlet_time_s']==11/16000


def test_fullgraph_2d_trace(monkeypatch):
    import afsi_torch.afsi340 as solid_module
    import afsi_torch.mac2d.flow as flow_module
    import afsi_torch.mac2d.transfer as transfer_module
    import afsi_torch.mac2d.multigrid as mg_module
    for module in (solid_module,flow_module,transfer_module,mg_module):
        monkeypatch.setattr(module,'tensor_kernel',lambda f,d:torch.compile(f,backend='eager',fullgraph=True,dynamic=False))
    solid=make_solid()
    grid=ChannelGrid((32,8))
    driver=ValveStepper(ChannelFlow(grid),TriangleTransfer(grid,solid.geometry),solid)
    state=driver.initialize()
    state,info=driver.step(state,diagnostics=True)
    assert info['flow']['pressure']['residual_norm']<=info['flow']['pressure']['tolerance']


@pytest.mark.parametrize('device',DEVICES)
def test_checkpoint_continuation_and_vtk_fields(tmp_path,device):
    import numpy as np
    import meshio
    from afsi_torch.mac2d import checkpoint
    from afsi_torch.mac2d.output import fields,collection
    solid=make_solid(device)
    grid=ChannelGrid((32,8))
    settings=dict(dt=1/16000,nx=32,ny=8,rho=1.,mu=.1,mass_backend='graph',fused=True,warm_start=True)
    def driver(s):
        return ValveStepper(ChannelFlow(grid,device=device),TriangleTransfer(grid,s.geometry,mass_backend='graph'),s)
    original=driver(solid)
    state=original.initialize()
    for _ in range(4):
        state,_=original.step(state)
    path=tmp_path/'checkpoint.npz'
    checkpoint.save(path,solid,state,settings,dict(elapsed_seconds=0.,segments=[],frames=[],summary={}))
    restored,restart,loaded,_=checkpoint.load(path,device)
    assert loaded==settings
    continued=driver(restored)
    for _ in range(4):
        state,_=original.step(state)
        restart,_=continued.step(restart)
    torch.testing.assert_close(state.x,restart.x,rtol=2e-10,atol=2e-11)
    for a,b in zip(state.velocity,restart.velocity):
        torch.testing.assert_close(a,b,rtol=2e-8,atol=2e-9)
    frame=fields(tmp_path,restored,restart,grid,fluid=True)
    collection(tmp_path,[frame])
    output=meshio.read(tmp_path/frame['solid'])
    np.testing.assert_allclose(output.points[:,:2],restart.x.cpu().numpy())
    np.testing.assert_array_equal(output.cells_dict['triangle6'],solid.mesh.cells.cpu().numpy()[:,[0,1,2,3,5,4]])
    output=meshio.read(tmp_path/frame['fluid'])
    np.testing.assert_allclose(output.cell_data_dict['pressure_dyn_per_cm2']['quad'],restart.pressure.cpu().numpy().reshape(-1))
    assert (tmp_path/'solid.pvd').exists() and (tmp_path/'fluid.pvd').exists()
    with np.load(path,allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    data['x']=data['x']+.0001
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='checksum'):
        checkpoint.load(path,device)
