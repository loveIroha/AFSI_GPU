"""Equations, curved P2 interaction, native execution and restart acceptance."""
from dataclasses import replace,asdict
import json
import math
import numpy as np
import pytest
import torch
from afsi_torch.mac.grid import MACGrid,zero_normal,gradient,divergence,velocity_laplacian
from afsi_torch.mac.cnab import CNABOptions,MACCNABFlow
from afsi_torch.mac.adaptive_transfer import AdaptiveP1Transfer,InteractionQuadratureOptions
from afsi_torch.tetrahedron import reference_nodes,EDGES
from afsi_torch.solid import prepare_p2

DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dt',[.001,.04])
def test_fgmres_manufactured_wall_stokes_and_pressure(device,dt):
    grid=MACGrid((4,)*3,(1.,1.2,1.4))
    flow=MACCNABFlow(grid,dt=dt,rho=1.3,mu=.7,device=device,
        cnab_options=CNABOptions(stokes_solver='fgmres',max_stokes_iterations=100))
    gen=torch.Generator(device=device).manual_seed(39)
    raw=zero_normal(tuple(torch.randn(v.shape,device=device,dtype=v.dtype,generator=gen) for v in grid.zeros(device=device)))
    exact=flow.project(raw).velocity
    p=torch.randn(grid.shape,device=device,dtype=torch.float64,generator=gen);p-=p.mean()
    gp=gradient(p,grid.spacing)
    b=tuple(u-flow.alpha*velocity_laplacian(u,c,grid.spacing)+dt/flow.rho*g for c,(u,g) in enumerate(zip(exact,gp)))
    result=flow.stokes(b,initial=torch.ones_like(p)*3)
    for actual,want in zip(result.velocity,exact):
        torch.testing.assert_close(actual,want,atol=2e-9,rtol=2e-9)
    torch.testing.assert_close(result.pressure,p,atol=2e-6,rtol=2e-6)
    info=result.diagnostics['stokes']
    assert info['solver']=='fgmres' and info['momentum_residual']<=info['momentum_tolerance']
    assert info['divergence_norm']<=info['divergence_tolerance']


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('native',[False,True])
def test_adaptive_curved_p2_mass_partition_force_torque_and_power(device,native):
    if native and device!='cuda':pytest.skip('native GPU only')
    X=reference_nodes(device=device)*.3+1.8
    cells=torch.arange(10,device=device)[None]
    geometry=prepare_p2(X,cells)
    grid=MACGrid((8,)*3,(5.,)*3)
    transfer=AdaptiveP1Transfer(grid,geometry,fused=False,
        ib_backend='cuda' if native else 'reference',
        quadrature_options=InteractionQuadratureOptions(mode='adaptive',point_density=3.,
            rule_family='xiao-gimbutas' if native else 'conical'))
    x=X.clone();x[4,1]+=.06
    stencil=transfer.prepare(x)
    # Project the identity coordinate field with the independently assembled
    # adaptive rule. P2 mass must remain exact in reference coordinates.
    rhs=torch.zeros_like(X)
    for group in stencil.rule.groups:
        local=torch.einsum('qa,eq,eqi->eai',group.values,group.weights,
            torch.einsum('qa,eai->eqi',group.values,X[group.cells]))
        rhs.index_add_(0,group.cells.reshape(-1),local.reshape(-1,3))
        torch.testing.assert_close(group.values.sum(-1),torch.ones_like(group.values[:,0]),atol=1e-14,rtol=0)
    torch.testing.assert_close(rhs,transfer.mass_action(X),atol=1e-16,rtol=1e-12)
    b=torch.sin(2*X)
    u=tuple(torch.sin(grid.coordinates(c,device=device)[...,(c+1)%3]) for c in range(3))
    f,_=transfer.spread(b,stencil)
    U,_=transfer.interpolate(u,stencil)
    solid=(b*U).sum();fluid=grid.volume*sum((a*d).sum() for a,d in zip(u,f))
    torch.testing.assert_close(solid,fluid,atol=1e-10,rtol=1e-11)
    total=torch.stack(tuple(grid.volume*a.sum() for a in f))
    torch.testing.assert_close(total,b.sum(0),atol=2e-10,rtol=1e-11)
    moment=sum((grid.volume*torch.linalg.cross(grid.coordinates(c,device=device),
        torch.nn.functional.one_hot(torch.tensor(c,device=device),3).to(X.dtype)*f[c][...,None])).sum((0,1,2)) for c in range(3))
    torch.testing.assert_close(moment,torch.linalg.cross(x,b).sum(0),atol=3e-10,rtol=1e-11)
    # An old stencil owns its rule after a later curved element needs more points.
    moved=x.clone();moved[4,1]+=.3
    newer=transfer.prepare(moved)
    assert newer.rule.point_count>=stencil.rule.point_count
    again,_=transfer.interpolate(u,stencil)
    torch.testing.assert_close(again,U,atol=1e-12,rtol=1e-11)


def test_cn_configuration_avoids_explicit_viscosity_veto_and_json(tmp_path):
    from afsi_torch.config import LVSimulationConfig,FluidConfig,TimeConfig,load_config,save_config
    from afsi_torch.mac.implicit import MACCouplingOptions
    cfg=LVSimulationConfig(fluid=FluidConfig((128,)*3),time=TimeConfig(.001,.002),
        coupling=MACCouplingOptions(scheme='cnab-midpoint',cnab=CNABOptions(stokes_solver='fgmres',advection='xsppm7')),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',point_density=3.))
    path=tmp_path/'config.json';save_config(path,cfg)
    assert load_config(path)==cfg
    with pytest.raises(ValueError,match='explicit viscous'):
        replace(cfg,coupling=MACCouplingOptions())


@pytest.mark.parametrize('device',DEVICES)
def test_xsppm7_axis_permutation_including_wall_ghosts(device):
    from afsi_torch.mac.xsppm7 import convection_xsppm7
    grid=MACGrid((6,8,10),(1.,1.3,1.7))
    rng=torch.Generator(device=device).manual_seed(418)
    u=zero_normal(tuple(torch.randn(v.shape,device=device,dtype=v.dtype,generator=rng) for v in grid.zeros(device=device)))
    original=convection_xsppm7(u,grid.spacing)
    order=(2,0,1)
    permuted=tuple(u[c].permute(order).contiguous() for c in order)
    changed=convection_xsppm7(permuted,tuple(grid.spacing[c] for c in order))
    for a,c in zip(changed,order):
        torch.testing.assert_close(a,original[c].permute(order),atol=2e-13,rtol=5e-13)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('backend',['torch','workspace'])
def test_inexact_mg_override_does_not_relax_outer_solver_options(device,backend):
    from afsi_torch.mac.multigrid import GeometricMultigrid,MGOptions
    from afsi_torch.mac.grid import negative_laplacian
    grid=MACGrid((8,)*3,(1.,)*3)
    options=MGOptions(rtol=1e-10)
    solver=GeometricMultigrid(grid,device=device,options=options,backend=backend)
    rhs=torch.sin(torch.arange(512,device=device,dtype=torch.float64)).reshape(grid.shape);rhs-=rhs.mean()
    loose,il=solver.solve(rhs,rtol=.01)
    assert solver.options==options
    tight,it=solver.solve(rhs)
    assert il['tolerance']>it['tolerance'] and il['cycles']<=it['cycles']
    assert torch.linalg.vector_norm(rhs-negative_laplacian(tight,grid.spacing)).item()<=it['tolerance']
    assert torch.linalg.vector_norm(rhs-negative_laplacian(loose,grid.spacing)).item()<=il['tolerance']


def test_xsppm7_smooth_advective_derivative_spatial_convergence():
    from afsi_torch.mac.xsppm7 import convection_xsppm7
    errors=[]
    for n in (16,32):
        grid=MACGrid((n,)*3,(1.,)*3)
        u=list(grid.zeros())
        xyz=grid.coordinates(0)
        u[0]=torch.sin(math.pi*xyz).prod(-1)
        actual=convection_xsppm7(tuple(u),grid.spacing)[0]
        exact=math.pi*torch.sin(math.pi*xyz[...,0])*torch.cos(math.pi*xyz[...,0])*torch.sin(math.pi*xyz[...,1]).square()*torch.sin(math.pi*xyz[...,2]).square()
        errors.append(torch.sqrt((actual-exact).square().mean()).item())
    assert errors[0]/errors[1]>3.2


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_xsppm7_compiled_nonzero_operator_matches_torch():
    from afsi_torch.mac.xsppm7 import convection_xsppm7
    from afsi_torch.mac.execution import tensor_kernel
    grid=MACGrid((8,)*3,(1.,1.3,1.7))
    rng=torch.Generator(device='cuda').manual_seed(81)
    u=zero_normal(tuple(torch.randn(v.shape,device='cuda',dtype=v.dtype,generator=rng) for v in grid.zeros(device='cuda')))
    expected=convection_xsppm7(u,grid.spacing)
    actual=tensor_kernel(convection_xsppm7,'cuda')(u,grid.spacing)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-12)


@pytest.mark.parametrize('device',DEVICES)
def test_p2_midpoint_actual_equations_checkpoint_and_continuation(device,tmp_path):
    from afsi_torch.config import LVSimulationConfig,TimeConfig,FluidConfig,OutputConfig,LVExecutionConfig
    from afsi_torch.mac.implicit import MACCouplingOptions
    from afsi_torch.mac.execution import build_driver
    from afsi_torch.mac.checkpoint import load_mac,save_mac
    from afsi_torch.simulation.lv_mac import run
    cfg=replace(LVSimulationConfig(),time=TimeConfig(5e-5,1e-4),
        geometry=replace(LVSimulationConfig().geometry,mesh_size=.5),fluid=FluidConfig((16,)*3),
        coupling=MACCouplingOptions(scheme='cnab-midpoint',cnab=CNABOptions(stokes_solver='fgmres',advection='xsppm7',max_stokes_iterations=100)),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',point_density=3.,max_order=12),
        output=OutputConfig(1,1,1,False))
    folder=tmp_path/'part'
    report=run(case_config=cfg,device=device,output=folder,end_time=5e-5)
    assert report['method_alignment']['stokes_solver']=='fgmres'
    model,state,settings,progress=load_mac(folder/'checkpoint.npz',device)
    assert state.previous_advection is not None and state.force_time==state.time
    driver=build_driver(model,settings,device)
    before=state.x.clone();dt=cfg.time.dt
    # Independently replay the equation at the second step's predicted midpoint.
    st_n=driver.transfer.prepare(state.x)
    U_n,_=driver.transfer.interpolate(state.velocity,st_n)
    half=state.x+.5*dt*U_n
    st_half=driver.transfer.prepare(half)
    force=model.force(half,state.time+.5*dt)
    density,_=driver.transfer.spread(force,st_half)
    current=driver.flow.advection(state.velocity)
    adv=tuple(1.5*a-.5*b for a,b in zip(current,state.previous_advection))
    expected_flow=driver.flow.advance(state.velocity,density,adv,state.pressure)
    average=tuple(.5*(a+b) for a,b in zip(state.velocity,expected_flow.velocity))
    U_half,_=driver.transfer.interpolate(average,st_half)
    expected_x=before+dt*U_half
    accepted,info=driver.step(state)
    torch.testing.assert_close(accepted.x,expected_x,atol=3e-13,rtol=1e-12)
    assert not info['startup_predictor_corrector'] and info['used_force_time_s']==pytest.approx(7.5e-5)
    assert info['power_error']<=1e-9*max(abs(info['solid_power']),1.)
    run(resume=folder/'checkpoint.npz',device=device,end_time=1e-4)
    _,continued,_,_=load_mac(folder/'checkpoint.npz',device)
    torch.testing.assert_close(continued.x,accepted.x,atol=3e-13,rtol=1e-12)
    for a,b in zip(continued.velocity,accepted.velocity):
        torch.testing.assert_close(a,b,atol=3e-11,rtol=1e-8)
    # The accepted state and its history were not mutated by replay/continuation.
    torch.testing.assert_close(state.x,before,atol=0,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_xsppm7_matches_independent_native_fortran_reconstruction(device):
    # The scalar oracle is compiled only on Linux validation servers; on other
    # platforms polynomial/reconstruction tests still exercise the operator.
    import ctypes,subprocess,shutil,tempfile
    from pathlib import Path
    from afsi_torch.mac.xsppm7 import xsppm7_states,_odd_pad,convection_xsppm7
    if not shutil.which('gfortran'):pytest.skip('Fortran oracle compiler unavailable')
    source=Path(__file__).parent/'reference/xsppm7_oracle.f90'
    with tempfile.TemporaryDirectory() as d:
        library=Path(d)/'oracle.so'
        subprocess.run(['gfortran','-O0','-shared','-fPIC','-ffree-line-length-none',str(source),'-o',str(library)],check=True)
        fun=ctypes.CDLL(str(library)).xsppm7_oracle
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        fun.argtypes=[ctypes.c_int,ptr,ptr,ptr]
        gen=torch.Generator().manual_seed(73)
        for component in (0,1):
            for q in (torch.randn(32,dtype=torch.float64,generator=gen),torch.sin(torch.arange(32,dtype=torch.float64)*.07),torch.ones(32,dtype=torch.float64)):
                if component==0:q[[0,-1]]=0
                q=q.reshape(32,1,1)
                padded=_odd_pad(q,0,component,4).numpy().reshape(-1).copy()
                lo,hi=np.empty(33),np.empty(33)
                fun(32,padded,lo,hi)
                a,b=xsppm7_states(q.to(device),0,component)
                torch.testing.assert_close(a.cpu().reshape(-1),torch.from_numpy(lo),atol=3e-14,rtol=2e-13)
                torch.testing.assert_close(b.cpu().reshape(-1),torch.from_numpy(hi),atol=3e-14,rtol=2e-13)
        grid=MACGrid((8,)*3,(1.,)*3)
        assert all(v.count_nonzero()==0 for v in convection_xsppm7(grid.zeros(device=device),grid.spacing))
