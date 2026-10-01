"""RK3 time accuracy, transport stability, projection and FE/IB restart."""
from dataclasses import replace, asdict
import json
import math
import pytest
import torch
from afsi_torch.mac.grid import MACGrid, zero_normal, divergence, convection, velocity_laplacian, gradient
from afsi_torch.mac.flow import MACFlow, MACFlowResult, MACTransportGuardError
from afsi_torch.mac.rk3 import MACRK3Flow
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.transport import rk3_policy
from test_real_lv import real_case, DEVICES


@pytest.mark.parametrize('device',DEVICES)
def test_centered_transport_above_euler_A_screen_is_bounded_and_divergence_free(device):
    grid = MACGrid((8,)*3,(15.,)*3)
    rk = MACRK3Flow(grid,dt=1e-4,device=device)
    euler = MACFlow(grid,dt=1e-4,device=device)
    coords = [grid.coordinates(c,device=device) for c in range(3)]
    raw = zero_normal(tuple(torch.sin(2*math.pi*q[...,(c+1)%3]/15)*torch.sin(math.pi*q[...,c]/15)
                            for c,q in enumerate(coords)))
    u = rk.project(raw).velocity
    scale = 100/max(v.abs().max().item() for v in u)
    u = tuple(v*scale for v in u)
    zero = grid.zeros(device=device)
    with pytest.raises(MACTransportGuardError,match='advection_diffusion'):
        euler.step(u,zero)
    energy0 = sum(v.square().sum().item() for v in u)
    for _ in range(12):
        result = rk.step(u,zero)
        u = result.velocity
        assert torch.linalg.vector_norm(divergence(u,grid.spacing)).item() < 1e-7
        assert all(s['residual_norm'] <= s['tolerance'] for s in result.diagnostics['pressure']['stages'])
    assert sum(v.square().sum().item() for v in u) <= energy0*(1+1e-10)
    assert result.diagnostics['advection_diffusion_number'] > .25
    with pytest.raises(MACTransportGuardError,match='CFL'):
        rk.step(tuple(1000*v for v in u),zero)
    # A safe initial velocity does not exempt intermediate stages.
    with pytest.raises(MACTransportGuardError,match='stage 2'):
        rk.step(zero,tuple(1e8*v for v in raw))


def test_rk3_third_order_for_fluid_ode_and_imaginary_axis_stability():
    # Isolate the temporal integrator with an exactly soluble rotation;
    # centered nondissipative transport has imaginary Fourier eigenvalues.
    errors = []
    grid = MACGrid((4,)*3,(4.,)*3)
    for dt in (.1,.05,.025):
        flow = MACRK3Flow(grid,dt=dt,mu=.01)
        u = list(grid.zeros()); u[0][1,1,1] = 1.
        def predict(v,density):
            w = [a.clone() for a in v]
            w[0][1,1,1] = v[0][1,1,1]-dt*v[0][2,1,1]
            w[0][2,1,1] = v[0][2,1,1]+dt*v[0][1,1,1]
            return tuple(w)
        flow._predict = predict
        flow.project = lambda v,initial=None:MACFlowResult(v,u[0].new_zeros(grid.shape),
            dict(pressure=dict(cycles=0,residual_norm=0.,tolerance=1e-12,backend='test')))
        for _ in range(round(1/dt)):
            u = flow.step(tuple(u),grid.zeros()).velocity
        x,y = u[0][1,1,1].item(),u[0][2,1,1].item()
        errors.append(math.hypot(x-math.cos(1),y-math.sin(1)))
        assert x*x+y*y <= 1.
    assert errors[0]/errors[1] > 7.5 and errors[1]/errors[2] > 7.5
    # Screen's frozen-coefficient rectangle, not a nonlinear FSI proof.
    real = torch.linspace(-1,0,151,dtype=torch.float64)
    imag = torch.linspace(-.25,.25,151,dtype=torch.float64)
    z = real[:,None]+1j*imag[None,:]
    assert (1+z+z*z/2+z*z*z/6).abs().max() <= 1+1e-14


@pytest.mark.parametrize('device',DEVICES)
def test_rk3_weighted_pressure_satisfies_integrated_momentum(device):
    grid = MACGrid((8,)*3,(8.,)*3)
    flow = MACRK3Flow(grid,dt=1e-3,device=device)
    u = grid.zeros(device=device)
    density = tuple(torch.sin(grid.coordinates(c,device=device)[...,(c+1)%3]) for c in range(3))
    stages = []
    original = flow._stage
    def record(v,f,p,i):
        stages.append(v)
        return original(v,f,p,i)
    flow._stage = record
    new = flow.step(u,density)
    accelerations = []
    for v in stages:
        adv = convection(v,grid.spacing)
        accelerations.append(zero_normal(tuple(-a+flow.mu/flow.rho*velocity_laplacian(w,c,grid.spacing)+f/flow.rho
                            for c,(a,w,f) in enumerate(zip(adv,v,density)))))
    gradp = gradient(new.pressure,grid.spacing)
    for c in range(3):
        rhs = flow.dt*(accelerations[0][c]/6+accelerations[1][c]/6+2*accelerations[2][c]/3-gradp[c]/flow.rho)
        torch.testing.assert_close(new.velocity[c]-u[c],rhs,atol=1e-14,rtol=1e-10)


@pytest.mark.parametrize('device',DEVICES)
def test_fused_rk3_matches_reference(device):
    grid = MACGrid((8,)*3,(8.,)*3)
    reference = MACRK3Flow(grid,dt=1e-4,device=device)
    optimized = MACRK3Flow(grid,dt=1e-4,device=device,execution_backend='fused',
                           pressure_backend='graph' if device=='cuda' else 'workspace')
    density = tuple(torch.sin(grid.coordinates(c,device=device)[...,c]) for c in range(3))
    a,b = grid.zeros(device=device),grid.zeros(device=device)
    for _ in range(3):
        ra,rb = reference.step(a,density),optimized.step(b,density)
        a,b = ra.velocity,rb.velocity
        for v,w in zip(a,b):
            torch.testing.assert_close(v,w,atol=1e-11,rtol=1e-8)
        torch.testing.assert_close(ra.pressure,rb.pressure,atol=1e-9,rtol=1e-8)


def test_real_lv_switch_restart_policy_and_force_clock(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv, refine_checkpoint_dt
    from validation.diagnose_real_lv_guard import diagnose
    config = replace(real_case,coupling=MACCouplingOptions(scheme='implicit-newton'),
                     output=replace(real_case.output,write_vtk=False))
    source = tmp_path/'implicit'
    run(case_config=config,device='cpu',output=source)
    before = (source/'checkpoint.npz').read_bytes()
    folder = tmp_path/'rk3'
    report = run(device='cpu',resume=source/'checkpoint.npz',output=folder,
                 coupling_scheme='explicit-rk3',end_time=4e-4)
    assert report['completed'] and report['transport_policy']==rk3_policy()
    assert report['last']['force_time_s'] == pytest.approx(3e-4)
    assert report['last']['newton_iterations'] is None
    assert len(report['last_solver_info']['flow']['pressure']['stages']) == 3
    assert (source/'checkpoint.npz').read_bytes() == before
    model,state,settings,_,cfg = load_real_lv(folder/'checkpoint.npz')
    assert cfg.coupling.scheme == settings['coupling']['scheme'] == 'explicit-rk3'
    assert refine_checkpoint_dt(model,state,settings,cfg,5e-5)[3]['lagged_force_resampled']
    assert diagnose(folder/'checkpoint.npz')['transport_policy']==rk3_policy()
    assert run(device='cpu',resume=folder/'checkpoint.npz',end_time=5e-4)['completed']


def test_benchmark_counts_nested_solves_without_modifying_source(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    config = replace(real_case,output=replace(real_case.output,write_vtk=False))
    folder = tmp_path/'source'
    run(case_config=config,device='cpu',output=folder)
    path = folder/'checkpoint.npz'
    original = path.read_bytes()
    report = benchmark(path,device='cpu',warmup=1,steps=2)
    assert all(c['completed'] for c in report['cases'].values())
    rk = report['cases']['explicit-rk3']
    implicit = report['cases']['implicit-newton']
    assert rk['counts']['pressure_solves']==6 and rk['counts']['mass_solves']==4
    assert rk['newton_iterations']==rk['gmres_iterations']==0
    assert implicit['counts']['pressure_solves']>rk['counts']['pressure_solves']
    assert rk['end_time_s']==implicit['end_time_s'] and report['implicit_over_rk3_speedup']>0
    assert original==path.read_bytes()


@pytest.mark.parametrize('device',DEVICES)
def test_rk3_fe_ib_reference_and_optimized_match(real_case,device):
    from afsi_torch.real_lv import RealLVConfig, imported_model
    from afsi_torch.mac.execution import build_driver
    from test_mac_implicit import settings
    config = replace(real_case,coupling=MACCouplingOptions(scheme='explicit-rk3'))
    model = imported_model(config,device)
    reference = build_driver(model,settings(config),device)
    execution = RealLVConfig().execution
    if device=='cpu':
        execution=replace(execution,pressure_backend='workspace')
    optimized = build_driver(model,settings(replace(config,execution=execution)),device)
    a,b = reference.initialize(model.mesh.X),optimized.initialize(model.mesh.X)
    for _ in range(3):
        a,ia = reference.step(a)
        b,ib = optimized.step(b)
        torch.testing.assert_close(a.x,b.x,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(a.force,b.force,rtol=1e-8,atol=1e-7)
        torch.testing.assert_close(a.pressure,b.pressure,rtol=1e-8,atol=1e-8)
        for u,v in zip(a.velocity,b.velocity):
            torch.testing.assert_close(u,v,rtol=1e-8,atol=1e-10)
        assert a.force_time==b.force_time==pytest.approx((a.step-1)*reference.flow.dt)
        for info in (ia,ib):
            assert info['power_error']/max(abs(info['solid_power']),abs(info['fluid_power']),1.) < 1e-9
            assert len(info['flow']['pressure']['stages'])==3
