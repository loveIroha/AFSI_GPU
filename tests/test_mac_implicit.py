"""Independent H-O tangent, reduced Jacobian and implicit coupled-equation checks."""
from dataclasses import replace, asdict
import json
import numpy as np
import torch
import pytest
from test_real_lv import real_case, DEVICES
from afsi_torch.real_lv import imported_model
from afsi_torch.ho_tangent import HOTangentAssembler
from afsi_torch.mac.implicit import MACCouplingOptions, implicit_policy
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.grid import convection, velocity_laplacian, zero_normal, gradient, divergence
from afsi_torch.nonlinear import NewtonOptions, GMRESOptions, NonlinearFailure
from afsi_torch.real_lv_checkpoint import load_real_lv, save_real_lv, refine_checkpoint_dt
from afsi_torch.mac.checkpoint import digest


def settings(config):
    return dict(dt=config.time.dt, fluid_shape=config.fluid.shape,fluid_lengths=config.fluid.lengths,
                fluid_origin=config.fluid.origin,rho=config.fluid.rho,mu=config.fluid.mu,
                interaction_degree=config.interaction_degree,mass_solver=asdict(config.mass_solver),
                pressure_solver=asdict(config.pressure_solver),coupling=asdict(config.coupling),
                **asdict(config.execution))


@pytest.mark.parametrize('device', DEVICES)
def test_assembled_ho_tangent_matches_complete_force_jvp_and_finite_difference(real_case, device):
    model = imported_model(real_case, device)
    X = model.mesh.X
    center = X.new_tensor([7.5,7.5,7.5])
    A = X.new_tensor([[1.025,.009,0.],[.008,1.015,.002],[.001,0.,.99]])
    x = (X-center)@A.T+center
    direction = .01*torch.sin(1.23*X)
    assembler = HOTangentAssembler(model, chunk_size=37)
    K = assembler.assemble(x,.6)
    assert K.layout == torch.sparse_csr
    actual = torch.sparse.mm(K,direction.reshape(-1,1)).reshape_as(x)
    expected = torch.func.jvp(lambda y:model.force(y,.6),(x,),(direction,))[1]
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-7)
    finite = (model.force(x+1e-5*direction,.6)-model.force(x-1e-5*direction,.6))/2e-5
    torch.testing.assert_close(actual,finite,rtol=2e-5,atol=2e-3)
    # An open follower-pressure surface contributes a nonsymmetric tangent.
    dense = K.to_dense()
    assert (dense-dense.T).abs().max() > 1.
    if device == 'cpu':
        # Test tracing of the vmap/jacfwd element and face kernels as well.
        volume = torch.compile(assembler._volume,backend='eager',fullgraph=True)
        expected = assembler._volume(model.element_gradient(x),model.mesh.fiber,model.mesh.sheet,
                                      model.gradients,model.volumes,x.new_tensor(123.))
        torch.testing.assert_close(volume(model.element_gradient(x),model.mesh.fiber,model.mesh.sheet,
                                         model.gradients,model.volumes,x.new_tensor(123.)),expected)
        face = torch.compile(assembler._follower,backend='eager',fullgraph=True)
        torch.testing.assert_close(face(x[model.endo.faces],x.new_tensor(500.)),
                                   assembler._follower(x[model.endo.faces],x.new_tensor(500.)))


@pytest.mark.parametrize('device', DEVICES)
def test_reduced_newton_jacobian_and_new_time_momentum_kinematics_power(real_case, device, monkeypatch):
    import afsi_torch.mac.implicit as module
    config = replace(real_case,coupling=MACCouplingOptions(scheme='implicit-newton'))
    model = imported_model(config, device)
    driver = build_driver(model,settings(config),device)
    initial = driver.initialize(model.mesh.X)
    center = initial.x.new_tensor([7.5,7.5,7.5])
    x = center+(initial.x-center)*1.005
    state = replace(initial,step=6000,time=.6,x=x,force_time=.6,force=model.force(x,.6))
    raw = tuple(.1*torch.sin(1.7*driver.flow.grid.coordinates(c,device=device)[...,c])
                for c in range(3))
    state = replace(state,velocity=driver.flow.project(zero_normal(raw)).velocity)
    if device == 'cpu':
        kernel = torch.compile(driver._linear_right,backend='eager',fullgraph=True)
        v = zero_normal(tuple(torch.cos(u) for u in state.velocity))
        zeros = driver.flow.grid.zeros()
        actual = kernel(state.velocity,v,zeros)
        for left,right in zip(actual,driver._linear_right(state.velocity,v,zeros)):
            torch.testing.assert_close(left,right)
    original = module.newton
    comparisons = []
    def checked(residual,x0,**kwargs):
        direction = driver.pack(zero_normal(tuple(torch.cos(u) for u in state.velocity)))
        action = kwargs['linearization_factory'](x0)
        actual = action(direction)
        finite = (residual(x0+1e-5*direction)-residual(x0-1e-5*direction))/2e-5
        torch.testing.assert_close(actual,finite,atol=2e-7,rtol=2e-6)
        comparisons.append(True)
        return original(residual,x0,**kwargs)
    monkeypatch.setattr(module,'newton',checked)
    new,info = driver.step(state)
    assert comparisons and info['nonlinear']['iterations'] > 0
    assert info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']
    assert new.force_time == new.time == pytest.approx(.6001)
    torch.testing.assert_close(new.force,model.force(new.x,new.time),rtol=1e-10,atol=1e-7)
    stencil = driver.transfer.prepare(state.x)
    U,_ = driver.transfer.interpolate(new.velocity,stencil)
    torch.testing.assert_close(new.x,state.x+driver.flow.dt*U,rtol=1e-12,atol=1e-12)
    density,_ = driver.transfer.spread(new.force,stencil)
    # Independent transcription of the new-time momentum equation.
    advection = convection(new.velocity,driver.flow.grid.spacing)
    grad = gradient(new.pressure,driver.flow.grid.spacing)
    residual = tuple(u-old+driver.flow.dt*(a-driver.flow.mu/driver.flow.rho*
        velocity_laplacian(u,c,driver.flow.grid.spacing)+g/driver.flow.rho-f/driver.flow.rho)
        for c,(u,old,a,g,f) in enumerate(zip(new.velocity,state.velocity,advection,grad,density)))
    assert torch.linalg.vector_norm(driver.pack(residual)).item() <= 2*info['nonlinear']['tolerance']
    assert info['power_error']/max(abs(info['solid_power']),abs(info['fluid_power']),1.) < 1e-10
    assert torch.linalg.vector_norm(divergence(new.velocity,driver.flow.grid.spacing)) < 1e-8


@pytest.mark.parametrize('device', DEVICES)
def test_implicit_transport_advances_above_old_explicit_A_limit(real_case, device):
    config = replace(real_case,coupling=MACCouplingOptions(scheme='implicit-newton'))
    model = imported_model(config,device)
    driver = build_driver(model,settings(config),device)
    state = driver.initialize(model.mesh.X)
    coords = [driver.flow.grid.coordinates(c,device=device) for c in range(3)]
    velocity = zero_normal(tuple(torch.sin(2*torch.pi*q[...,(c+1)%3]/15)*torch.sin(torch.pi*q[...,c]/15)
                                 for c,q in enumerate(coords)))
    velocity = driver.flow.project(velocity).velocity
    # Scale a divergence-free initial flow beyond the old A guard.
    scale = 100./max(u.abs().max().item() for u in velocity)
    velocity = tuple(u*scale for u in velocity)
    state = replace(state,velocity=velocity)
    new,info = driver.step(state)
    assert new.step == 1 and info['nonlinear']['residual_norm'] <= info['nonlinear']['tolerance']
    assert info['flow']['advection_diffusion_number'] > .25
    assert model.diagnostics(new.x)['minimum_detF'] > 0


def test_switch_to_implicit_preserves_source_restart_and_force_clock(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    config = replace(real_case,output=replace(real_case.output,write_vtk=False))
    source = tmp_path/'explicit'
    run(case_config=config,device='cpu',output=source)
    path = source/'checkpoint.npz'
    # Recreate a real legacy checkpoint without the newly introduced fields.
    with np.load(path,allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in archive.files if k != 'metadata'}
        metadata = json.loads(str(archive['metadata']))
    metadata['config'].pop('coupling')
    metadata['settings'].pop('coupling')
    metadata.pop('sha256')
    metadata['sha256'] = digest(metadata,arrays)
    np.savez_compressed(path,metadata=json.dumps(metadata),**arrays)
    before = path.read_bytes()
    target = tmp_path/'implicit'
    report = run(device='cpu',resume=path,output=target,coupling_scheme='implicit-newton',end_time=3e-4)
    assert report['completed'] and report['transport_policy'] == implicit_policy()
    assert report['restart_from']['old_scheme'] == 'explicit-lagged'
    assert report['restart_from']['new_scheme'] == 'implicit-newton'
    assert report['last']['force_time_s'] == report['last']['time_s']
    assert path.read_bytes() == before
    model,state,solver,progress,cfg = load_real_lv(target/'checkpoint.npz')
    assert cfg.coupling.scheme == solver['coupling']['scheme'] == 'implicit-newton'
    from validation.diagnose_real_lv_guard import diagnose
    assert diagnose(target/'checkpoint.npz')['transport_policy'] == implicit_policy()
    assert not diagnose(target/'checkpoint.npz')['triggered']
    assert state.force_time == state.time
    _,_,smaller,_ = refine_checkpoint_dt(model,state,solver,cfg,5e-5,4e-4)
    assert smaller.coupling.scheme == 'implicit-newton'
    assert refine_checkpoint_dt(model,state,solver,cfg,5e-5,4e-4)[0].force_time == state.time
    report = run(device='cpu',resume=target/'checkpoint.npz',end_time=4e-4)
    assert report['completed'] and report['last']['newton_iterations'] > 0
    with pytest.raises(ValueError,match='new output directory'):
        run(device='cpu',resume=path,coupling_scheme='implicit-newton')


def test_implicit_config_cli_and_legacy_force_clock(real_case,tmp_path,monkeypatch):
    from demo.real_lv_fsi import run_mac
    from afsi_torch.config import save_config,load_config
    calls = []
    monkeypatch.setattr(run_mac,'run',lambda **kwargs:calls.append(kwargs))
    run_mac.main(['--resume','old/checkpoint.npz','--coupling','implicit-newton',
                  '--output',str(tmp_path/'new'),'--cycles','3'])
    assert calls[0]['coupling_scheme'] == 'implicit-newton' and calls[0]['resume_dt'] is None
    config = replace(real_case,coupling=MACCouplingOptions(scheme='implicit-newton'))
    path = tmp_path/'implicit.json'
    save_config(path,config)
    assert load_config(path,config) == config
    run_mac.main(['--config',str(path),'--coupling','explicit-lagged',
                  '--write-config',str(tmp_path/'explicit.json')])
    assert load_config(tmp_path/'explicit.json',config).coupling.scheme == 'explicit-lagged'


def test_failed_newton_retains_last_accepted_checkpoint(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    # Force a genuine one-iteration Newton limit with strict tolerances.
    opt = MACCouplingOptions(scheme='implicit-newton',newton=NewtonOptions(
        rtol=0.,atol=1e-20,max_iterations=1,linear=GMRESOptions(rtol=.01,atol=1e-18)))
    config = replace(real_case,coupling=opt,output=replace(real_case.output,write_vtk=False))
    target = tmp_path/'failed'
    with pytest.raises((NonlinearFailure,RuntimeError)):
        run(case_config=config,device='cpu',output=target)
    report = json.loads((target/'report.json').read_text())
    assert report['status'] == 'failed' and report['accepted_steps'] == 0
    _,state,_,_,_ = load_real_lv(target/'checkpoint.npz')
    assert state.step == 0 and state.time == 0.
    assert report['failure']['nonlinear']['residual_norm'] > report['failure']['nonlinear']['tolerance']


@pytest.mark.parametrize('device', DEVICES)
def test_reference_and_fused_implicit_paths_match_during_contraction(real_case,device):
    from afsi_torch.real_lv import RealLVConfig
    config = replace(real_case,coupling=MACCouplingOptions(scheme='implicit-newton'))
    model = imported_model(config,device)
    reference = build_driver(model,settings(config),device)
    execution = RealLVConfig().execution
    if device == 'cpu':
        execution = replace(execution,pressure_backend='workspace')
    optimized = build_driver(model,settings(replace(config,execution=execution)),device)
    a,b = reference.initialize(model.mesh.X),optimized.initialize(model.mesh.X)
    force = model.force(a.x,.6)
    a = replace(a,step=6000,time=.6,force=force,force_time=.6)
    b = replace(b,step=6000,time=.6,force=force.clone(),force_time=.6)
    for _ in range(2):
        a,ia = reference.step(a)
        b,ib = optimized.step(b)
        torch.testing.assert_close(a.x,b.x,rtol=1e-10,atol=1e-11)
        torch.testing.assert_close(a.pressure,b.pressure,rtol=1e-7,atol=1e-6)
        torch.testing.assert_close(a.force,b.force,rtol=1e-8,atol=2e-6)
        for u,v in zip(a.velocity,b.velocity):
            torch.testing.assert_close(u,v,rtol=1e-7,atol=1e-9)
        assert ia['nonlinear']['residual_norm'] <= ia['nonlinear']['tolerance']
        assert ib['nonlinear']['residual_norm'] <= ib['nonlinear']['tolerance']
