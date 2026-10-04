"""Identical FE/IB quadrature under fused and cell-reduced execution."""
from dataclasses import replace
import os
import pytest
import torch
from test_adaptive_p1_transfer import transfer
from test_real_lv import real_case, DEVICES
from test_mac_implicit import settings
from test_mac_compact_quadrature import adaptive_config
from afsi_torch.mac.adaptive_cell import support_extent
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.semiimplicit import MidpointProblem
from afsi_torch.real_lv import imported_model


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
@pytest.mark.parametrize('backend',['fused','cell'])
def test_transfer_same_fields_work_torque_and_old_stencil(device,family,backend):
    if family=='xiao-gimbutas':
        pytest.importorskip('basix')
    X,reference = transfer(device,rule_family=family)
    _,candidate = transfer(device,rule_family=family,transfer_backend=backend)
    x = X+X.new_tensor([.1,.2,-.1])
    old = reference.prepare(x)
    new = candidate.prepare(x)
    # Change current rule; the earlier pair must own its cell membership/order.
    other = x.clone(); other[5,0] += .6
    candidate.prepare(other)
    for c in range(3):
        offset = 0
        for group in new.rule.groups:
            count = group.weights.numel()
            base = new.base[c,offset:offset+count].reshape(len(group.cells),-1,3)
            assert ((base.amax(1)-base.amin(1)+4)<=support_extent(group,2.)).all()
            offset += count
    velocity = tuple(torch.sin(1.3*reference.grid.coordinates(c,device=device)[...,c])+.2 for c in range(3))
    force = torch.sin(1.7*X)
    fa,_ = reference.spread(force,old)
    fb,_ = candidate.spread(force,new)
    ua,_ = reference.interpolate(velocity,old)
    ub,_ = candidate.interpolate(velocity,new)
    torch.testing.assert_close(ub,ua,rtol=3e-11,atol=3e-12)
    for a,b in zip(fa,fb):
        torch.testing.assert_close(b,a,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close((force*ub).sum(),candidate.grid.volume*sum((u*f).sum() for u,f in zip(velocity,fb)),rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(candidate.grid.volume*torch.stack([f.sum() for f in fb]),force.sum(0),rtol=3e-11,atol=3e-12)
    torque = torch.zeros(3,device=device,dtype=X.dtype)
    for c,f in enumerate(fb):
        field = torch.zeros((*f.shape,3),device=device,dtype=X.dtype)
        field[...,c] = f
        torque += candidate.grid.volume*torch.linalg.cross(candidate.grid.coordinates(c,device=device),field).sum((0,1,2))
    torch.testing.assert_close(torque,torch.linalg.cross(x,force).sum(0),rtol=3e-11,atol=3e-11)
    torch.testing.assert_close(candidate.mass.values(),reference.mass.values(),rtol=0,atol=0)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('backend',['fused','cell'])
def test_coupled_active_jacobian_and_accepted_step(real_case,device,backend):
    pytest.importorskip('basix')
    cfg = adaptive_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,rule_family='xiao-gimbutas'))
    opt = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend=backend))
    model = imported_model(cfg,device)
    reference = build_driver(model,settings(cfg),device)
    candidate = build_driver(model,settings(opt),device)
    state,_ = reference.step(reference.initialize(model.mesh.X))
    a,ia = reference.step(state)
    b,ib = candidate.step(state)
    assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
    torch.testing.assert_close(b.x,a.x,rtol=2e-11,atol=2e-12)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(v,u,rtol=2e-7,atol=2e-10)
    # Independent finite difference with the same frozen paired stencil.
    active = replace(state,step=6000,time=.6,force=model.force(state.x,.6),force_time=.6)
    stencil = candidate.transfer.prepare(state.x)
    problem = MidpointProblem(candidate,active,state.x,stencil,candidate.flow.advection(state.velocity))
    y,v = torch.zeros_like(state.x),.01*torch.sin(state.x)
    action = problem.linearization(y)
    finite = (problem.residual(y+1e-5*v)-problem.residual(y-1e-5*v))/2e-5
    torch.testing.assert_close(action(v),finite,rtol=3e-5,atol=3e-8)


def test_benchmark_compares_execution_against_same_compact_quadrature(real_case,tmp_path):
    pytest.importorskip('basix')
    from afsi_torch.simulation.real_lv_mac import run
    from validation.benchmark_real_lv_schemes import benchmark
    folder = tmp_path/'source'
    run(case_config=adaptive_config(real_case),device='cpu',output=folder)
    path = folder/'checkpoint.npz'
    original = path.read_bytes()
    report = benchmark(path,device='cpu',schemes=('cnab-semiimplicit',),
        nonlinear_solvers=('anderson-newton',),execution_variants=('compact','fused','cell'),warmup=0,steps=2,profile=True)
    prefix = 'cnab-semiimplicit/anderson-newton/'
    for variant in ('compact','fused','cell'):
        case = report['cases'][prefix+variant]
        assert case['completed']
        assert case['interaction_quadrature']['rule_family']=='xiao-gimbutas'
        assert case['counts']==report['cases'][prefix+'compact']['counts']
        assert 'profile_failure' not in case
        assert case['phases']['mass_solves']['calls']==case['profile_counts']['mass_solves']
    for variant in ('fused','cell'):
        comparison = report['execution_comparisons'][prefix+variant]
        assert comparison['reference']==prefix+'compact'
        assert comparison['final_state_differences']['x_max_abs']<2e-12
        assert comparison['relative_l2']['displacement']<1e-6
    assert path.read_bytes()==original


@pytest.mark.skipif(os.environ.get('TRITON_INTERPRET')!='1',reason='optional Triton CPU interpreter')
@pytest.mark.parametrize('family',['conical','xiao-gimbutas'])
@pytest.mark.parametrize('stretch',[.6,2.6])
def test_actual_triton_kernel_math_with_cpu_interpreter(family,stretch):
    triton = pytest.importorskip('triton')
    if family=='xiao-gimbutas':
        pytest.importorskip('basix')
    from afsi_torch.mac._triton_adaptive import _spread_points,_spread_cells,_gather_assemble
    X,t = transfer('cpu',rule_family=family)
    x = X.clone(); x[5,0] += stretch
    stencil = t.prepare(x)
    coefficient = torch.sin(X)
    velocity = tuple(torch.cos(t.grid.coordinates(c)[...,c]) for c in range(3))
    reference,_ = t.spread(t.mass_action(coefficient),stencil)
    expected = torch.zeros_like(t.diagonal)
    gathered = t.gather_grid(velocity,stencil)
    offset = 0
    point_fields,cell_fields = [t.grid.zeros(dtype=X.dtype) for _ in range(2)]
    rhs = torch.zeros_like(t.diagonal)
    for group in stencil.rule.groups:
        ne,nq = group.weights.shape
        count = ne*nq
        extent = support_extent(group,2.)
        q = triton.next_power_of_2(nq)
        expected += t._assemble_kernel(gathered[offset:offset+count],group.values,group.cells,group.weights,t.diagonal)
        for c,u in enumerate(velocity):
            args = (stencil.base[c,offset:offset+count],stencil.phi[c,offset:offset+count],
                group.cells,group.values,group.weights,coefficient)
            _spread_points[(triton.cdiv(count,4),)](*args,point_fields[c],ne,nq,*t.grid.face_shape(c)[1:],c,t.grid.volume,4)
            _spread_cells[(ne,triton.cdiv(extent**3,32))](*args,cell_fields[c],nq,q,*t.grid.face_shape(c)[1:],c,t.grid.volume,extent,32)
            tile = min(32,q)
            _gather_assemble[(ne,triton.cdiv(nq,tile))](*args[:5],u,rhs,nq,tile,*t.grid.face_shape(c)[1:],c)
        offset += count
    for a,b,c in zip(reference,point_fields,cell_fields):
        torch.testing.assert_close(b,a,rtol=3e-11,atol=3e-12)
        torch.testing.assert_close(c,a,rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(rhs,expected,rtol=3e-11,atol=3e-12)


def test_cell_execution_checkpoint_and_resume_preserve_saved_backend(real_case,tmp_path):
    pytest.importorskip('basix')
    from afsi_torch.simulation.real_lv_mac import run
    from afsi_torch.real_lv_checkpoint import load_real_lv
    cfg = adaptive_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,
        rule_family='xiao-gimbutas',transfer_backend='cell'))
    folder = tmp_path/'cell'
    run(case_config=cfg,device='cpu',output=folder)
    _,_,opts,_,restored = load_real_lv(folder/'checkpoint.npz')
    assert restored.interaction_quadrature==cfg.interaction_quadrature
    assert opts['interaction_quadrature']['transfer_backend']=='cell'
    report = run(resume=folder/'checkpoint.npz',device='cpu',end_time=3e-4)
    assert report['completed'] and report['interaction_quadrature']['transfer_backend']=='cell'


def test_direct_benchmark_cli_profiles_without_repository_on_python_path(real_case,tmp_path):
    """The user's script entry point must also work outside a pytest import path."""
    import json
    from pathlib import Path
    import subprocess
    import sys
    pytest.importorskip('basix')
    from afsi_torch.simulation.real_lv_mac import run
    folder = tmp_path/'source'
    run(case_config=adaptive_config(real_case),device='cpu',output=folder)
    checkpoint = folder/'checkpoint.npz'
    original = checkpoint.read_bytes()
    output = tmp_path/'profile.json'
    script = Path(__file__).resolve().parents[1]/'validation'/'benchmark_real_lv_schemes.py'
    result = subprocess.run([sys.executable,'-I',str(script),
        '--checkpoint',str(checkpoint),'--schemes','cnab-semiimplicit',
        '--nonlinear-solvers','anderson-newton','--execution-variants','compact',
        '--device','cpu','--warmup','0','--steps','1','--profile','--output',str(output)],
        cwd=tmp_path,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    case = json.loads(output.read_text())['cases']['cnab-semiimplicit/anderson-newton/compact']
    assert case['completed'] and 'profile_failure' not in case
    assert case['phases']['mass_solves']['calls']>0
    assert checkpoint.read_bytes()==original
