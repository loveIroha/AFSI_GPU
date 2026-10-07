"""Adaptive Gaussian IB quadrature, unchanged UFL force, paired FE/MAC work."""
from dataclasses import asdict, replace
from math import factorial
import json
import numpy as np
import pytest
import torch
from test_real_lv import DEVICES, real_case, ufl_energy
from test_mac_implicit import settings
from afsi_torch.p1 import prepare_p1
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.adaptive_transfer import (
    AdaptiveP1Transfer, InteractionQuadratureOptions, gaussian_tetra_rule, interaction_quadrature_plan)
from afsi_torch.real_lv import imported_model
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.semiimplicit import MidpointProblem
from afsi_torch.real_lv_checkpoint import load_real_lv


@pytest.mark.parametrize('order',[2,3,4])
def test_positive_gaussian_rule_integrates_degree_and_consistent_mass(order):
    q,w = gaussian_tetra_rule(order,torch.zeros((),dtype=torch.float64))
    assert (w>0).all() and (q>=0).all() and (q.sum(-1)<=1+1e-15).all()
    degree = 2*order-1
    for a in range(degree+1):
        for b in range(degree+1-a):
            for c in range(degree+1-a-b):
                expected = factorial(a)*factorial(b)*factorial(c)/factorial(a+b+c+3)
                actual = (w*q[:,0]**a*q[:,1]**b*q[:,2]**c).sum().item()
                assert actual==pytest.approx(expected,rel=2e-13,abs=2e-16)
    N = torch.cat((1-q.sum(-1,keepdim=True),q),-1)
    expected = (torch.ones((4,4),dtype=q.dtype)+torch.eye(4,dtype=q.dtype))/120
    torch.testing.assert_close(N.T@(w[:,None]*N),expected,atol=2e-15,rtol=2e-13)


def transfer(device,**options):
    # Two disjoint affine cells requiring distinct Gaussian orders (2,3).
    X = torch.tensor([[3,3,3],[3.3,3,3],[3,3.3,3],[3,3,3.3],
                      [6,6,6],[7,6,6],[6,7,6],[6,6,7]],dtype=torch.float64,device=device)
    cells = torch.arange(8,device=device).reshape(2,4)
    geometry = prepare_p1(X,cells,degree=2)
    return X,AdaptiveP1Transfer(MACGrid((16,)*3,(16.,)*3),geometry,
        quadrature_options=InteractionQuadratureOptions(mode='adaptive',**options))


@pytest.mark.parametrize('device',DEVICES)
def test_adaptive_pair_preserves_force_torque_power_affine_field_and_reference_mass(device):
    X,t = transfer(device)
    x = X+X.new_tensor([.1,.2,-.1])
    stencil = t.prepare(x)
    assert stencil.rule.orders.tolist()==[2,3] and stencil.rule.point_count==23
    assert sum(g.weights.sum().item() for g in stencil.rule.groups)==pytest.approx(t.geometry.weights.sum().item())
    old_mass = t.mass.values().clone()
    A = X.new_tensor([[.07,.02,0.],[0.,-.03,.01],[.01,0.,.04]])
    b = X.new_tensor([.3,-.2,.5])
    field = tuple((t.grid.coordinates(c,device=device)@A.T+b)[...,c] for c in range(3))
    U,_ = t.interpolate(field,stencil)
    torch.testing.assert_close(U,x@A.T+b,rtol=2e-11,atol=2e-12)
    force = torch.sin(1.7*X)
    density,_ = t.spread(force,stencil)
    resultant = t.grid.volume*torch.stack([v.sum() for v in density])
    torch.testing.assert_close(resultant,force.sum(0),rtol=2e-11,atol=2e-12)
    torque = torch.zeros(3,dtype=X.dtype,device=device)
    for c,v in enumerate(density):
        f = torch.zeros((*v.shape,3),dtype=X.dtype,device=device)
        f[...,c] = v
        torque += t.grid.volume*torch.linalg.cross(t.grid.coordinates(c,device=device),f).sum((0,1,2))
    torch.testing.assert_close(torque,torch.linalg.cross(x,force).sum(0),rtol=2e-11,atol=2e-11)
    solid_power = (force*U).sum()
    fluid_power = t.grid.volume*sum((u*f).sum() for u,f in zip(field,density))
    torch.testing.assert_close(solid_power,fluid_power,rtol=2e-11,atol=2e-12)
    # A future stencil can use a denser rule; the old pair must remain intact.
    stretched = x.clone(); stretched[5:,0] += X.new_tensor([.6,0.,0.])
    other = t.prepare(stretched)
    assert other.rule.orders.tolist()==[2,4]
    torch.testing.assert_close(t.mass.values(),old_mass,atol=0,rtol=0)
    again,_ = t.interpolate(field,stencil)
    torch.testing.assert_close(again,U,rtol=2e-11,atol=2e-12)
    repeated,_ = t.spread(force,stencil)
    for a,b in zip(repeated,density):
        torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-12)


def test_quadrature_budgets_fail_without_silent_density_clipping():
    X,t = transfer('cpu',max_order=2)
    with pytest.raises(ValueError,match='order 3.*no density clipping'):
        t.prepare(X)
    X,t = transfer('cpu',max_points=20)
    with pytest.raises(ValueError,match='23 points.*no density clipping'):
        t.prepare(X)
    for kw in ({'point_density':1.},{'max_order':1},{'max_points':7},{'mode':'bad'}):
        with pytest.raises(ValueError):
            InteractionQuadratureOptions(**kw)


def test_quadrature_inspection_counts_and_limits_without_allocating_stencils(real_case):
    from validation.inspect_real_lv_interaction import inspect
    X,t = transfer('cpu',max_points=20)
    plan = interaction_quadrature_plan(X,t.geometry.cells,t.grid,t.quadrature_options)
    assert plan['point_count']==23 and plan['gaussian_order_cell_counts']=={2:1,3:1}
    assert not plan['within_budget']
    report = inspect(real_case)
    assert report['material_unchanged'] and not report['simulation_started']
    assert report['within_budget']
    assert report['point_count']==8*report['mesh_cells']


@pytest.mark.parametrize('device',DEVICES)
def test_adaptive_midpoint_retains_supplied_ufl_and_csr_tangent(real_case,device):
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-semiimplicit'),
                  interaction_quadrature=InteractionQuadratureOptions(mode='adaptive'))
    model = imported_model(cfg,device)
    F = model.mesh.X.new_tensor([[1.02,.01,0.],[0.,.99,.01],[0.,0.,1.01]]).requires_grad_()
    f,s = model.mesh.fiber[0],model.mesh.sheet[0]
    T = 12345.
    expected = torch.autograd.grad(ufl_energy(F,f,s,model.parameters),F)[0]
    Ff = F@f
    expected += T*(1+4.9*(torch.linalg.vector_norm(Ff)-1))*torch.outer(Ff,f)
    from afsi_torch.holzapfel_ogden import ho_pk1
    torch.testing.assert_close(ho_pk1(F,f,s,model.parameters,T),expected,rtol=3e-12,atol=1e-8)
    driver = build_driver(model,settings(cfg),device)
    state = driver.initialize(model.mesh.X)
    predicted = state.x+1e-4*torch.sin(state.x)
    stencil = driver.transfer.prepare(predicted)
    problem = MidpointProblem(driver,state,predicted,stencil,driver.flow.advection(state.velocity))
    y,v = torch.zeros_like(predicted),.01*torch.sin(1.7*predicted)
    action = problem.linearization(y)
    finite = (problem.residual(y+1e-4*v)-problem.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(action(v),finite,rtol=2e-5,atol=2e-8)
    assert problem.tangent.layout==torch.sparse_csr
    _,info = driver.step(state)
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    assert info['power_error']/max(abs(info['solid_power']),abs(info['fluid_power']),1.)<1e-9


def test_adaptive_config_and_checkpoint_restart_keep_user_material(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    cfg = replace(real_case,coupling=MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton'),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive'),
        output=replace(real_case.output,write_vtk=False))
    whole,split = tmp_path/'whole',tmp_path/'split'
    run(case_config=replace(cfg,time=replace(cfg.time,end_time=4e-4)),device='cpu',output=whole)
    run(case_config=cfg,device='cpu',output=split)
    report = run(resume=split/'checkpoint.npz',device='cpu',end_time=4e-4)
    _,a,_,_,ca = load_real_lv(whole/'checkpoint.npz')
    _,b,opts,_,cb = load_real_lv(split/'checkpoint.npz')
    torch.testing.assert_close(a.x,b.x,atol=2e-13,rtol=1e-12)
    for u,v in zip(a.velocity,b.velocity):
        torch.testing.assert_close(u,v,atol=1e-12,rtol=1e-9)
    assert ca.material==cb.material==cfg.material and ca.loads==cb.loads==cfg.loads
    assert opts['interaction_quadrature']==asdict(cfg.interaction_quadrature)
    assert report['material_model']=='user-HO-iso-I1-DG0'
    assert report['interaction_quadrature']['mode']=='adaptive'
    assert report['last']['interaction_points']>=8*len(imported_model(cfg).mesh.cells)
    with np.load(split/'checkpoint.npz',allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
    assert 'stress_form' not in metadata['config']


def test_demo_paper_material_and_quadrature_cli(tmp_path):
    from demo.real_lv_fsi import run_mac
    path = tmp_path/'config.json'
    run_mac.main(['--write-config',str(path)])
    config = json.loads(path.read_text())
    assert config['material']==asdict(run_mac.CONFIG.material)
    assert config['material']['b']==10.81 and config['loads']['target_mmhg']==8.
    assert config['interaction_quadrature']['mode']=='adaptive'
    run_mac.main(['--reference','--interaction-quadrature','fixed','--write-config',str(path)])
    assert json.loads(path.read_text())['interaction_quadrature']['mode']=='fixed'
    with pytest.raises(SystemExit):
        run_mac.main(['--resume','checkpoint.npz','--interaction-quadrature','adaptive'])


def test_old_checkpoint_restores_fixed_quadrature_and_rejects_mismatched_settings(real_case,tmp_path):
    from afsi_torch.real_lv_checkpoint import save_real_lv
    from afsi_torch.mac.checkpoint import digest
    model = imported_model(real_case)
    opts = settings(real_case)
    state = build_driver(model,opts,'cpu').initialize(model.mesh.X)
    path = tmp_path/'legacy.npz'
    save_real_lv(path,model,state,opts,{},real_case)
    with np.load(path,allow_pickle=False) as archive:
        data = {name:archive[name] for name in archive.files}
    metadata = json.loads(str(data.pop('metadata')))
    metadata.pop('sha256')
    metadata['config'].pop('interaction_quadrature')
    metadata['settings'].pop('interaction_quadrature')
    metadata['sha256'] = digest(metadata,data)
    np.savez_compressed(path,metadata=json.dumps(metadata),**data)
    _,_,_,_,restored = load_real_lv(path)
    assert restored.interaction_quadrature.mode=='fixed'
    metadata.pop('sha256')
    metadata['config']['interaction_quadrature'] = {'mode':'adaptive'}
    metadata['config']['coupling']['scheme'] = 'cnab-semiimplicit'
    metadata['settings']['coupling']['scheme'] = 'cnab-semiimplicit'
    metadata['sha256'] = digest(metadata,data)
    np.savez_compressed(path,metadata=json.dumps(metadata),**data)
    with pytest.raises(ValueError,match='interaction quadratures differ'):
        load_real_lv(path)


def test_variable_quadrature_tensor_kernels_trace_without_host_graph_breaks():
    from afsi_torch.mac.compact_transfer import _evaluate, _weighted, _assemble, _prepare
    evaluate = torch.compile(_evaluate,backend='eager',fullgraph=True,dynamic=True)
    weighted = torch.compile(_weighted,backend='eager',fullgraph=True,dynamic=True)
    assemble = torch.compile(_assemble,backend='eager',fullgraph=True,dynamic=True)
    prepare = torch.compile(_prepare,backend='eager',fullgraph=True,dynamic=True)
    X,t = transfer('cpu')
    for n in (2,3,4):
        q,w = gaussian_tetra_rule(n,X)
        N = torch.cat((1-q.sum(-1,keepdim=True),q),-1)
        for count in (1,2):
            cells = t.geometry.cells[:count]
            W = t.reference_determinants[:count,None]*w
            points = evaluate(X,N,cells).reshape(-1,3)
            torch.testing.assert_close(points,_evaluate(X,N,cells).reshape(-1,3))
            torch.testing.assert_close(weighted(X,N,cells,W),_weighted(X,N,cells,W))
            torch.testing.assert_close(assemble(points,N,cells,W,t.diagonal),_assemble(points,N,cells,W,t.diagonal))
            base,phi = prepare(points,t.origins,t.spacing,t.axis_offsets)
            expected_base,expected_phi = _prepare(points,t.origins,t.spacing,t.axis_offsets)
            torch.testing.assert_close(base,expected_base)
            torch.testing.assert_close(phi,expected_phi)
