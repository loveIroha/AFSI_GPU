"""Reduce coupled work while retaining the original equations/acceptance."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from afsi_torch.mac.midpoint_solver import AndersonOptions,anderson_policy,accelerated_midpoint
from afsi_torch.nonlinear import NewtonOptions,GMRESOptions,newton,gmres,NonlinearFailure,NewtonResult
from afsi_torch.mac.solid_preconditioner import SolidBlockPreconditioner
from afsi_torch.mac.paper_coupling import BEIBStepper
from afsi_torch.paper_lv import imported_model
from afsi_torch.paper_lv_checkpoint import save,load
from test_real_lv import real_case,DEVICES
from test_paper_lv import config_for


class LinearProblem:
    def __init__(self,values):
        self.values=torch.as_tensor(values,dtype=torch.float64)
        self.target=torch.full_like(self.values,.01)
        self.residual_calls=self.action_calls=0
    def validate(self,x):
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite trial')
    def residual(self,x):
        self.residual_calls+=1
        return self.values*(x-self.target)
    def linearization(self,x):
        def action(v):
            self.action_calls+=1
            return self.values*v
        return action
    def preconditioner(self,x):
        return None


OPTIONS=NewtonOptions(rtol=1e-6,atol=1e-9,linear=GMRESOptions(
    rtol=1e-3,atol=1e-12,check_every=5,max_iterations=100))


def test_trend_extension_avoids_newton_and_reduces_actual_operator_evaluations():
    results=[]
    for policy in ('legacy','adaptive'):
        problem=LinearProblem(torch.linspace(.8,1.4,8))
        result,info=accelerated_midpoint(problem,torch.zeros_like(problem.target),OPTIONS,
            anderson_policy(AndersonOptions(),policy),newton_solve=newton)
        assert result.converged and result.residual_norm<=result.tolerance
        assert result.tolerance==max(OPTIONS.atol,OPTIONS.rtol*torch.linalg.vector_norm(problem.values*problem.target).item())
        results.append((problem,result,info))
    legacy,adaptive=results
    assert legacy[2]['newton_fallback'] and not adaptive[2]['newton_fallback']
    assert adaptive[2]['anderson_extra_iterations']>0
    assert adaptive[0].residual_calls+adaptive[0].action_calls < legacy[0].residual_calls+legacy[0].action_calls
    torch.testing.assert_close(adaptive[1].x,adaptive[0].target,atol=5e-8,rtol=0)


def test_fallback_reuses_only_the_exact_best_residual_without_loosening_target():
    results=[]
    for use_seed in (False,True):
        problem=LinearProblem(torch.full((6,),100.))
        solver=newton if use_seed else lambda r,x,**kw:newton(problem.residual,x,**kw)
        result,info=accelerated_midpoint(problem,torch.zeros_like(problem.target),OPTIONS,
            AndersonOptions(),newton_solve=solver)
        results.append((problem,result,info))
    old,new=results
    assert new[0].residual_calls==old[0].residual_calls-1
    assert new[2]['seeded_residual_reuses']==1 and old[2]['seeded_residual_reuses']==0
    assert new[1].tolerance==old[1].tolerance
    torch.testing.assert_close(new[1].x,old[1].x,atol=0,rtol=0)


def test_stagnant_residual_has_bounded_trials_and_never_claims_convergence():
    problem=LinearProblem(torch.ones(6))
    problem.residual=lambda x:torch.ones_like(x)
    calls=[]
    def fail(r,x,**kw):
        calls.append(kw['options'])
        raise NonlinearFailure('no root',NewtonResult(x,False,6**.5,kw['options'].atol,0,[]))
    with pytest.raises(NonlinearFailure) as failure:
        accelerated_midpoint(problem,torch.zeros_like(problem.target),OPTIONS,
            anderson_policy(AndersonOptions(), 'adaptive'),newton_solve=fail)
    assert len(calls)==1 and not failure.value.result.converged
    assert failure.value.result.iterations==4
    assert calls[0].rtol==0 and calls[0].atol==pytest.approx(OPTIONS.rtol*6**.5)


@pytest.mark.parametrize('kwargs',[{'extra_iterations':True},{'extra_iterations':-1},
    {'stall_iterations':-1},{'extension_window':0},{'extension_ratio':1.},{'stall_ratio':float('nan')}])
def test_bad_policy_controls_rejected(kwargs):
    with pytest.raises(ValueError):
        AndersonOptions(**kwargs)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_local_block_preconditioner_reduces_krylov_work_and_retains_true_residual(device,dtype):
    # Independent local fluid-inertia model, with four anisotropic node blocks.
    stiffness=torch.diag(torch.logspace(1,4,12,device=device,dtype=dtype))
    nodes=torch.zeros((4,3),device=device,dtype=dtype)
    model=SimpleNamespace(mesh=SimpleNamespace(X=nodes,cells=torch.arange(4,device=device)[None]),
        volumes=nodes.new_tensor([4.]))
    assembler=SimpleNamespace(model=model,
        nodal_diagonal_blocks=lambda K:torch.stack([K.to_dense()[3*i:3*i+3,3*i:3*i+3] for i in range(4)]))
    preconditioner=SolidBlockPreconditioner(assembler,dt=.1,rho=1.)
    inverse=preconditioner.build((-stiffness).to_sparse_csr())
    matrix=torch.eye(12,device=device,dtype=dtype)+.01*stiffness
    action=lambda v:(matrix@v.reshape(-1)).reshape_as(v)
    rhs=torch.ones_like(nodes)
    options=GMRESOptions(rtol=2e-5 if dtype==torch.float32 else 1e-9,
        atol=1e-8,restart=12,max_iterations=24,check_every=1)
    a,plain=gmres(action,rhs,options=options)
    b,blocked=gmres(action,rhs,precondition=inverse,options=options)
    assert blocked['iterations']<plain['iterations']
    assert torch.linalg.vector_norm(action(b)-rhs)<=blocked['tolerance']
    torch.testing.assert_close(b,a,atol=2e-5 if dtype==torch.float32 else 1e-9,rtol=2e-5)
    # Destabilizing approximate stiffness cannot create a singular inverse.
    safe=preconditioner.build((1000*stiffness).to_sparse_csr())
    torch.testing.assert_close(safe(rhs),rhs,atol=2e-6,rtol=2e-6)


@pytest.mark.parametrize('device',DEVICES)
def test_actual_be_newton_block_preconditioning_keeps_endpoint_equations(real_case,device):
    cfg=config_for(real_case,'newton')
    model=imported_model(cfg,device)
    states=[]
    for selected in ('none','solid-block'):
        driver=BEIBStepper(model,replace(cfg,newton_preconditioner=selected),device)
        initial=driver.initialize(model.mesh.X)
        old=replace(initial,step=3299,time=.3299,force_time=.3299,pressure_time=.3299,
            force=model.force(initial.x,.3299))
        state,info=driver.step(old)
        n=info['nonlinear']
        assert n['gmres_iterations']>0 and n['tangent_assemblies']>0
        assert n['newton_iterations']==n['iterations']>0
        assert n['mass_solves']==2*n['fluid_solves']==2*n['pressure_solves']
        assert n['jacobian_actions']>0
        assert (n['preconditioner_applications']>0)==(selected=='solid-block')
        nodal,_=driver.transfer.interpolate(state.velocity,driver.transfer.prepare(old.x))
        assert torch.linalg.vector_norm(state.x-old.x-cfg.time.dt*nodal)<=1.05*n['tolerance']
        K=driver.assembler.assemble(state.x,state.time)
        dense=K.to_dense()
        expected=torch.stack([dense[3*i:3*i+3,3*i:3*i+3] for i in range(len(state.x))])
        torch.testing.assert_close(driver.assembler.nodal_diagonal_blocks(K),expected,atol=0,rtol=0)
        states.append(state)
    torch.testing.assert_close(states[0].x,states[1].x,rtol=1e-9,atol=1e-9)
    for a,b in zip(states[0].velocity,states[1].velocity):
        torch.testing.assert_close(a,b,rtol=2e-6,atol=1e-8)


def test_checkpoint_benchmark_policy_comparison_and_solver_only_resume(real_case,tmp_path):
    from validation.benchmark_paper_lv import benchmark
    from afsi_torch.simulation.paper_lv_mac import run
    cfg=config_for(real_case,'anderson-newton')
    model=imported_model(cfg)
    driver=BEIBStepper(model,cfg,'cpu')
    path=tmp_path/'run'/'checkpoint.npz'
    save(path,model,driver.initialize(model.mesh.X),cfg,{'elapsed_seconds':0.})
    original=path.read_bytes()
    report=benchmark(path,device='cpu',warmup=0,steps=1,intervals=(5,),
        solvers=('anderson-newton',),anderson_policies=('legacy','adaptive'),
        newton_preconditioners=('none','solid-block'),profile=True,profile_steps=1)
    assert path.read_bytes()==original and len(report['variants'])==4
    for v in report['variants']:
        assert v['status']=='completed' and v['max_accepted_residual_to_tolerance']<=1
        assert len(v['sampled_histories'])==1
        assert v['per_step']['mass_solves']==2*v['per_step']['fluid_solves']
    assert report['fastest_variant']['milliseconds_per_step']>0
    # Old CSV schema can resume and acquire new counters without losing rows.
    path.parent.joinpath('history.csv').write_text('step,time_s\n0,0.0\n',encoding='utf-8')
    outcome=run(resume=path,device='cpu',end_time=.0001,
        anderson_policy='adaptive',newton_preconditioner='solid-block')
    _,state,restored,_=load(path)
    assert state.step==1 and restored.anderson.extra_iterations==4
    assert restored.newton_preconditioner=='solid-block' and restored.material==cfg.material
    assert outcome['last']['mass_solves']==2*outcome['last']['fluid_solves']
    from demo.real_lv_fsi.run_mac import main
    exported=tmp_path/'config.json'
    main(['--anderson-policy','adaptive','--newton-preconditioner','solid-block','--write-config',str(exported)])
    from afsi_torch.config import load_config
    assert load_config(exported,type(cfg)).anderson.extra_iterations==4
