"""Bounded early Newton switching with unchanged endpoint acceptance."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.mac.midpoint_solver import AndersonOptions,accelerated_midpoint
from afsi_torch.nonlinear import NewtonOptions,GMRESOptions,newton,coupled_linear_policy,NonlinearFailure,NewtonResult
from test_real_lv import DEVICES,real_case


class Problem:
    def __init__(self,device):
        self.target = torch.linspace(.01,.02,8,device=device,dtype=torch.float64)
        self.gain = torch.linspace(.8,1.4,8,device=device,dtype=torch.float64)
        self.residual_calls = self.action_calls = 0
    def validate(self,x):
        if not torch.isfinite(x).all():
            raise ValueError('invalid trial')
    def residual(self,x):
        self.residual_calls += 1
        d = x-self.target
        return self.gain*d+.5*d**3
    def linearization(self,x):
        matrix = torch.diag(self.gain+1.5*(x-self.target)**2).to_sparse_csr()
        def apply(v):
            self.action_calls += 1
            return torch.sparse.mm(matrix,v[:,None]).squeeze(-1)
        return apply
    def preconditioner(self,x):
        return None


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('budget',[1,2,3,6])
def test_early_switch_keeps_original_target_and_bounded_trials(device,budget):
    problem = Problem(device)
    options = coupled_linear_policy(NewtonOptions(atol=1e-10,rtol=1e-6,
        linear=GMRESOptions(atol=1e-13,rtol=1e-3,restart=8,max_iterations=32)), 'inexact')
    initial = torch.zeros_like(problem.target)
    norm = torch.linalg.vector_norm(problem.gain*(-problem.target)+.5*(-problem.target)**3).item()
    target = max(options.atol,options.rtol*norm)
    result,info = accelerated_midpoint(problem,initial,options,
        AndersonOptions(max_iterations=budget),newton_solve=newton)
    true = problem.gain*(result.x-problem.target)+.5*(result.x-problem.target)**3
    assert result.converged and result.tolerance==target
    assert torch.linalg.vector_norm(true).item()<=target
    assert info['anderson_iterations']<=budget and info['anderson_extra_iterations']==0
    assert not torch.count_nonzero(initial)
    torch.testing.assert_close(result.x,problem.target,atol=target/.8,rtol=0)


def test_early_switch_does_not_accept_failed_newton():
    problem = Problem('cpu')
    options = NewtonOptions(atol=1e-12,rtol=0.)
    def fail(residual,x,**kwargs):
        raise NonlinearFailure('injected linear failure',NewtonResult(x,False,1.,kwargs['options'].atol,0,[]))
    with pytest.raises(NonlinearFailure,match='injected linear failure') as failure:
        accelerated_midpoint(problem,torch.zeros_like(problem.target),options,
            AndersonOptions(max_iterations=1),newton_solve=fail)
    assert not failure.value.result.converged and failure.value.result.tolerance==options.atol


def test_switch_cli_roundtrip_and_invalid_budgets(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.config import load_config
    from afsi_torch.paper_lv import PaperLVConfig
    from validation.benchmark_paper_lv import benchmark
    from afsi_torch.simulation.paper_lv_mac import run
    path = tmp_path/'settings.json'
    main(['--anderson-policy','legacy','--anderson-max-iterations','2','--write-config',str(path)])
    config = load_config(path,PaperLVConfig)
    defaults = PaperLVConfig()
    assert config.anderson.max_iterations==2 and config.anderson.extra_iterations==0
    assert config.material==defaults.material and config.time==defaults.time
    assert config.fluid==defaults.fluid and config.nonlinear==defaults.nonlinear
    for invalid in (0,-1):
        with pytest.raises(SystemExit):
            main(['--anderson-max-iterations',str(invalid),'--write-config',str(path)])
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError):
            benchmark('not-opened.npz',device='cpu',anderson_budgets=(invalid,))
        with pytest.raises(ValueError):
            run(resume='not-opened.npz',device='cpu',anderson_max_iterations=invalid)


@pytest.mark.parametrize('device',DEVICES)
def test_actual_be_endpoint_across_switch_budgets(real_case,device):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from test_paper_lv import config_for
    config = config_for(real_case,'anderson-newton')
    config = replace(config,nonlinear=coupled_linear_policy(config.nonlinear,'inexact'))
    model = imported_model(config,device)
    initial_driver = BEIBStepper(model,config,device)
    initial = initial_driver.initialize(model.mesh.X)
    # A nonzero inflation load exercises endpoint force and coupled Newton.
    initial = replace(initial,step=3299,time=.3299,force_time=.3299,pressure_time=.3299,
        force=model.force(initial.x,.3299),previous_x=initial.x.clone())
    original = initial.x.clone()
    states = []
    for budget in (6,1,2,3):
        cfg = replace(config,anderson=replace(config.anderson,max_iterations=budget,extra_iterations=0))
        driver = BEIBStepper(model,cfg,device)
        state,info = driver.step(initial,diagnostics=False)
        n = info['nonlinear']
        nodal,_ = driver.transfer.interpolate(state.velocity,driver.transfer.prepare(initial.x))
        true = state.x-initial.x-cfg.time.dt*nodal
        assert torch.linalg.vector_norm(true).item()<=1.05*n['tolerance']
        assert n['residual_norm']<=n['tolerance'] and n['anderson_iterations']<=budget
        assert n['mass_solves']==2*n['fluid_solves']
        torch.testing.assert_close(state.force,model.force(state.x,state.time),atol=1e-7,rtol=1e-10)
        torch.testing.assert_close(initial.x,original,atol=0,rtol=0)
        states.append(state)
    for state in states[1:]:
        torch.testing.assert_close(state.x,states[0].x,atol=2e-9,rtol=0)


def test_checkpoint_budget_comparison_and_resume(real_case,tmp_path):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.paper_lv_checkpoint import save,load
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.simulation.paper_lv_mac import run
    from validation.benchmark_paper_lv import benchmark
    from test_paper_lv import config_for
    cfg = config_for(real_case,'anderson-newton')
    model = imported_model(cfg)
    driver = BEIBStepper(model,cfg,'cpu')
    path = tmp_path/'run'/'checkpoint.npz'
    save(path,model,driver.initialize(model.mesh.X),cfg,{'elapsed_seconds':0.})
    original = path.read_bytes()
    report = benchmark(path,device='cpu',warmup=0,steps=1,intervals=(5,),
        solvers=('anderson-newton','newton'),anderson_budgets=(6,1,2,3),
        anderson_policies=('legacy',),linear_policies=('inexact',))
    assert original==path.read_bytes()
    variants = report['variants']
    assert len(variants)==5  # Pure Newton must not repeat for each AA budget.
    assert [v['anderson_budget'] for v in variants[:4]]==[6,1,2,3]
    for v in variants:
        assert v['status']=='completed' and v['max_accepted_residual_to_tolerance']<=1
        assert v['per_step']['mass_solves']==2*v['per_step']['fluid_solves']
        assert v['speedup_vs_first_completed']>0
    assert report['fastest_variant']['anderson_budget'] in (1,2,3,6)
    # Resume overrides only the switching budget, and persists it in config.
    run(resume=path,device='cpu',end_time=cfg.time.dt,
        anderson_policy='legacy',anderson_max_iterations=2)
    _,state,restored,_ = load(path)
    assert state.step==1 and restored.anderson.max_iterations==2
    assert restored.anderson.extra_iterations==0 and restored.material==cfg.material
    assert restored.fluid==cfg.fluid and restored.time.dt==cfg.time.dt
    # The pure Newton comparison can also be applied through the demo CLI.
    from demo.real_lv_fsi.run_mac import main
    main(['--resume',str(path),'--device','cpu','--end-time',str(2*cfg.time.dt),
          '--nonlinear-solver','newton','--anderson-max-iterations','1'])
    _,state,restored,_ = load(path)
    assert state.step==2 and restored.nonlinear_solver=='newton'
    assert restored.anderson.max_iterations==1 and restored.material==cfg.material
    assert restored.fluid==cfg.fluid and restored.time.dt==cfg.time.dt
