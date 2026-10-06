"""Independent residual acceptance, GPU CSR IB pairing and reduced work."""
from dataclasses import replace
import torch
import pytest
from afsi_torch.nonlinear import GMRESOptions,NewtonOptions,gmres,newton,coupled_linear_policy
from afsi_torch.p1 import prepare_p1
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions,AdaptiveP1Transfer
from afsi_torch.mac.assembled_transfer import AssembledP1Transfer,_cpu_entries,_bases
from afsi_torch.fluid.solvers import SolverOptions
from test_real_lv import DEVICES,real_case
from test_paper_lv import config_for


@pytest.mark.parametrize('device',DEVICES)
def test_estimated_gmres_reduces_operator_actions_with_true_acceptance(device):
    A = torch.diag(torch.linspace(1.,2.,24,device=device,dtype=torch.float64))
    A += torch.diag(torch.full((23,),.08,device=device,dtype=A.dtype),1)
    b = torch.sin(torch.arange(24,device=device,dtype=A.dtype)+.1)
    results = []
    for policy in ('periodic','estimated'):
        calls = []
        def action(v): calls.append(1); return A@v
        x,info = gmres(action,b,options=GMRESOptions(rtol=1e-9,atol=1e-13,
            check_policy=policy,check_every=5,true_check_interval=25,restart=24))
        assert torch.linalg.vector_norm(b-A@x)<=info['tolerance']
        assert info['operator_actions']==len(calls)
        results.append((x,info))
    assert results[1][1]['operator_actions']<results[0][1]['operator_actions']
    assert results[1][1]['true_residual_checks']<results[0][1]['true_residual_checks']
    torch.testing.assert_close(results[0][0],results[1][0],rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize('device',DEVICES)
def test_estimated_gmres_rejects_false_recurrence_convergence(device):
    b = torch.tensor([.03,.02,.01],device=device,dtype=torch.float64)
    def action(v):
        result = v.clone()
        if 0<torch.linalg.vector_norm(v)<.1: result += .001
        return result
    with pytest.raises(RuntimeError,match='breakdown|failed'):
        gmres(action,b,options=GMRESOptions(check_policy='estimated',rtol=1e-10,
            atol=1e-13,max_iterations=8))


@pytest.mark.parametrize('device',DEVICES)
def test_inexact_newton_keeps_nonlinear_target_and_reduces_linear_work(device):
    A = torch.diag(torch.linspace(1.,3.,20,device=device,dtype=torch.float64))
    target = A.new_full((20,),2e-8)
    options = NewtonOptions(atol=1e-9,rtol=0.,linear=GMRESOptions(rtol=1e-3,atol=1e-12))
    states = []
    for policy in ('reference','inexact'):
        calls = []
        def tangent(x):
            def action(v): calls.append(1); return A@v
            return action
        result = newton(lambda x:A@(x-target),torch.zeros_like(target),validate=lambda x:None,
                        linearization_factory=tangent,options=coupled_linear_policy(options,policy))
        assert result.converged and result.tolerance==1e-9
        assert torch.linalg.vector_norm(A@(result.x-target))<=result.tolerance
        states.append((result,len(calls)))
    assert states[1][1]<states[0][1]


def pair(device,dtype,layout='shared',chunk=512):
    X = torch.tensor([[4,4,4],[4.3,4,4],[4,4.3,4],[4,4,4.3],[4,4,2.6]],device=device,dtype=dtype)
    cells = torch.tensor([[0,1,2,3],[0,2,1,4]],device=device)
    geometry = prepare_p1(X,cells,degree=2)
    grid = MACGrid((12,14,16),(10.,12.,14.))
    quadrature = InteractionQuadratureOptions(mode='adaptive',stencil_backend=layout)
    mass = SolverOptions(rtol=2e-6,atol=1e-7) if dtype==torch.float32 else SolverOptions(rtol=1e-12,atol=1e-13)
    common = dict(quadrature_options=quadrature,fused=False,options=mass)
    reference = AdaptiveP1Transfer(grid,geometry,**common)
    assembled = AssembledP1Transfer(grid,geometry,chunk_entries=chunk,**common)
    return X,reference,assembled


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('layout',['component','shared'])
def test_csr_ib_original_quadrature_force_torque_and_power(device,dtype,layout):
    X,reference,assembled = pair(device,dtype,layout)
    old = reference.prepare(X)
    stencil = assembled.assemble_stencil(old)
    tolerance = dict(rtol=2e-5,atol=1e-6) if dtype==torch.float32 else dict(rtol=2e-11,atol=2e-12)
    force = torch.sin(1.7*X)
    density,_ = assembled.spread(force,stencil)
    expected,_ = reference.spread(force,old)
    for a,b in zip(density,expected): torch.testing.assert_close(a,b,**tolerance)
    A = X.new_tensor([[.07,.02,0],[0,-.03,.01],[.01,0,.04]])
    v = X.new_tensor([.3,-.2,.5])
    field = tuple((assembled.grid.coordinates(c,device=device,dtype=dtype)@A.T+v)[...,c] for c in range(3))
    U,_ = assembled.interpolate(field,stencil)
    oracle,_ = reference.interpolate(field,old)
    torch.testing.assert_close(U,oracle,**tolerance)
    torch.testing.assert_close(U,X@A.T+v,**tolerance)
    resultant = assembled.grid.volume*torch.stack([f.sum() for f in density])
    torch.testing.assert_close(resultant,force.sum(0),**tolerance)
    # A zero resultant moment still comes from nonzero cancelling terms.
    # Accumulate diagnostics in float64; bound float32 transfer roundoff by
    # the unsigned moment contributions, not the near-zero resultant.
    target_torque = torch.linalg.cross(X.double(),force.double()).sum(0)
    for values in (expected,density):
        torque = torch.zeros(3,device=device,dtype=torch.float64)
        unsigned_moment = torch.zeros_like(torque)
        for c,f in enumerate(values):
            vector = torch.zeros((*f.shape,3),device=device,dtype=torch.float64)
            vector[...,c] = f.double()
            coordinates = assembled.grid.coordinates(c,device=device,dtype=torch.float64)
            moment = assembled.grid.volume*torch.linalg.cross(coordinates,vector)
            torque += moment.sum((0,1,2))
            unsigned_moment += moment.abs().sum((0,1,2))
        bound = (tolerance['atol'] + tolerance['rtol']*target_torque.abs()
                 + 8*torch.finfo(dtype).eps*unsigned_moment)
        assert torch.all((torque-target_torque).abs()<=bound), (
            f'torque error {(torque-target_torque).abs().tolist()} exceeds '
            f'cancellation-aware bound {bound.tolist()}')
    torch.testing.assert_close((force*U).sum(),assembled.grid.volume*sum((u*f).sum() for u,f in zip(field,density)),**tolerance)
    assert all(B.layout==torch.sparse_csr and B.device==X.device for B in stencil.gather+stencil.spread)
    assert max(v['maximum_batch_entries'] for v in stencil.assembly['components'])<=512
    torch.testing.assert_close(reference.mass.values(),assembled.mass.values(),atol=0,rtol=0)
    # Another preparation changes geometry; this snapshot must remain valid.
    other = assembled.prepare(X+X.new_tensor([.12,.07,.04]))
    repeated,_ = assembled.spread(force,stencil)
    for a,b in zip(repeated,density): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert other is not stencil


def test_csr_budget_fails_without_truncating_weights():
    X,_,assembled = pair('cpu',torch.float64)
    assembled.max_entries = 8
    with pytest.raises(RuntimeError,match='budget exceeded'):
        assembled.prepare(X)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('layout',['component','shared'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_triton_cell_entries_match_independent_tensor_oracle(dtype,layout):
    pytest.importorskip('triton')
    from afsi_torch.mac._triton_transfer_assembly import entries
    X,reference,_ = pair('cuda',dtype,layout)
    stencil = reference.prepare(X)
    offset = 0
    for group in stencil.rule.groups:
        E,Q = len(group.cells),len(group.values)
        for c in range(3):
            base = _bases(stencil,c,offset,group.weights.numel()).reshape(E,Q,3)
            low,width = base.amin(1),base.amax(1)-base.amin(1)+4
            size = width.prod(-1)
            prefix = torch.cat((size.new_zeros(1),size.cumsum(0)))
            total = int(prefix[-1])
            for start,count in ((0,min(19,total)),(total-7,7)):
                args = (stencil,group,c,offset,low,width,prefix,start,count,reference.grid.face_shape(c))
                keys,values = entries(*args)
                expected_keys,expected_values = _cpu_entries(*args)
                torch.testing.assert_close(keys,expected_keys,atol=0,rtol=0)
                torch.testing.assert_close(values,expected_values,
                    atol=2e-8 if dtype==torch.float32 else 2e-16,
                    rtol=2e-6 if dtype==torch.float32 else 2e-13)
        offset += group.weights.numel()


@pytest.mark.parametrize('device',DEVICES)
def test_actual_endpoint_equations_and_checkpoint_solver_execution_overrides(real_case,tmp_path,device):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.paper_lv_checkpoint import save,load
    from afsi_torch.simulation.paper_lv_mac import run
    cfg = config_for(real_case,'anderson-newton')
    cfg = replace(cfg,interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'))
    model = imported_model(cfg,device)
    states = []
    for policy,backend in (('reference','quadrature'),('inexact','quadrature'),('inexact','csr')):
        selected = replace(cfg,nonlinear=coupled_linear_policy(cfg.nonlinear,policy),ib_response_backend=backend)
        driver = BEIBStepper(model,selected,device)
        old = driver.initialize(model.mesh.X)
        old = replace(old,step=3299,time=.3299,force_time=.3299,pressure_time=.3299,force=model.force(old.x,.3299))
        state,info = driver.step(old)
        nodal,_ = driver.transfer.interpolate(state.velocity,driver.transfer.prepare(old.x))
        assert torch.linalg.vector_norm(state.x-old.x-selected.time.dt*nodal)<=1.05*info['nonlinear']['tolerance']
        assert info['nonlinear']['mass_solves']==2*info['nonlinear']['fluid_solves']
        states.append(state)
    torch.testing.assert_close(states[0].x,states[2].x,rtol=0,atol=2e-9)
    if device=='cpu':
        path = tmp_path/'checkpoint.npz'
        driver = BEIBStepper(model,cfg,device)
        save(path,model,driver.initialize(model.mesh.X),cfg,{'elapsed_seconds':0})
        run(resume=path,device='cpu',end_time=cfg.time.dt,linear_policy='inexact',ib_response_backend='csr')
        _,state,restored,_ = load(path)
        assert state.step==1 and restored.ib_response_backend=='csr'
        assert restored.nonlinear.linear.check_policy=='estimated'
        assert restored.material==cfg.material and restored.time.dt==cfg.time.dt


def test_benchmark_policies_include_assembly_and_preserve_checkpoint(real_case,tmp_path):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.paper_lv_checkpoint import save
    from validation.benchmark_paper_lv import benchmark
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.config import load_config
    cfg = config_for(real_case,'anderson-newton')
    cfg = replace(cfg,interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'))
    model = imported_model(cfg)
    driver = BEIBStepper(model,cfg,'cpu')
    path = tmp_path/'seed.npz'
    save(path,model,driver.initialize(model.mesh.X),cfg,{'elapsed_seconds':0})
    before = path.read_bytes()
    report = benchmark(path,device='cpu',warmup=0,steps=1,intervals=(5,),
        solvers=('anderson-newton',),linear_policies=('reference','estimated','inexact'),
        ib_response_backends=('quadrature','csr'),profile=True,profile_steps=1)
    assert path.read_bytes()==before and len(report['variants'])==6
    for v in report['variants']:
        assert v['status']=='completed'
        assert v['max_accepted_residual_to_tolerance']<=1
        if v['ib_response_backend']=='csr':
            assert 'ib_csr_assembly' in v['profile']['phases']
            assert v['interaction_quadrature']['assembly']['total_nnz']>0
            assert v['interaction_quadrature']['csr_builds']==1
    exported = tmp_path/'settings.json'
    main(['--nonlinear-solver','anderson-newton','--interaction-quadrature','adaptive','--linear-policy','inexact',
          '--ib-response-backend','csr','--write-config',str(exported)])
    config = load_config(exported,type(cfg))
    assert config.ib_response_backend=='csr' and config.nonlinear.linear_forcing=='adaptive'
