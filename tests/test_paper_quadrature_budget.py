"""Deforming active LV: high-order IB and budget-only checkpoint recovery."""
from dataclasses import replace, asdict
from itertools import product, permutations
from math import factorial
import json
import pytest
import torch
from afsi_torch.mac.adaptive_transfer import (AdaptiveP1Transfer, InteractionQuadratureOptions,
    QuadratureBudgetError, gaussian_tetra_rule, increase_quadrature_budget)
from afsi_torch.mac.assembled_transfer import AssembledP1Transfer
from afsi_torch.mac.grid import MACGrid
from afsi_torch.p1 import prepare_p1

DEVICES = ['cpu', pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))]


@pytest.mark.parametrize('order',[9,12,22])
def test_high_order_fallback_positive_mass_and_polynomial_moments(order):
    q,w = gaussian_tetra_rule(order,torch.zeros((),dtype=torch.float64),'xiao-gimbutas')
    assert len(w)==order**3 and (w>0).all() and (q>=0).all() and (q.sum(-1)<=1).all()
    N = torch.cat((1-q.sum(-1,keepdim=True),q),-1)
    expected = (torch.ones((4,4),dtype=q.dtype)+torch.eye(4,dtype=q.dtype))/120
    torch.testing.assert_close(N.T@(w[:,None]*N),expected,atol=2e-15,rtol=2e-13)
    degree = 2*order-1
    # All monomials through degree 17 at the formerly failing order 9;
    # selected mixed and extremal moments at larger supported orders.
    powers = ([(a,b,c) for a in range(degree+1) for b in range(degree+1-a)
               for c in range(degree+1-a-b)] if order==9 else
              [(degree,0,0),(0,degree,0),(0,0,degree),(degree//3,)*3,(2,3,degree-5)])
    for a,b,c in powers:
        exact = factorial(a)*factorial(b)*factorial(c)/factorial(a+b+c+3)
        actual = (w*q[:,0]**a*q[:,1]**b*q[:,2]**c).sum().item()
        assert actual==pytest.approx(exact,rel=5e-12,abs=1e-28)


@pytest.mark.parametrize('device',DEVICES)
def test_deformation_crosses_order_eight_with_csr_duality_and_frozen_old_stencil(device):
    pytest.importorskip('basix')
    X = torch.tensor([[5,5,5],[8.8,5,5],[5,5.1,5],[5,5,5.1]],device=device,dtype=torch.float64)
    cells = torch.arange(4,device=device).reshape(1,4)
    geometry = prepare_p1(X,cells,degree=2)
    grid = MACGrid((16,)*3,(16.,)*3)
    opts = InteractionQuadratureOptions(mode='adaptive',rule_family='xiao-gimbutas',
        stencil_backend='shared',max_order=8,max_points=1000)
    old = AdaptiveP1Transfer(grid,geometry,quadrature_options=opts,fused=False)
    moved = X.clone(); moved[1,0] = 9.1
    with pytest.raises(QuadratureBudgetError) as error:
        old.check_configuration_support(moved)
    assert error.value.diagnosis['maximum_order']==9
    assert error.value.diagnosis['gaussian_order_cell_counts'][9]==1
    opts = increase_quadrature_budget(opts,max_order=22)
    oracle = AdaptiveP1Transfer(grid,geometry,quadrature_options=opts,fused=False)
    cached = AssembledP1Transfer(grid,geometry,quadrature_options=opts,fused=False,
        assembly_backend='cached-hash',contraction_backend='sites',chunk_entries=4096)
    initial = cached.prepare(X)
    mass = cached.mass.values().clone()
    snapshots = [a.values().clone() for a in initial.gather+initial.spread]
    current = cached.prepare(moved)
    reference = oracle.prepare(moved)
    assert initial.rule.point_count==214 and current.rule.point_count==729
    field = tuple((grid.coordinates(c,device=device)[...,c]*.1+.3) for c in range(3))
    velocity,_ = cached.interpolate(field,current)
    torch.testing.assert_close(velocity,moved*.1+.3,rtol=2e-10,atol=2e-11)
    force = torch.sin(X)
    density,_ = cached.spread(force,current)
    expected,_ = oracle.spread(force,reference)
    for actual,target in zip(density,expected):
        torch.testing.assert_close(actual,target,rtol=3e-10,atol=3e-11)
    torch.testing.assert_close(grid.volume*torch.stack([v.sum() for v in density]),
        force.sum(0),rtol=2e-10,atol=2e-11)
    torch.testing.assert_close((force*velocity).sum(),
        grid.volume*sum((u*f).sum() for u,f in zip(field,density)),rtol=2e-10,atol=2e-11)
    torch.testing.assert_close(cached.mass.values(),mass,rtol=0,atol=0)
    for matrix,saved in zip(initial.gather+initial.spread,snapshots):
        torch.testing.assert_close(matrix.values(),saved,atol=0,rtol=0)
    # Increasing the order ceiling must not disable the independent point cap.
    limited = AdaptiveP1Transfer(grid,geometry,quadrature_options=replace(opts,max_points=728),fused=False)
    with pytest.raises(QuadratureBudgetError) as error:
        limited.prepare(moved)
    assert error.value.diagnosis['resource']=='points' and error.value.diagnosis['required']==729


def test_budget_override_cli_and_physics_are_separate(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.paper_lv import PaperLVConfig
    opts = InteractionQuadratureOptions(mode='adaptive')
    raised = increase_quadrature_budget(opts,max_order=22,max_points=13000000)
    assert raised.point_density==opts.point_density and raised.rule_family==opts.rule_family
    for change in ({'max_order':7},{'max_points':11_000_000},{'max_order':23}):
        with pytest.raises(ValueError):
            increase_quadrature_budget(opts,**change)
    path = tmp_path/'config.json'
    main(['--ib-max-order','12','--ib-max-points','14000000','--write-config',str(path)])
    saved = json.loads(path.read_text())
    assert saved['interaction_quadrature']['max_order']==12
    assert saved['interaction_quadrature']['max_points']==14000000
    assert saved['material']==asdict(PaperLVConfig().material)
    assert PaperLVConfig().interaction_quadrature.max_order==22
    with pytest.raises(SystemExit):
        main(['--resume','unused.npz','--ib-point-density','3'])
    with pytest.raises(SystemExit):
        main(['--interaction-quadrature','fixed','--ib-max-order','22'])


def small_model():
    """Analytic open well, positive cavity, no Gmsh or external input files."""
    from afsi_torch.mesh_io import ImportedSolidMesh
    from afsi_torch.paper_lv import PaperLVConfig,PaperLVSolid
    from afsi_torch.config import TimeConfig,FluidConfig,OutputConfig,LVExecutionConfig
    xyz = list(product(range(4),range(4),range(3)))
    ids = {p:i for i,p in enumerate(xyz)}
    X = torch.tensor(xyz,dtype=torch.float64)+5
    elements = []
    for origin in product(range(3),range(3),range(2)):
        if origin==(1,1,1):
            continue
        for order in permutations(range(3)):
            vertex = list(origin); tetra = [ids[tuple(vertex)]]
            for axis in order:
                vertex[axis] += 1; tetra.append(ids[tuple(vertex)])
            if torch.linalg.det((X[tetra[1:]]-X[tetra[0]]).T)<0:
                tetra[1],tetra[2] = tetra[2],tetra[1]
            elements.append(tetra)
    cells = torch.tensor(elements)
    owners = {}
    for e,cell in enumerate(elements):
        for local,vertices in enumerate(((1,2,3),(0,3,2),(0,1,3),(0,2,1))):
            face = [cell[i] for i in vertices]
            owners.setdefault(tuple(sorted(face)),[]).append((face,e,local))
    faces,bc,bl,tags = [],[],[],[]
    for entries in owners.values():
        if len(entries)!=1:
            continue
        face,e,local = entries[0]; center = X[face].mean(0)
        tag = 1 if 6<=center[0]<=7 and 6<=center[1]<=7 and center[2]>=6 else 2
        if center[2]==7:
            tag = 3
        faces.append(face); bc.append(e); bl.append(local); tags.append(tag)
    mesh = ImportedSolidMesh(X,cells,torch.tensor(faces),torch.tensor(tags),torch.tensor(bc),
        torch.tensor(bl),X.new_tensor([[1.,0.,0.]]).expand(len(cells),3).clone(),
        X.new_tensor([[0.,1.,0.]]).expand(len(cells),3).clone(),len(X),{'units':'cm'})
    cfg = PaperLVConfig(load_protocol='active-cycle',time=TimeConfig(1e-4,1e-4),
        fluid=FluidConfig((8,)*3,(13.,)*3,mu=.04),execution=LVExecutionConfig(warm_start=True),
        output=OutputConfig(1,1,1,False),nonlinear_solver='anderson-newton',
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'),
        ib_response_backend='csr',ib_csr_assembly_backend='cached-hash')
    return PaperLVSolid(mesh,cfg),cfg


def test_failed_checkpoint_resume_cli_preserves_state_loads_and_records_budget(tmp_path,monkeypatch):
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.paper_lv_checkpoint import save,load
    from demo.real_lv_fsi.run_mac import main
    model,cfg = small_model()
    initial = BEIBStepper(model,cfg,'cpu').initialize(model.mesh.X)
    expected,_ = BEIBStepper(model,cfg,'cpu').step(initial)
    path = tmp_path/'run'/'checkpoint.npz'
    save(path,model,initial,cfg,dict(elapsed_seconds=0.))
    # Exercise the runner's real failure path after an endpoint solve. The
    # isolated 8->9 geometric transition is tested above without a fluid solve.
    def budget_failure(self,x):
        raise QuadratureBudgetError('order',9,cfg.interaction_quadrature,
            torch.tensor([9],device=x.device))
    with monkeypatch.context() as patch:
        patch.setattr(BEIBStepper,'check_support',budget_failure)
        with pytest.raises(QuadratureBudgetError):
            main(['--resume',str(path),'--device','cpu'])
    _,failed,_,progress = load(path)
    assert failed.step==0 and progress['failure']['diagnosis']['required']==9
    torch.testing.assert_close(failed.x,initial.x,atol=0,rtol=0)
    report = main(['--resume',str(path),'--device','cpu','--ib-max-order','22'])
    _,state,restored,progress = load(path)
    assert restored.interaction_quadrature.max_order==22
    assert replace(restored,interaction_quadrature=cfg.interaction_quadrature)==cfg
    assert 'failure' not in report and len(progress['previous_failures'])==1
    assert len(progress['quadrature_budget_changes'])==1
    assert report['status']=='completed' and state.step==1
    torch.testing.assert_close(state.x,expected.x,rtol=1e-11,atol=1e-12)
    for a,b in zip(state.velocity,expected.velocity):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
