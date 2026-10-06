"""Bounded direct GPU hash assembly versus original quadrature and CSR."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.mac.assembled_transfer import AssembledP1Transfer,_bases,_cpu_entries
from afsi_torch.mac.hash_transfer_assembly import HashWorkspace,hash_slot,cpu_accumulate,sorted_entries
from test_gpu_coupled_work import pair,DEVICES
from test_real_lv import real_case


def insert(keys,values,workspace):
    if keys.is_cuda:
        pytest.importorskip('triton')
        from afsi_torch.mac._triton_transfer_assembly import hash_accumulate
        hash_accumulate(keys,values,workspace)
    else:
        cpu_accumulate(keys,values,workspace)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_hash_collisions_duplicates_zero_padding_and_overflow(device,dtype):
    # Forty distinct keys deliberately have the same initial bucket; more
    # than one warp claims it concurrently. Repeats cross program boundaries.
    colliders = [k for k in range(10000) if hash_slot(k,128)==3][:40]
    assert len(colliders)==40
    keys = torch.tensor(colliders*7+[987654],device=device,dtype=torch.int64)
    values = torch.full(keys.shape,.125,device=device,dtype=dtype); values[-1] = 0
    work = HashWorkspace(); work.reset(128,values)
    insert(keys,values,work)
    assert work.flag.item()==0
    actual,result = sorted_entries(work)
    torch.testing.assert_close(actual,torch.tensor(sorted(colliders),device=device),rtol=0,atol=0)
    torch.testing.assert_close(result,torch.full((40,),.875,device=device,dtype=dtype),rtol=0,atol=0)
    work.reset(8,values)
    insert(torch.arange(20,device=device),values[:20],work)
    assert work.flag.item()==1  # A failed table cannot silently accept data.


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('layout',['component','shared'])
def test_hash_csr_equivalent_transfer_snapshot_and_scratch_reuse(device,dtype,layout):
    X,reference,coalesce = pair(device,dtype,layout)
    hashed = AssembledP1Transfer(reference.grid,reference.geometry,
        quadrature_options=reference.quadrature_options,options=reference.options,
        fused=False,chunk_entries=512,assembly_backend='hash')
    old = reference.prepare(X)
    first = hashed.assemble_stencil(old)
    standard = coalesce.assemble_stencil(old)
    tolerance = dict(rtol=2e-5,atol=2e-6) if dtype==torch.float32 else dict(rtol=2e-11,atol=2e-12)
    for a,b in zip(first.gather+first.spread,standard.gather+standard.spread):
        torch.testing.assert_close(a.to_dense(),b.to_dense(),**tolerance)
    force = torch.sin(1.7*X)
    density,_ = hashed.spread(force,first)
    expected,_ = reference.spread(force,old)
    for a,b in zip(density,expected): torch.testing.assert_close(a,b,**tolerance)
    field = tuple(torch.sin(reference.grid.coordinates(c,device=device,dtype=dtype)[...,c]) for c in range(3))
    U,_ = hashed.interpolate(field,first)
    oracle,_ = reference.interpolate(field,old)
    torch.testing.assert_close(U,oracle,**tolerance)
    torch.testing.assert_close((force*U).sum(),reference.grid.volume*sum((u*f).sum() for u,f in zip(field,density)),**tolerance)
    resultant = reference.grid.volume*torch.stack([f.sum() for f in density])
    torch.testing.assert_close(resultant,force.sum(0),**tolerance)
    copies = [B.values().clone() for B in first.gather+first.spread]
    addresses = [w.keys.data_ptr() for w in hashed.hash_workspaces]
    # Shift inside the same allocation sizes; all numeric weights change.
    other = hashed.prepare(X+X.new_tensor([.003,.002,.001]))
    assert addresses==[w.keys.data_ptr() for w in hashed.hash_workspaces]
    assert all(w.reuses>0 for w in hashed.hash_workspaces)
    for B,saved in zip(first.gather+first.spread,copies):
        torch.testing.assert_close(B.values(),saved,atol=0,rtol=0)
    repeated,_ = hashed.spread(force,first)
    for a,b in zip(repeated,density): torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not torch.equal(other.gather[0].to_dense(),first.gather[0].to_dense())
    stats = first.assembly
    assert stats['assembly_backend']=='hash' and stats['hash_workspace_bytes']>0
    assert all(v['unique_key_sorts']==1 and v['duplicate_merge_sorts']==0 for v in stats['components'])
    if device=='cuda':
        assert all(not v['raw_entries_materialized'] and v['maximum_batch_entries']==0 for v in stats['components'])
    torch.testing.assert_close(reference.mass.values(),hashed.mass.values(),atol=0,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_hash_entry_budget_failure_is_explicit(device):
    X,r,_ = pair(device,torch.float64)
    hashed = AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
        options=r.options,fused=False,max_entries=8,assembly_backend='hash')
    with pytest.raises(RuntimeError,match='budget exceeded|probe/budget exceeded'):
        hashed.prepare(X)


@pytest.mark.parametrize('device',DEVICES)
def test_hash_rebuild_changes_quadrature_and_support_without_stale_values(device):
    X,r,_ = pair(device,torch.float64)
    hashed = AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
        options=r.options,fused=False,assembly_backend='hash')
    first = hashed.prepare(X)
    # Change both adaptive orders and lattice support, not just subcell weights.
    moved = (X-X.new_tensor([4.,4.,4.]))*1.4+X.new_tensor([4.3,4.2,4.1])
    other = hashed.prepare(moved)
    assert not torch.equal(first.rule.orders,other.rule.orders)
    force = torch.cos(moved)
    got,_ = hashed.spread(force,other)
    expected,_ = r.spread(force,r.prepare(moved))
    for a,b in zip(got,expected):
        torch.testing.assert_close(a,b,rtol=2e-11,atol=2e-12)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('layout',['component','shared'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_actual_fused_hash_cell_integrals_match_tensor_oracle(dtype,layout):
    pytest.importorskip('triton')
    from afsi_torch.mac._triton_transfer_assembly import hash_entries
    X,r,_ = pair('cuda',dtype,layout)
    stencil = r.prepare(X)
    offset = 0
    for group in stencil.rule.groups:
        E,Q = len(group.cells),len(group.values)
        for c in range(3):
            base = _bases(stencil,c,offset,group.weights.numel()).reshape(E,Q,3)
            low,width = base.amin(1),base.amax(1)-base.amin(1)+4
            size = width.prod(-1)
            prefix = torch.cat((size.new_zeros(1),size.cumsum(0)))
            count = int(prefix[-1])
            work = HashWorkspace(); work.reset(1<<(4*count-1).bit_length(),group.weights)
            shape = r.grid.face_shape(c)
            hash_entries(stencil,group,c,offset,low,width,prefix,count,shape,work)
            assert work.flag.item()==0
            keys,values = _cpu_entries(stencil,group,c,offset,low,width,prefix,0,count,shape)
            from afsi_torch.mac.assembled_transfer import _coalesce
            expected_keys,expected_values = _coalesce(keys,values)
            actual_keys,actual_values = sorted_entries(work)
            torch.testing.assert_close(actual_keys,expected_keys,atol=0,rtol=0)
            torch.testing.assert_close(actual_values,expected_values,
                rtol=3e-6 if dtype==torch.float32 else 3e-13,
                atol=3e-8 if dtype==torch.float32 else 3e-16)
        offset += group.weights.numel()


def test_hash_configuration_and_cli_roundtrip(tmp_path):
    from afsi_torch.paper_lv import PaperLVConfig
    from afsi_torch.config import load_config,save_config
    from demo.real_lv_fsi.run_mac import main
    cfg = PaperLVConfig()
    assert cfg.ib_csr_assembly_backend=='coalesce'
    with pytest.raises(ValueError,match='ib_csr_assembly_backend'):
        replace(cfg,ib_csr_assembly_backend='invalid')
    path = tmp_path/'settings.json'
    main(['--nonlinear-solver','anderson-newton','--linear-policy','inexact',
          '--ib-response-backend','csr','--ib-csr-assembly-backend','hash','--write-config',str(path)])
    restored = load_config(path,type(cfg))
    assert restored.ib_csr_assembly_backend=='hash' and restored.ib_response_backend=='csr'
    assert restored.time==cfg.time and restored.material==cfg.material
    # Older JSON without the new option has the compatible default.
    import json
    contents = json.loads(path.read_text()); del contents['ib_csr_assembly_backend']
    path.write_text(json.dumps(contents))
    assert load_config(path,type(cfg)).ib_csr_assembly_backend=='coalesce'


def test_hash_benchmark_and_resume_keep_endpoint_equations(real_case,tmp_path):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.paper_lv_checkpoint import save,load
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.simulation.paper_lv_mac import run
    from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
    from afsi_torch.nonlinear import coupled_linear_policy
    from test_paper_lv import config_for
    from validation.benchmark_paper_lv import benchmark
    cfg = config_for(real_case,'anderson-newton')
    cfg = replace(cfg,interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'),
                  ib_response_backend='csr',nonlinear=coupled_linear_policy(cfg.nonlinear,'inexact'))
    model = imported_model(cfg)
    driver = BEIBStepper(model,cfg,'cpu')
    initial = driver.initialize(model.mesh.X)
    # Exercise a loaded endpoint, rather than a zero-load startup only.
    initial = replace(initial,step=1,time=cfg.time.dt,force_time=cfg.time.dt,
        pressure_time=cfg.time.dt,force=model.force(initial.x,cfg.time.dt),previous_x=initial.x.clone())
    states = []
    for backend in ('coalesce','hash'):
        solver = BEIBStepper(model,replace(cfg,ib_csr_assembly_backend=backend),'cpu')
        state,info = solver.step(initial)
        assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
        stencil = solver.transfer.prepare(initial.x)
        nodal,_ = solver.transfer.interpolate(state.velocity,stencil)
        assert torch.linalg.vector_norm(state.x-initial.x-cfg.time.dt*nodal)<=1.05*info['nonlinear']['tolerance']
        states.append(state)
    torch.testing.assert_close(states[0].x,states[1].x,rtol=0,atol=2e-9)
    path = tmp_path/'checkpoint.npz'
    save(path,model,initial,cfg,{'elapsed_seconds':0.})
    before = path.read_bytes()
    result = benchmark(path,device='cpu',warmup=0,steps=1,intervals=(5,),solvers=('anderson-newton',),
        csr_assembly_backends=('coalesce','hash'),profile=True,profile_steps=1)
    assert path.read_bytes()==before and len(result['variants'])==2
    for variant in result['variants']:
        assert variant['status']=='completed' and variant['max_accepted_residual_to_tolerance']<=1
        assert variant['per_step']['mass_solves']==2*variant['per_step']['fluid_solves']
    hashed = result['variants'][1]
    assert 'ib_hash_accumulate' in hashed['profile']['phases']
    assert 'ib_hash_finalize' in hashed['profile']['phases']
    assert result['fastest_variant']['ib_csr_assembly_backend'] in ('coalesce','hash')
    run(resume=path,device='cpu',end_time=2*cfg.time.dt,ib_csr_assembly_backend='hash')
    _,restored,settings,_ = load(path)
    assert restored.step==2 and settings.ib_csr_assembly_backend=='hash'
    assert settings.material==cfg.material and settings.time.dt==cfg.time.dt
