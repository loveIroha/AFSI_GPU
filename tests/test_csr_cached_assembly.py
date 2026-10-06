"""Exact cached CSR updates, changing support/rules and bounded fallbacks."""
from dataclasses import replace
import torch
import pytest
from afsi_torch.mac.assembled_transfer import AssembledP1Transfer
from afsi_torch.mac.cached_transfer_assembly import (CSRPatternCache,_cpu_cached_add,
    merge_missing,finish_cached_component,numeric_snapshot)
from afsi_torch.mac.hash_transfer_assembly import HashWorkspace,cpu_accumulate,hash_slot,sorted_entries
from test_gpu_coupled_work import pair,DEVICES
from test_csr_hash_assembly import insert
from test_real_lv import real_case


def cached_pair(device,dtype,layout='shared',chunk=4096):
    X,r,oracle = pair(device,dtype,layout)
    cached = AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
        options=r.options,fused=False,chunk_entries=chunk,assembly_backend='cached-hash')
    return X,r,oracle,cached


def close_operators(actual,expected,dtype):
    tol = dict(rtol=3e-5,atol=3e-6) if dtype==torch.float32 else dict(rtol=3e-11,atol=3e-12)
    for a,b in zip(actual.gather+actual.spread,expected.gather+expected.spread):
        torch.testing.assert_close(a.to_dense(),b.to_dense(),**tol)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('layout',['component','shared'])
def test_numeric_weights_update_without_sort_and_old_snapshots_survive(device,dtype,layout):
    X,r,oracle,cached = cached_pair(device,dtype,layout)
    first = cached.prepare(X)
    saved = [B.values().clone() for B in first.gather+first.spread]
    pointers = [w.keys.data_ptr() for w in cached.hash_workspaces]
    # Repeating an identical geometry exercises the actual numeric-only path.
    repeated = cached.prepare(X)
    assert all(i['symbolic_reused'] and i['unique_key_sorts']==0 and i['transpose_key_sorts']==0
               for i in repeated.assembly['components'])
    moved = X+X.new_tensor([.0001,.0002,.0003])
    other = cached.prepare(moved)
    close_operators(other,oracle.prepare(moved),dtype)
    assert pointers==[w.keys.data_ptr() for w in cached.hash_workspaces]
    assert not torch.equal(other.gather[0].values(),first.gather[0].values())
    for B,values in zip(first.gather+first.spread,saved):
        torch.testing.assert_close(B.values(),values,atol=0,rtol=0)
    # Values and transpose values are fresh, indices stay immutable/shared.
    for a,b in zip(first.gather+first.spread,repeated.gather+repeated.spread):
        assert a.values().data_ptr()!=b.values().data_ptr()
        assert a.crow_indices().data_ptr()==b.crow_indices().data_ptr()
    force = torch.sin(1.7*moved)
    fields = tuple(torch.sin(r.grid.coordinates(c,device=device,dtype=dtype)[...,c]) for c in range(3))
    density,_ = cached.spread(force,other)
    velocity,_ = cached.interpolate(fields,other)
    expected,_ = r.spread(force,r.prepare(moved))
    tol = dict(rtol=3e-5,atol=3e-6) if dtype==torch.float32 else dict(rtol=3e-11,atol=3e-12)
    for a,b in zip(density,expected): torch.testing.assert_close(a,b,**tol)
    torch.testing.assert_close((force*velocity).sum(),r.grid.volume*sum((u*f).sum() for u,f in zip(fields,density)),**tol)
    torch.testing.assert_close(r.grid.volume*torch.stack([f.sum() for f in density]),force.sum(0),**tol)
    constant = tuple(torch.full(r.grid.face_shape(c),c+.2,device=device,dtype=dtype) for c in range(3))
    value,_ = cached.interpolate(constant,other)
    torch.testing.assert_close(value,value.new_tensor([.2,1.2,2.2]).expand_as(value),**tol)
    torque = r.grid.volume*sum(torch.cross(r.grid.coordinates(c,device=device,dtype=dtype).reshape(-1,3),
        torch.nn.functional.one_hot(torch.tensor(c,device=device),3).to(dtype)[None,:]*density[c].reshape(-1,1),dim=-1).sum(0)
        for c in range(3))
    torch.testing.assert_close(torque,torch.cross(moved,force,dim=-1).sum(0),**tol)


@pytest.mark.parametrize('device',DEVICES)
def test_new_support_and_changed_quadrature_extend_then_reuse_exact_pattern(device):
    X,r,oracle,cached = cached_pair(device,torch.float64,chunk=16384)
    first = cached.prepare(X)
    moved = (X-X.new_tensor([4.,4.,4.]))*1.4+X.new_tensor([4.3,4.2,4.1])
    other = cached.prepare(moved)
    assert not torch.equal(first.rule.orders,other.rule.orders)
    close_operators(other,oracle.prepare(moved),X.dtype)
    assert sum(i['missing_entries'] for i in other.assembly['components'])>0
    assert any(i['symbolic_extensions']>0 for i in other.assembly['components'])
    assert all(i['sorted_unique_entries']<i['nnz'] for i in other.assembly['components']
               if i['symbolic_reason']=='support-extension')
    repeated = cached.prepare(moved)
    assert all(i['symbolic_reused'] for i in repeated.assembly['components'])
    # Returning to the old support must zero the retained, now inactive keys.
    returned = cached.prepare(X)
    close_operators(returned,oracle.prepare(X),X.dtype)
    assert all(i['symbolic_reused'] for i in returned.assembly['components'])


@pytest.mark.parametrize('device',DEVICES)
def test_missing_stream_overflow_discards_partial_numeric_update(device):
    X,r,oracle,cached = cached_pair(device,torch.float64,chunk=16)
    cached.prepare(X)
    moved = X+X.new_tensor([.9,.9,.9])
    actual = cached.prepare(moved)
    close_operators(actual,oracle.prepare(moved),X.dtype)
    assert any(i['overflow_fallbacks']>0 and i['symbolic_reason']=='missing-stream-overflow-reset'
               for i in actual.assembly['components'])
    repeated = cached.prepare(moved)
    assert all(i['symbolic_reused'] for i in repeated.assembly['components'])
    cached.max_entries = 8
    with pytest.raises(RuntimeError,match='budget exceeded'):
        cached.prepare(moved)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_concurrent_cached_lookup_large_keys_missing_duplicates_and_collisions(device,dtype):
    if device=='cuda': pytest.importorskip('triton')
    # Mixed old/new keys share long probe chains, revisited by many programs.
    keys = [k+4294967296 for k in range(10000) if hash_slot(k+4294967296,256)==7][:40]
    assert len(keys)==40
    unique = torch.tensor(keys,device=device,dtype=torch.int64)
    w = HashWorkspace(); w.reset(256,torch.zeros((),device=device,dtype=dtype))
    insert(unique[:20],torch.ones(20,device=device,dtype=dtype),w)
    before = w.keys.clone(); w.values.zero_()
    repeats = 64 if device=='cuda' else 4
    stream = unique.repeat(repeats)
    values = torch.full(stream.shape,.125,device=device,dtype=dtype)
    cache = CSRPatternCache(); cache.scratch(len(stream),values)
    if device=='cuda':
        from afsi_torch.mac._triton_transfer_assembly import cached_accumulate
        cached_accumulate(stream,values,w,cache)
    else:
        _cpu_cached_add(stream,values,w,cache)
    torch.testing.assert_close(w.keys,before,atol=0,rtol=0)
    assert int(cache.missing_count)==20*repeats
    merge_missing(w,cache,int(cache.missing_count))
    assert w.flag.item()==0
    got,result = sorted_entries(w)
    torch.testing.assert_close(got,unique.sort().values,atol=0,rtol=0)
    torch.testing.assert_close(result,torch.full_like(result,repeats/8),atol=0,rtol=0)
    # Overflow is counted beyond capacity, without writing out of bounds.
    cache.scratch(3,values)
    unknown = unique+2147483648
    if device=='cuda': cached_accumulate(unknown,values[:40],w,cache)
    else: _cpu_cached_add(unknown,values[:40],w,cache)
    assert int(cache.missing_count)==40 and len(cache.missing_keys)==3


def test_cached_numeric_snapshot_rejects_nonfinite_values_and_bounded_union():
    w = HashWorkspace(); w.reset(8,torch.zeros((),dtype=torch.float64))
    cache = CSRPatternCache()
    cpu_accumulate(torch.tensor([0,1]),torch.tensor([1.,2.]),w)
    matrix,transpose = finish_cached_component(w,cache,1,(1,1,2),2)
    cpu_accumulate(torch.tensor([0]),torch.tensor([float('nan')]),w)
    with pytest.raises(FloatingPointError,match='nonfinite'):
        numeric_snapshot(w,cache,1,(1,1,2))
    torch.testing.assert_close(matrix.to_dense(),torch.tensor([[1.,2.]],dtype=torch.float64))
    torch.testing.assert_close(transpose.to_dense(),matrix.to_dense().T)


@pytest.mark.parametrize('device',DEVICES)
def test_union_budget_compacts_inactive_keys_without_changing_current_matrix(device):
    from afsi_torch.mac.cached_transfer_assembly import assemble_component_cached
    X,r,oracle,cached = cached_pair(device,torch.float64)
    moved = X+X.new_tensor([.9,.9,.9])
    expected_old,expected_new = oracle.prepare(X),oracle.prepare(moved)
    limit = max(expected_old.gather[0].values().numel(),expected_new.gather[0].values().numel())
    work,cache = HashWorkspace(),CSRPatternCache()
    old,_,_ = assemble_component_cached(r.prepare(X),0,r.grid,r.geometry.node_count,
        chunk_entries=16384,max_entries=limit,workspace=work,cache=cache)
    saved = old.values().clone()
    matrix,transpose,info = assemble_component_cached(r.prepare(moved),0,r.grid,r.geometry.node_count,
        chunk_entries=16384,max_entries=limit,workspace=work,cache=cache)
    assert info['symbolic_reason']=='union-budget-reset' and info['symbolic_resets']==1
    torch.testing.assert_close(matrix.to_dense(),expected_new.gather[0].to_dense(),rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(transpose.to_dense(),matrix.to_dense().T,atol=0,rtol=0)
    torch.testing.assert_close(old.values(),saved,atol=0,rtol=0)
    assert info['nnz']<=limit


def test_cached_cli_configuration_preserves_solver_physics(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.config import load_config
    from afsi_torch.paper_lv import PaperLVConfig
    path = tmp_path/'config.json'
    main(['--ib-response-backend','csr','--ib-csr-assembly-backend','cached-hash','--write-config',str(path)])
    cfg = load_config(path,PaperLVConfig)
    assert cfg.ib_csr_assembly_backend=='cached-hash'
    assert cfg.material==PaperLVConfig().material and cfg.time==PaperLVConfig().time


def test_failed_pattern_extension_cannot_leave_stale_indices_on_retry(monkeypatch):
    from afsi_torch.mac import cached_transfer_assembly as builder
    X,r,oracle,cached = cached_pair('cpu',torch.float64,chunk=16384)
    first = cached.prepare(X)
    saved = first.gather[0].values().clone()
    original = builder.finish_cached_component
    def reject(*args,**kwargs):
        if kwargs.get('missing'):
            raise FloatingPointError('injected rejection after key insertion')
        return original(*args,**kwargs)
    moved = X+X.new_tensor([.9,.9,.9])
    with monkeypatch.context() as patch:
        patch.setattr(builder,'finish_cached_component',reject)
        with pytest.raises(FloatingPointError,match='injected rejection'):
            cached.prepare(moved)
    assert cached.symbolic_caches[0].slots is None
    result = cached.prepare(moved)
    close_operators(result,oracle.prepare(moved),X.dtype)
    torch.testing.assert_close(first.gather[0].values(),saved,atol=0,rtol=0)


@pytest.mark.parametrize('device',DEVICES)
def test_cached_loaded_be_endpoints_match_hash(real_case,device):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
    from afsi_torch.nonlinear import coupled_linear_policy
    from test_paper_lv import config_for
    cfg = config_for(real_case,'anderson-newton')
    cfg = replace(cfg,ib_response_backend='csr',
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'),
        nonlinear=coupled_linear_policy(cfg.nonlinear,'inexact'))
    model = imported_model(cfg,device)
    initial = BEIBStepper(model,cfg,device).initialize(model.mesh.X)
    states = []
    for backend in ('hash','cached-hash'):
        driver = BEIBStepper(model,replace(cfg,ib_csr_assembly_backend=backend),device)
        state = initial
        for _ in range(2):
            previous = state
            state,info = driver.step(state,diagnostics=False)
            stencil = driver.transfer.prepare(previous.x)
            nodal,_ = driver.transfer.interpolate(state.velocity,stencil)
            assert torch.linalg.vector_norm(state.x-previous.x-cfg.time.dt*nodal)<=1.05*info['nonlinear']['tolerance']
            assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
        states.append(state)
    torch.testing.assert_close(states[0].x,states[1].x,atol=2e-9,rtol=0)
    torch.testing.assert_close(initial.x,model.mesh.X,atol=0,rtol=0)


def test_cached_benchmark_and_resume_roundtrip(real_case,tmp_path):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
    from afsi_torch.nonlinear import coupled_linear_policy
    from afsi_torch.paper_lv_checkpoint import save,load
    from afsi_torch.simulation.paper_lv_mac import run
    from validation.benchmark_paper_lv import benchmark
    from test_paper_lv import config_for
    cfg = config_for(real_case,'anderson-newton')
    cfg = replace(cfg,ib_response_backend='csr',
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'),
        nonlinear=coupled_linear_policy(cfg.nonlinear,'inexact'))
    model = imported_model(cfg)
    path = tmp_path/'run'/'checkpoint.npz'
    save(path,model,BEIBStepper(model,cfg,'cpu').initialize(model.mesh.X),cfg,{'elapsed_seconds':0.})
    saved = path.read_bytes()
    report = benchmark(path,device='cpu',warmup=1,steps=1,intervals=(5,),
        solvers=('anderson-newton',),csr_assembly_backends=('hash','cached-hash'),profile=True,profile_steps=1)
    assert path.read_bytes()==saved
    assert all(v['status']=='completed' and v['max_accepted_residual_to_tolerance']<=1 for v in report['variants'])
    cached = report['variants'][1]
    assert 'ib_cached_numeric' in cached['profile']['phases']
    assert cached['ib_csr_cache_per_step']['rebuilds']>=0
    run(resume=path,device='cpu',end_time=cfg.time.dt,ib_csr_assembly_backend='cached-hash')
    _,state,restored,_ = load(path)
    assert state.step==1 and restored.ib_csr_assembly_backend=='cached-hash'
    assert cfg.material==restored.material and cfg.time.dt==restored.time.dt
