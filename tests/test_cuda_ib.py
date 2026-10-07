"""Actual optional C++/CUDA operators against independent quadrature oracles."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.mac.assembled_transfer import AssembledP1Transfer, _cpu_entries
from afsi_torch.mac.hash_transfer_assembly import HashWorkspace, sorted_entries, cpu_accumulate, power_of_two
from afsi_torch.mac.cached_transfer_assembly import CSRPatternCache
from afsi_torch.mac import cuda_ib
from test_gpu_coupled_work import pair
from test_csr_cell_contraction import cell_kernel_case

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU unavailable')


def test_cuda_backend_config_roundtrip_and_incompatible_modes(tmp_path):
    from demo.real_lv_fsi.run_mac import main
    from afsi_torch.config import load_config
    from afsi_torch.paper_lv import PaperLVConfig
    path=tmp_path/'cuda.json'
    main(['--ib-response-backend','csr','--ib-csr-assembly-backend','cached-hash',
          '--ib-csr-contraction-backend','cuda','--write-config',str(path)])
    cfg=load_config(path,PaperLVConfig)
    assert cfg.ib_csr_contraction_backend=='cuda'
    assert cfg.material==PaperLVConfig().material and cfg.time==PaperLVConfig().time
    for change in ({'ib_response_backend':'quadrature'},{'ib_csr_assembly_backend':'coalesce'}):
        with pytest.raises(ValueError,match='cuda contraction requires'):
            replace(cfg,**change)


def test_cuda_backend_does_not_silently_run_cpu():
    _,r,_=pair('cpu',torch.float64)
    with pytest.raises(ValueError,match='requires CUDA tensors'):
        AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
            options=r.options,fused=False,assembly_backend='cached-hash',contraction_backend='cuda')


@GPU
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('q,layout,c',[(3,'shared',0),(21,'component',1),
                                     (129,'shared',2),(257,'component',0)])
def test_cuda_contract_offsets_quadrature_padding_and_int64_keys(dtype,q,layout,c):
    # Compare with torch's independent quadrature expression, not Triton.
    s,g,offset,low,width,prefix,count,face=cell_kernel_case('cuda',dtype,q,layout,c)
    keys,values=_cpu_entries(s,g,c,offset,low,width,prefix,0,count,face)
    oracle=HashWorkspace(); oracle.reset(power_of_two(count*8),values.cpu())
    cpu_accumulate(keys.cpu(),values.cpu(),oracle)
    expected_keys,expected_values=sorted_entries(oracle)
    assert expected_keys.max()>2147483647
    work=HashWorkspace(); work.reset(oracle.capacity,values)
    cuda_ib.contract(s,g,c,offset,low,width,prefix,count,face,work)
    assert work.flag.item()==0
    actual_keys,actual_values=sorted_entries(work)
    tol=dict(rtol=3e-5,atol=3e-6) if dtype==torch.float32 else dict(rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(actual_keys.cpu(),expected_keys,atol=0,rtol=0)
    torch.testing.assert_close(actual_values.cpu(),expected_values,**tol)
    # A cached pass may read keys but must not modify them.
    saved_keys=work.keys.clone(); work.values.zero_()
    cache=CSRPatternCache(); cache.scratch(4,values)
    cuda_ib.contract(s,g,c,offset,low,width,prefix,count,face,work,cache)
    assert cache.missing_count.item()==0
    torch.testing.assert_close(work.keys,saved_keys,atol=0,rtol=0)
    torch.testing.assert_close(sorted_entries(work)[1].cpu(),expected_values,**tol)


@GPU
def test_cuda_atomic_collisions_and_overflow_are_reported():
    from afsi_torch.mac.hash_transfer_assembly import hash_slot
    unique=torch.tensor([n for n in range(5000) if hash_slot(n,64)==7][:32],device='cuda')
    keys=unique.repeat(32); values=torch.full((keys.numel(),),.125,device='cuda',dtype=torch.float64)
    work=HashWorkspace(); work.reset(64,values)
    cuda_ib.hash_accumulate(keys,values,work)
    assert work.flag.item()==0
    actual,v=sorted_entries(work)
    torch.testing.assert_close(actual,unique.sort().values,rtol=0,atol=0)
    torch.testing.assert_close(v,torch.full_like(v,4.),rtol=0,atol=0)
    work.reset(8,values)
    cuda_ib.hash_accumulate(torch.arange(9,device='cuda'),values[:9],work)
    assert work.flag.item()==1 and len(sorted_entries(work)[0])==8


@GPU
@pytest.mark.parametrize('layout',['shared','component'])
@pytest.mark.parametrize('backend',['hash','cached-hash'])
def test_cuda_csr_support_rule_updates_snapshots_and_power(layout,backend):
    X,r,oracle=pair('cuda',torch.float64,layout)
    transfer=AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
        options=r.options,fused=False,chunk_entries=16384,
        assembly_backend=backend,contraction_backend='cuda')
    old=transfer.prepare(X); saved=[B.values().clone() for B in old.gather+old.spread]
    moved=(X-4)*1.4+X.new_tensor([4.3,4.2,4.1])
    for position in (moved,moved,X):
        result=transfer.prepare(position); expected=oracle.prepare(position)
        assert result.assembly['contraction_execution']=='cpp-cuda-warp-sites'
        for a,b in zip(result.gather+result.spread,expected.gather+expected.spread):
            torch.testing.assert_close(a.to_dense(),b.to_dense(),rtol=3e-11,atol=3e-12)
        for B,BT in zip(result.gather,result.spread):
            torch.testing.assert_close(B.to_dense().T,BT.to_dense(),rtol=0,atol=0)
        force=torch.sin(1.7*position)
        fields=tuple(torch.sin(r.grid.coordinates(c,device='cuda',dtype=X.dtype)[...,c]) for c in range(3))
        density,_=transfer.spread(force,result); velocity,_=transfer.interpolate(fields,result)
        torch.testing.assert_close((force*velocity).sum(),
            r.grid.volume*sum((u*f).sum() for u,f in zip(fields,density)),rtol=3e-11,atol=3e-12)
        torch.testing.assert_close(r.grid.volume*torch.stack([f.sum() for f in density]),
                                  force.sum(0),rtol=3e-11,atol=3e-12)
    for B,v in zip(old.gather+old.spread,saved):
        torch.testing.assert_close(B.values(),v,rtol=0,atol=0)


@GPU
def test_cuda_bounded_missing_stream_rebuild():
    X,r,oracle=pair('cuda',torch.float64)
    transfer=AssembledP1Transfer(r.grid,r.geometry,quadrature_options=r.quadrature_options,
        options=r.options,fused=False,chunk_entries=4,assembly_backend='cached-hash',contraction_backend='cuda')
    old=transfer.prepare(X); saved=old.gather[0].values().clone()
    moved=X+.9; current=transfer.prepare(moved); expected=oracle.prepare(moved)
    assert any(i['symbolic_reason']=='missing-stream-overflow-reset' for i in current.assembly['components'])
    for a,b in zip(current.gather,expected.gather):
        torch.testing.assert_close(a.to_dense(),b.to_dense(),rtol=3e-11,atol=3e-12)
    torch.testing.assert_close(old.gather[0].values(),saved,rtol=0,atol=0)


@GPU
def test_cuda_current_stream_graph_replay_and_input_validation():
    cuda_ib.build()
    s,g,offset,low,width,prefix,count,face=cell_kernel_case('cuda',torch.float64,21,'shared',1)
    work=HashWorkspace(); work.reset(power_of_two(count*8),g.weights)
    cuda_ib.contract(s,g,1,offset,low,width,prefix,count,face,work)
    expected_keys,expected_values=sorted_entries(work)
    expected_values=expected_values.clone()
    cache=CSRPatternCache(); cache.scratch(4,g.weights)
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph,stream=stream):
            work.values.zero_(); cache.missing_count.zero_()
            cuda_ib.contract(s,g,1,offset,low,width,prefix,count,face,work,cache)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    keys,values=sorted_entries(work)
    assert cache.missing_count.item()==0
    torch.testing.assert_close(keys,expected_keys,rtol=0,atol=0)
    torch.testing.assert_close(values,expected_values,rtol=3e-11,atol=3e-12)
    with pytest.raises(RuntimeError,match='offset is out of range'):
        cuda_ib.contract(s,g,1,s.base.size(1),low,width,prefix,count,face,work)
    with pytest.raises(RuntimeError,match='contiguous'):
        cuda_ib.hash_accumulate(work.keys[::2],work.values[::2],work)


@GPU
def test_cuda_loaded_be_endpoint_and_checkpoint_backend(real_case,tmp_path):
    from afsi_torch.paper_lv import imported_model
    from afsi_torch.paper_lv_checkpoint import save,load
    from afsi_torch.mac.paper_coupling import BEIBStepper
    from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
    from afsi_torch.nonlinear import coupled_linear_policy
    from test_paper_lv import config_for
    cfg=config_for(real_case,'anderson-newton')
    cfg=replace(cfg,interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared'),
        ib_response_backend='csr',ib_csr_assembly_backend='cached-hash',
        nonlinear=coupled_linear_policy(cfg.nonlinear,'inexact'))
    model=imported_model(cfg,'cuda')
    initial=BEIBStepper(model,cfg,'cuda').initialize(model.mesh.X)
    initial=replace(initial,step=1,time=cfg.time.dt,force_time=cfg.time.dt,
        pressure_time=cfg.time.dt,force=model.force(initial.x,cfg.time.dt),previous_x=initial.x.clone())
    states=[]
    for backend in ('sites','cuda'):
        settings=replace(cfg,ib_csr_contraction_backend=backend)
        driver=BEIBStepper(model,settings,'cuda')
        state,info=driver.step(initial)
        assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
        stencil=driver.transfer.prepare(initial.x)
        nodal,_=driver.transfer.interpolate(state.velocity,stencil)
        assert torch.linalg.vector_norm(state.x-initial.x-cfg.time.dt*nodal)<=1.05*info['nonlinear']['tolerance']
        states.append(state)
    torch.testing.assert_close(states[0].x,states[1].x,rtol=0,atol=2e-9)
    path=tmp_path/'checkpoint.npz'
    save(path,model,states[-1],settings,{'elapsed_seconds':0.})
    _,restored,loaded,_=load(path,'cuda')
    assert loaded.ib_csr_contraction_backend=='cuda'
    assert loaded.material==cfg.material and loaded.time.dt==cfg.time.dt
    torch.testing.assert_close(restored.x,states[-1].x,rtol=0,atol=0)
