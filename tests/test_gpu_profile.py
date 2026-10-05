"""Profile evidence respects overlap, frozen IB inputs and restart history."""
from dataclasses import replace
import json
import pytest
import torch
from test_real_lv import real_case, DEVICES
from test_mac_shared_pressure import warm_config
from validation.gpu_trace import RangeRecorder, summarize_trace, union_duration
from validation.profile_real_lv_gpu import profile_checkpoint, prepare, ib_operations


def test_trace_union_clipping_and_missing_events(tmp_path):
    assert union_duration([(0,5),(2,4),(4,9),(20,21),(10,10)])==10
    path = tmp_path/'trace.json'
    def event(name,cat,ts,dur):return dict(name=name,cat=cat,ph='X',ts=ts,dur=dur)
    events = [event('afsi.capture','user_annotation',1000,10000),
              event('gather','kernel',900,1100),event('spread','kernel',1500,1500),
              event('copy','gpu_memcpy',3000,1000),event('outside','kernel',12000,1000),
              event('cudaStreamSynchronize','cuda_runtime',5000,2000),
              event('cudaLaunchKernel','cuda_runtime',1100,20),
              event('invalid','kernel',0,-1)]
    path.write_text(json.dumps(dict(traceEvents=events)))
    result = summarize_trace(path)
    assert result['capture_ms']==10 and result['gpu_kernel_union_ms']==2
    assert result['gpu_activity_union_ms']==3
    assert result['capture_without_recorded_gpu_activity_ms']==7
    assert result['gpu_kernel_launches']==2 and result['host_launch_api_calls']==1
    assert result['host_synchronizations'][0]['total_ms']==2
    path.write_text(json.dumps([events[0]]))
    result = summarize_trace(path)
    assert not result['gpu_events_available']
    assert result['capture_without_recorded_gpu_activity_ms'] is None


def test_range_wrapping_preserves_exception_and_does_not_synchronize(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('unexpected synchronization')
    monkeypatch.setattr(torch.cuda,'synchronize',forbidden)
    recorder = RangeRecorder()
    def inner(value):return value+1
    wrapped = recorder.wrap(inner,'nested')
    with recorder.range('outer'):assert wrapped(3)==4
    def fail():raise ValueError('expected')
    with pytest.raises(ValueError,match='expected'):recorder.wrap(fail,'failure')()
    assert dict(recorder.calls)==dict(outer=1,nested=1,failure=1)


def _checkpoint(real_case,tmp_path):
    from afsi_torch.simulation.real_lv_mac import run
    cfg = warm_config(real_case)
    cfg = replace(cfg,interaction_quadrature=replace(cfg.interaction_quadrature,transfer_backend='fused'))
    run(case_config=cfg,device='cpu',output=tmp_path/'source')
    return tmp_path/'source'/'checkpoint.npz'


@pytest.mark.parametrize('device',DEVICES)
def test_profile_read_only_history_and_frozen_kernel_scope(real_case,tmp_path,device):
    pytest.importorskip('basix')
    path = _checkpoint(real_case,tmp_path)
    before = path.read_bytes()
    from afsi_torch.real_lv_checkpoint import load_real_lv
    _,state,_,_,_ = load_real_lv(path,device)
    _,restored,driver = prepare(path,device,'reference')
    assert restored.previous_dt==state.previous_dt and restored.pressure_time==state.pressure_time
    for a,b in zip(restored.previous_advection,state.previous_advection):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    stencil = driver.transfer.prepare(restored.x)
    coefficient,_ = driver.transfer.solve_mass(restored.force,None)
    operations = ib_operations(driver.transfer,restored.velocity,coefficient,stencil)
    def forbidden(*a,**k):raise AssertionError('mass/prepare entered pure kernel scope')
    driver.transfer.solve_mass = forbidden
    driver.transfer.prepare = forbidden
    assert operations['gather']().shape==restored.x.shape
    assert len(operations['spread']())==3
    report = profile_checkpoint(path,device=device,scope='ib',warmup=1,repeats=2,batches=1,
                                steps=1,output=tmp_path/'ib')
    assert report['source_time_s']==report['capture_end_time_s']
    assert not report['frozen_stencil']['mass_solve_in_timing']
    assert report['equivalence']['gather']['max_abs']<1e-11
    assert (tmp_path/'ib'/'trace.json').exists()
    assert (tmp_path/'ib'/'operators.txt').exists()
    coupled = profile_checkpoint(path,device=device,warmup=1,batches=1,steps=1,output=tmp_path/'coupled')
    assert coupled['capture_end_time_s']>coupled['source_time_s']
    assert coupled['ranges']['step']==1 and coupled['ranges']['mass_solves']>0
    assert path.read_bytes()==before
    with pytest.raises(FileExistsError):
        profile_checkpoint(path,device=device,scope='ib',output=tmp_path/'ib')
