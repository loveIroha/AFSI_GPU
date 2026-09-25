"""Preloaded-response comparisons must not hide zero response or changed solid grids."""
import numpy as np
import pytest
from validation.study_preloaded_ib import compare,relative_change


def _case(path,n,amplitude):
    X=np.arange(30,dtype=np.float64).reshape(10,3)/100
    x0=X+0.01
    cells=np.arange(10,dtype=np.int64).reshape(1,10)
    np.savez_compressed(path,X=X,x_preload=x0,x=x0+amplitude,cells=cells)
    return dict(snapshot=str(path),fluid_cells=n,time_s=.001,reference_sha256='same',completed=True,
        summary=dict(delta_cavity_ml=amplitude,max_incremental_displacement_cm=amplitude))


def test_response_comparison_uses_increment_and_rejects_changed_reference(tmp_path):
    a=_case(tmp_path/'a.npz',6,1e-5)
    b=_case(tmp_path/'b.npz',8,1.02e-5)
    result=compare(a,b)
    assert result['screen']=='within_5pct'
    assert result['relative_metrics']['displacement_nodal_l2']==pytest.approx(.02/1.02)
    b['reference_sha256']='changed'
    with pytest.raises(ValueError,match='reference'):
        compare(a,b)
    b['reference_sha256']='same'
    b['time_s']+=1e-5
    with pytest.raises(ValueError,match='physical time'):
        compare(a,b)


def test_near_zero_response_is_inconclusive(tmp_path):
    a=_case(tmp_path/'a.npz',6,0.)
    b=_case(tmp_path/'b.npz',8,0.)
    result=compare(a,b)
    assert result['screen']=='inconclusive'
    assert all(value is None for value in result['relative_metrics'].values())
    assert relative_change(0.,0.,1e-10) is None


def test_large_grid_difference_is_recorded_not_called_converged(tmp_path):
    a=_case(tmp_path/'a.npz',6,2e-5)
    b=_case(tmp_path/'b.npz',8,1e-5)
    result=compare(a,b)
    assert result['screen']=='needs_refinement'
    assert result['relative_metrics']['delta_cavity_response']==pytest.approx(1.)
