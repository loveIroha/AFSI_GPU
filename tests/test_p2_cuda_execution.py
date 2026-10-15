"""Same P2 quadrature transfer, including curved geometry and old ownership."""
from dataclasses import replace
import pytest
import torch
from afsi_torch.mac.adaptive_transfer import AdaptiveFETransfer,InteractionQuadratureOptions
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.p2_cuda_transfer import assemble_velocity,spread
from afsi_torch.solid import prepare_p2
from afsi_torch.tetrahedron import reference_nodes

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')


def pair(dtype=torch.float64,scale=.31,shape=(16,12,10),density=3.):
    X=reference_nodes(device='cuda',dtype=dtype)*scale+1.831
    cells=torch.arange(10,device='cuda')[None]
    geometry=prepare_p2(X,cells)
    grid=MACGrid(shape,(5.,5.5,6.),(-.2,.1,-.4))
    options=InteractionQuadratureOptions(mode='adaptive',point_density=density,max_order=22,
        max_points=500000,rule_family='xiao-gimbutas')
    reference=AdaptiveFETransfer(grid,geometry,fused=False,ib_backend='cuda',quadrature_options=options)
    native=AdaptiveFETransfer(grid,geometry,fused=False,ib_backend='cuda',
        quadrature_options=replace(options,transfer_backend='fused',stencil_backend='shared',
            prepare_backend='cuda',reuse_stencil_buffers=True))
    return X,reference,native


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('scale,shape,density',[(.31,(16,12,10),3.),(.31,(32,)*3,3.),(.9,(32,)*3,2.)])
def test_actual_p2_cuda_raw_operators_and_constant_affine_projection(dtype,scale,shape,density):
    X,a,b=pair(dtype,scale,shape,density)
    x=X.clone();x[4,1]+=.017
    sa,sb=a.prepare(x),b.prepare(x)
    assert torch.equal(sa.rule.orders,sb.rule.orders)
    expanded=sb.expanded()
    # Floating point evaluation near a lattice knot may pick either adjacent
    # base at FP32; the operator below is the invariant, not the base index.
    tol=dict(rtol=8e-5,atol=3e-5) if dtype==torch.float32 else dict(rtol=4e-12,atol=2e-12)
    torch.testing.assert_close(expanded.phi,sa.phi,**tol)
    coefficient=torch.sin(X*1.3)
    velocity=tuple(torch.sin(a.grid.coordinates(c,device='cuda',dtype=dtype)[...,(c+1)%3]) for c in range(3))
    expected_force=torch.cat([a._weighted_kernel(coefficient,g.values,g.cells,g.weights) for g in sa.rule.groups])
    fa=a.spread_grid(expected_force,sa);fb=spread(b,coefficient,sb)
    for actual,expected in zip(fb,fa):torch.testing.assert_close(actual,expected,**tol)
    gathered=a.gather_grid(velocity,sa)
    rhs=torch.zeros_like(a.diagonal);offset=0
    for g in sa.rule.groups:
        count=g.weights.numel()
        rhs+=a._assemble_kernel(gathered[offset:offset+count],g.values,g.cells,g.weights,a.diagonal)
        offset+=count
    actual=assemble_velocity(b,velocity,sb)
    torch.testing.assert_close(actual,rhs,**tol)
    solid=(coefficient*actual).sum()
    fluid=b.grid.volume*sum((u*f).sum() for u,f in zip(velocity,fb))
    torch.testing.assert_close(solid,fluid,**tol)
    if dtype==torch.float64:
        A=X.new_tensor([[.02,.01,0.],[0.,-.03,.01],[.01,0.,.04]])
        constant=X.new_tensor([.3,-.2,.5])
        affine=tuple((a.grid.coordinates(c,device='cuda')@A.T+constant)[...,c] for c in range(3))
        projected,_=b.interpolate(affine,sb)
        torch.testing.assert_close(projected,x@A.T+constant,atol=3e-11,rtol=3e-10)
        f=torch.sin(2*X)
        density_grid,_=b.spread(f,sb)
        total=torch.stack([b.grid.volume*v.sum() for v in density_grid])
        torch.testing.assert_close(total,f.sum(0),atol=5e-10,rtol=3e-10)


def test_p2_cuda_owned_buffers_survive_later_rules_and_failed_support():
    X,a,b=pair()
    old=b.prepare(X)
    saved=old.phi.clone()
    x=X.clone();x[4,0]+=.2
    new=b.prepare(x)
    third=b.prepare(x+.05)
    assert old.base.data_ptr()!=new.base.data_ptr()!=third.base.data_ptr()
    torch.testing.assert_close(old.phi,saved,atol=0,rtol=0)
    velocity=tuple(torch.ones_like(v) for v in b.grid.zeros(device='cuda'))
    before=assemble_velocity(b,velocity,old)
    with pytest.raises(ValueError,match='support'):
        b.prepare(X-2.)
    after=assemble_velocity(b,velocity,old)
    torch.testing.assert_close(after,before,atol=2e-14,rtol=3e-13)
    assert b.quadrature_summary()['stencil_workspace']['overflow_allocations']>=1


def test_p2_cuda_requires_native_device_and_explicit_backend():
    X=reference_nodes()*.3+1.8
    opt=InteractionQuadratureOptions(mode='adaptive',transfer_backend='fused',stencil_backend='shared',prepare_backend='cuda')
    with pytest.raises(ValueError,match='CUDA'):
        AdaptiveFETransfer(MACGrid((8,)*3,(5.,)*3),prepare_p2(X,torch.arange(10)[None]),quadrature_options=opt)


def test_native_midpoint_startup_checkpoint_switch_and_resume(tmp_path):
    from dataclasses import asdict
    from afsi_torch.config import LVSimulationConfig,TimeConfig,FluidConfig,OutputConfig,LVExecutionConfig
    from afsi_torch.mac.cnab import CNABOptions
    from afsi_torch.mac.implicit import MACCouplingOptions
    from afsi_torch.mac.checkpoint import load_mac
    from afsi_torch.mac.execution import build_driver
    from afsi_torch.mac.p2_cuda_transfer import execution_settings
    from afsi_torch.simulation.lv_mac import run
    cfg=replace(LVSimulationConfig(),time=TimeConfig(5e-5,1.5e-4),
        geometry=replace(LVSimulationConfig().geometry,mesh_size=.5),fluid=FluidConfig((16,)*3),
        coupling=MACCouplingOptions(scheme='cnab-midpoint',cnab=CNABOptions(advection='xsppm7',stokes_solver='fgmres',max_stokes_iterations=100)),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',point_density=3.,max_order=12,
            transfer_backend='fused',stencil_backend='shared',prepare_backend='cuda',reuse_stencil_buffers=True),
        execution=LVExecutionConfig(execution_backend='fused',pressure_backend='graph',solid_backend='pointwise',
            mass_backend='graph',coupling_backend='optimized',warm_start=True,ib_backend='cuda'),
        output=OutputConfig(1,1,1,False))
    folder=tmp_path/'saved'
    run(case_config=cfg,device='cuda',output=folder,end_time=5e-5)
    model,state,settings,_=load_mac(folder/'checkpoint.npz','cuda')
    assert settings['interaction_quadrature']['prepare_backend']=='cuda'
    old=settings['interaction_quadrature'].copy()
    restored=execution_settings(settings,'reference')
    assert settings['interaction_quadrature']==old
    assert restored['interaction_quadrature']['point_density']==old['point_density']
    reference=build_driver(model,restored,'cuda')
    native=build_driver(model,settings,'cuda')
    expected,er=reference.step(state)
    accepted,info=native.step(state)
    torch.testing.assert_close(accepted.x,expected.x,atol=5e-13,rtol=3e-13)
    for a,b in zip(accepted.velocity,expected.velocity):torch.testing.assert_close(a,b,atol=3e-10,rtol=3e-8)
    assert info['power_error']<1e-8*max(abs(info['solid_power']),1.)
    run(resume=folder/'checkpoint.npz',device='cuda',end_time=1e-4)
    _,resumed,_,_=load_mac(folder/'checkpoint.npz','cuda')
    torch.testing.assert_close(resumed.x,accepted.x,atol=5e-13,rtol=3e-13)
    run(resume=folder/'checkpoint.npz',device='cuda',end_time=1.5e-4,p2_execution='reference')
    _,_,switched,_=load_mac(folder/'checkpoint.npz','cuda')
    assert switched['interaction_quadrature']['prepare_backend']=='torch'
