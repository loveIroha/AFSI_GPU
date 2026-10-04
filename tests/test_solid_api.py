"""Alternative material/boundary models use the same assembled-CSR integrators."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from test_real_lv import DEVICES,real_case
from test_adaptive_p1_transfer import transfer
from test_mac_implicit import settings
from afsi_torch.solids import P1Solid,BoundaryForce,make_tangent
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.mac.semiimplicit import MidpointProblem
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
from afsi_torch.real_lv import imported_model


def block(device):
    X,t=transfer(device)
    X=X+2.  # Full Peskin support inside the 15 cm, 8^3 test fluid box.
    mesh=SimpleNamespace(X=X,cells=t.geometry.cells)
    # St. Venant--Kirchhoff PK1, with independently replaceable cell fields.
    def stress(F,fields,time):
        mu,lame=fields
        I=torch.eye(3,device=F.device,dtype=F.dtype)
        E=.5*(F.T@F-I)
        return F@(2*mu*E+lame*torch.trace(E)*I)
    faces=mesh.cells[:,:3]
    reference=X[faces].clone()
    def spring_and_load(nodes,fields,time):
        return -100*(nodes-fields[0])+nodes.new_tensor([1200.,100.,0.])*(1+time)
    term=BoundaryForce(faces,spring_and_load,(reference,))
    return P1Solid(mesh,stress,cell_fields=(X.new_full((2,),5000.),X.new_full((2,),1000.)),boundaries=(term,))


@pytest.mark.parametrize('device',DEVICES)
def test_replaceable_material_boundary_csr_matches_force_derivative(device):
    m=block(device)
    x=m.mesh.X+1e-3*torch.sin(m.mesh.X)
    v=.01*torch.cos(x)
    assembly=make_tangent(m,1)
    K=assembly.assemble(x,.6)
    expected=torch.func.jvp(lambda y:m.force(y,.6),(x,),(v,))[1]
    actual=torch.sparse.mm(K,v.reshape(-1,1)).reshape_as(x)
    assert K.layout==torch.sparse_csr and K.shape==(x.numel(),x.numel())
    torch.testing.assert_close(actual,expected,rtol=2e-10,atol=2e-9)
    finite=(m.force(x+1e-5*v,.6)-m.force(x-1e-5*v,.6))/2e-5
    torch.testing.assert_close(actual,finite,rtol=3e-6,atol=2e-6)
    torch.testing.assert_close(assembly.diagonal(K),K.to_dense().diagonal(),rtol=0,atol=0)
    # Cell/boundary parameters are actually independent, not just renamed.
    free=P1Solid(m.mesh,m.stress,cell_fields=m.cell_fields)
    torch.testing.assert_close(free.force(x,.6).sum(0),torch.zeros(3,device=device,dtype=x.dtype),atol=1e-12,rtol=0)
    assert torch.linalg.vector_norm(m.force(x,.6)-free.force(x,.6))>1000


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('scheme',['implicit-newton','cnab-semiimplicit'])
def test_generic_solid_uses_shared_coupler_and_csr_newton(real_case,device,scheme):
    m=block(device)
    cfg=replace(real_case,coupling=MACCouplingOptions(scheme=scheme))
    driver=build_driver(m,settings(cfg),device)
    state=driver.initialize(m.mesh.X)
    state=replace(state,step=6000,time=.6,force_time=.6,force=m.force(state.x,.6),
        previous_advection=driver.flow.advection(state.velocity) if scheme=='cnab-semiimplicit' else None)
    result,info=driver.step(state)
    assert info['nonlinear']['residual_norm']<=info['nonlinear']['tolerance']
    assert info['nonlinear']['iterations']>0
    assert result.force_time==pytest.approx(result.time)
    assert torch.linalg.vector_norm(result.x-state.x)>0
    assert driver.tangent.assemble(result.x,result.time).layout==torch.sparse_csr


@pytest.mark.parametrize('device',DEVICES)
def test_generic_optimized_midpoint_matches_reference_and_linearization(real_case,device):
    m=block(device)
    cfg=replace(real_case,coupling=MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton',stokes_warm_start=True,reuse_validation=True),
        interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',stencil_backend='shared',prepare_backend='triton'))
    a=build_driver(m,settings(cfg),device)
    opt=replace(cfg,execution=replace(cfg.execution,execution_backend='fused',coupling_backend='optimized'))
    b=build_driver(m,settings(opt),device)
    sa=a.initialize(m.mesh.X)
    sa=replace(sa,step=6000,time=.6,force_time=.6,force=m.force(sa.x,.6),previous_advection=a.flow.advection(sa.velocity))
    sb=sa
    for _ in range(3):
        sa,ia=a.step(sa)
        sb,ib=b.step(sb)
        torch.testing.assert_close(sb.x,sa.x,rtol=2e-9,atol=2e-11)
        for u,v in zip(sa.velocity,sb.velocity):
            torch.testing.assert_close(u,v,rtol=2e-6,atol=2e-9)
        assert ib['nonlinear']['residual_norm']<=ib['nonlinear']['tolerance']
        assert ib['power_error']/max(abs(ib['solid_power']),abs(ib['fluid_power']),1.)<1e-8
    problem=MidpointProblem(a,sa,sa.x,a.transfer.prepare(sa.x),a.flow.advection(sa.velocity))
    y=torch.zeros_like(sa.x); v=.01*torch.sin(sa.x)
    action=problem.linearization(y)
    finite=(problem.residual(y+1e-4*v)-problem.residual(y-1e-4*v))/2e-4
    torch.testing.assert_close(action(v),finite,rtol=3e-5,atol=3e-8)
    from afsi_torch.solids.contracts import failure_diagnostics
    assert failure_diagnostics(m,a.flow.grid,sa,problem)['accepted_solid']['minimum_detF']>0


def test_existing_ho_factory_preserves_original_assembler_and_diagnostics(real_case):
    from afsi_torch.ho_tangent import HOTangentAssembler
    m=imported_model(real_case)
    a=HOTangentAssembler(m,37).assemble(m.mesh.X,.6)
    b=make_tangent(m,37).assemble(m.mesh.X,.6)
    torch.testing.assert_close(a.values(),b.values(),rtol=0,atol=0)
    assert type(make_tangent(m)) is HOTangentAssembler
    assert callable(m.failure_diagnostics)


def test_generic_element_and_boundary_kernels_trace_full_graph():
    m=block('cpu')
    x=m.mesh.X+1e-3*torch.sin(m.mesh.X)
    time=x.new_tensor(.6)
    force=torch.compile(m.force_from_geometry,backend='eager',fullgraph=True)
    torch.testing.assert_close(force(x,m.element_gradient(x),time),m.force(x,.6))
    tangent=make_tangent(m,1)
    volume=torch.compile(tangent._volume,backend='eager',fullgraph=True)
    args=(m.element_gradient(x),m.gradients,m.volumes,time,m.cell_fields)
    torch.testing.assert_close(volume(*args),tangent._volume(*args))
    term=m.boundaries[0]
    boundary=tangent._boundary_kernels[0]
    args=(x[term.connectivity],time,term.fields)
    torch.testing.assert_close(torch.compile(boundary,backend='eager',fullgraph=True)(*args),boundary(*args))


def test_missing_capability_fails_before_fluid_allocation(real_case):
    m=block('cpu')
    cfg=replace(real_case,coupling=MACCouplingOptions(scheme='cnab-semiimplicit'))
    with pytest.raises(TypeError,match='tangent_factory'):
        build_driver(SimpleNamespace(mesh=m.mesh,geometry=m.geometry,force=m.force,validate=m.validate),settings(cfg),'cpu')
    def bad_factory(chunk_size):
        return object()
    m.tangent_factory=bad_factory
    with pytest.raises(TypeError,match='assemble'):
        make_tangent(m)


def test_composable_validity_and_cached_geometry_reject_mutated_state():
    m=block('cpu')
    execution=m.execution_factory()
    x=m.mesh.X.clone()
    execution.validate(x)
    torch.testing.assert_close(execution.force(x,.6),m.force(x,.6))
    # In-place edits must invalidate the geometry cache, including inversion.
    x[1]=2*x[0]-x[1]
    with pytest.raises(ValueError,match='deformation'):
        execution.validate(x)
    check=lambda y:torch.linalg.vector_norm(y-m.mesh.X,dim=-1)<.01
    guarded=P1Solid(m.mesh,m.stress,cell_fields=m.cell_fields,validity_checks=(check,))
    with pytest.raises(ValueError,match='validity'):
        guarded.validate(m.mesh.X+.02)
    with pytest.raises(TypeError,match='BoundaryForce'):
        P1Solid(m.mesh,m.stress,cell_fields=m.cell_fields,boundaries=(object(),))
