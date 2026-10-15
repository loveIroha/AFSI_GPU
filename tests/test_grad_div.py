"""Exact full grad-div assembly, coupled solve, and zero-coefficient compatibility."""
import pytest
import torch
from afsi_torch.fluid import create_box, prepare_operators, CSRFluidOperators, ChorinSolver, SolverOptions

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', DEVICES)
def test_full_csr_blocks_match_quadrature_energy_symmetry_diagonal_and_cross_components(device):
    mesh = create_box((2,3,2), (1.3,2.1,.7), (-.2,.1,-.5), device=device)
    op = prepare_operators(mesh)
    csr = CSRFluidOperators(op)
    u = torch.sin(2.3*mesh.velocity_coordinates)
    v = torch.cos(1.7*mesh.velocity_coordinates)
    Bu, Bv = csr.velocity_grad_div(u), csr.velocity_grad_div(v)
    torch.testing.assert_close(Bu, op.velocity_grad_div(u), rtol=3e-12, atol=3e-14)
    torch.testing.assert_close((u*Bv).sum(), (v*Bu).sum(), rtol=3e-12, atol=3e-14)
    div = torch.einsum('qaj,eaj->eq', op.dN, u[mesh.velocity_cells])
    torch.testing.assert_close((u*Bu).sum(), torch.einsum('eq,eq,q->',div,div,op.weights), rtol=3e-12, atol=3e-14)
    diagonal=torch.stack([csr._grad_div_blocks[3*i+i].to_dense().diagonal() for i in range(3)],dim=1)
    torch.testing.assert_close(diagonal, op.grad_div_diagonal(), rtol=3e-12, atol=3e-14)
    only_x=torch.zeros_like(u); only_x[:,0]=u[:,0]*u[:,1]
    assert csr.velocity_grad_div(only_x)[:,1:].abs().max() > .001
    # Affine divergence-free field: grad-div vanishes even before BC elimination.
    X=mesh.velocity_coordinates
    free_div=torch.stack((X[:,0],-X[:,1],torch.zeros_like(X[:,0])),dim=1)
    torch.testing.assert_close(csr.velocity_grad_div(free_div), torch.zeros_like(u), rtol=0, atol=2e-13)
    pattern=csr.matrices['velocity_mass']
    blocks=csr._grad_div_blocks
    assert len(blocks)==9 and all(b.layout==torch.sparse_csr and b.device==u.device for b in blocks)
    assert all(b.col_indices().data_ptr()==pattern.col_indices().data_ptr() for b in blocks)
    csr.prepare_grad_div()
    assert csr._grad_div_blocks is blocks  # No repeated assembly during stepping.


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('backend', ['quadrature','csr'])
def test_stabilized_tentative_equation_matches_independent_dense_solve_with_boundary_lift(device,backend):
    # Small enough to construct an independent dense oracle, including all cross blocks.
    mesh=create_box((2,2,2),(1.3,1.1,.9),device=device)
    op=prepare_operators(mesh)
    nv=len(mesh.velocity_coordinates)
    A=torch.zeros((3*nv,3*nv),device=device,dtype=op.N.dtype)
    local_scalar=(1.4/.003*torch.einsum('qa,qb,q->ab',op.N,op.N,op.weights)
                  +.2*torch.einsum('qai,qbi,q->ab',op.dN,op.dN,op.weights))
    local=torch.einsum('ab,ij->aibj',local_scalar,torch.eye(3,device=device,dtype=op.N.dtype))
    local=local+.7*torch.einsum('qai,qbj,q->aibj',op.dN,op.dN,op.weights)
    ids=(3*mesh.velocity_cells[:,:,None]+torch.arange(3,device=device)).reshape(-1,81)
    for cell in ids:
        A.index_put_((cell[:,None].expand(81,81),cell[None,:].expand(81,81)),local.reshape(81,81),accumulate=True)
    options=SolverOptions(rtol=1e-12,atol=1e-13,max_iterations=2000,check_every=2)
    flow=ChorinSolver(CSRFluidOperators(op) if backend=='csr' else op,dt=.003,rho=1.4,mu=.2,
                     grad_div_gamma=.7,options=options)
    X=mesh.velocity_coordinates
    boundary=torch.zeros_like(X);boundary[:,0]=.1*X[:,1]
    velocity=boundary.clone();density=torch.sin(2.7*X)
    result=flow.step(velocity,density=density,boundary_values=boundary)
    rhs=1.4/.003*op.velocity_mass(velocity)-1.4*op.convection(velocity)+op.density_load(density)
    fixed=flow.velocity_fixed.reshape(-1);free=~fixed
    lift=torch.where(fixed,boundary.reshape(-1),0.)
    expected=lift.clone()
    expected[free]=torch.linalg.solve(A[free][:,free],(rhs.reshape(-1)-A@lift)[free])
    torch.testing.assert_close(result.tentative_velocity.reshape(-1),expected,rtol=3e-10,atol=3e-12)
    assert result.diagnostics['grad_div_tentative_dissipation'] >= 0
    assert all(s['residual_norm']<=s['tolerance'] for s in result.diagnostics['solves'].values())


@pytest.mark.parametrize('device', DEVICES)
def test_zero_coefficient_reuses_original_action_and_three_step_solution(device):
    mesh=create_box((2,2,2),(1.,)*3,device=device)
    op=CSRFluidOperators(prepare_operators(mesh))
    default=ChorinSolver(op,dt=.001)
    zero=ChorinSolver(op,dt=.001,grad_div_gamma=0.)
    X=mesh.velocity_coordinates
    u=torch.zeros_like(X);density=torch.sin(X)
    a=default.step(u,density=density);b=zero.step(u,density=density)
    # CUDA reduction orders may differ by roundoff between independent solves.
    for name in ('tentative_velocity','velocity','pressure'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=2e-11,atol=2e-13)
    assert '_grad_div_blocks' not in op.__dict__
    assert 'grad_div_gamma' not in a.diagnostics


@pytest.mark.parametrize('device', DEVICES)
def test_grad_div_penalty_reduces_tentative_divergence_for_fixed_load(device):
    mesh=create_box((3,2,2),(1.,1.2,.9),device=device)
    op=CSRFluidOperators(prepare_operators(mesh))
    u=torch.zeros_like(mesh.velocity_coordinates)
    results=[ChorinSolver(op,dt=.02,mu=.1,grad_div_gamma=gamma).step(
        u,density=mesh.velocity_coordinates) for gamma in (0.,1.,10.)]
    norms=[r.diagnostics['tentative_divergence_l2'] for r in results]
    assert norms[2]<norms[1]<norms[0]


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),True,'one'])
def test_invalid_stabilization_fails_before_assembly(value):
    from afsi_torch.config import FEMFluidConfig
    with pytest.raises(ValueError,match='grad_div_alpha'):
        FEMFluidConfig(grad_div_alpha=value)
    op=prepare_operators(create_box((2,)*3,(1.,)*3))
    with pytest.raises(ValueError,match='grad_div_gamma'):
        ChorinSolver(op,dt=.001,grad_div_gamma=value)
