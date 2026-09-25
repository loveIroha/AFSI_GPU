"""Newton/GMRES convergence, boundary lifting, invalid-trial rejection and FEM."""
import pytest
import torch
from afsi_torch.nonlinear import gmres,newton,GMRESOptions,NewtonOptions,NonlinearFailure
from afsi_torch.solid_preconditioner import guccione_blocks
from afsi_torch import solid,boundary as bd
from afsi_torch.tetrahedron import reference_nodes
from afsi_torch.fields import prepare_reference_fields
from afsi_torch.materials import GuccioneParameters
from examples.nonlinear_patch import run,setup


@pytest.fixture(params=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA device unavailable'))])
def device(request):
    return request.param


def test_gmres_nonsymmetric_indefinite_and_zero_rhs(device):
    A = torch.tensor([[2.,4.,1.],[0.,-3.,2.],[1.,0.,1.]],device=device,dtype=torch.float64)
    b = A.new_tensor([1.,-2.,3.])
    inverse = lambda v:v/A.diagonal()
    x,info = gmres(lambda v:A@v,b,precondition=inverse,
                  options=GMRESOptions(rtol=1e-12,atol=1e-13,check_every=1))
    torch.testing.assert_close(x,torch.linalg.solve(A,b),atol=1e-11,rtol=1e-11)
    assert info['residual_norm'] <= info['tolerance']
    zero,info = gmres(lambda v:A@v,torch.zeros_like(b))
    assert info['iterations']==0 and zero.count_nonzero()==0
    with pytest.raises(RuntimeError,match='singular|breakdown'):
        gmres(lambda v:0*v,b)


def test_gmres_restart_and_nonconvergence(device):
    A = torch.diag(torch.linspace(1,2,12,device=device,dtype=torch.float64))
    A += torch.diag(torch.full((11,),.1,device=device,dtype=A.dtype),1)
    rhs = torch.arange(12,device=device,dtype=A.dtype)+1
    x,info = gmres(lambda v:A@v,rhs,options=GMRESOptions(restart=3,rtol=1e-10,check_every=2))
    assert info['iterations']>3
    torch.testing.assert_close(A@x,rhs,atol=1e-8,rtol=1e-9)
    with pytest.raises(RuntimeError,match='failed'):
        gmres(lambda v:A@v,rhs,options=GMRESOptions(max_iterations=1,restart=1,rtol=1e-14))


def test_newton_rejects_invalid_trials_and_does_not_mutate(device):
    x = torch.tensor([10.],device=device,dtype=torch.float64)
    seen=[]
    def validate(y):
        seen.append(y.item())
        if (y<=0).any():
            raise ValueError('invalid domain')
    result = newton(torch.log,x,validate=validate)
    torch.testing.assert_close(result.x,torch.ones_like(x),atol=1e-8,rtol=0)
    assert x.item()==10 and min(seen)<0
    assert any(r.get('backtracks',0)>0 for r in result.history)
    assert all(b['residual_norm']<a['residual_norm'] for a,b in zip(result.history,result.history[1:]))


def test_newton_dirichlet_lift_and_failure_state(device):
    x = torch.tensor([.1,1.],device=device,dtype=torch.float64)
    fixed = torch.tensor([True,False],device=device)
    values = x.new_tensor([1.5,0.])
    residual = lambda y: torch.stack((y[0]**2+y[1]-5,y[1]**3-8))
    result = newton(residual,x,validate=lambda y:None,fixed=fixed,values=values)
    torch.testing.assert_close(result.x,x.new_tensor([1.5,2.]),atol=1e-8,rtol=0)
    assert x[0].item()==.1
    with pytest.raises(NonlinearFailure) as err:
        newton(residual,x,validate=lambda y:None,fixed=fixed,values=values,
               options=NewtonOptions(max_iterations=1,rtol=1e-14,atol=1e-14))
    assert not err.value.result.converged
    assert err.value.result.x[0].item()==1.5 and len(err.value.result.history)==2
    with pytest.raises(NonlinearFailure,match='linear') as err:
        newton(lambda y:torch.ones_like(y),x,validate=lambda y:None)
    torch.testing.assert_close(err.value.result.x,x)


def test_nodal_blocks_match_dense_diagonal(device):
    X = reference_nodes(device=device)
    geometry = solid.prepare_p2(X,torch.arange(10,device=device).reshape(1,10))
    base = bd.prepare_surface(X,bd.extract_boundary(X,geometry.cells))
    fields = prepare_reference_fields(geometry,X.new_tensor([1.,0.,0.]).expand_as(X),
                                      X.new_tensor([0.,1.,0.]).expand_as(X),123.)
    parameters = GuccioneParameters()
    x = X@X.new_tensor([[1.03,.02,0.],[0.,.98,.01],[0.,0.,1.01]]).T
    residual = lambda y: -solid.guccione_force(y,geometry,fields,parameters)-bd.spring_force(y,base,500.)
    J = torch.func.jacrev(residual)(x)
    diagonal = torch.stack([J[i,:,i,:] for i in range(len(x))])
    blocks = guccione_blocks(x,geometry,fields,parameters,base=base,beta=500.)
    torch.testing.assert_close(blocks,diagonal,atol=1e-7,rtol=1e-11)


def test_p2_analytic_and_follower_equilibrium(device,tmp_path):
    report = run(device,tmp_path)
    assert report['converged'] and report['cases']['affine']['analytic_position_max_error_cm']<1e-8
    for case in report['cases'].values():
        assert case['iterations']>1 and case['residual_norm']<=case['tolerance']
        assert case['minimum_detF']>0


def test_follower_against_dense_newton(device):
    X,cells,geo,surface,loaded,fixed,fields = setup(device)
    params = GuccioneParameters()
    residual = lambda y: -solid.guccione_force(y,geo,fields,params)-bd.pressure_force(y,loaded,1000.)
    free = (~fixed).reshape(-1).nonzero().flatten()
    x = X.clone()
    for _ in range(10):
        r = residual(x).reshape(-1)[free]
        if torch.linalg.vector_norm(r).item()<1e-8:
            break
        J = torch.func.jacrev(residual)(x).reshape(X.numel(),X.numel())[free][:,free]
        delta = torch.linalg.solve(J,-r)
        flat = x.reshape(-1).clone()
        flat[free] += delta
        x = flat.reshape_as(X)
        solid.validate_deformation(x,geo)
    assert torch.linalg.vector_norm(residual(x).reshape(-1)[free])<1e-8
    result = newton(residual,X,validate=lambda y:solid.validate_deformation(y,geo),fixed=fixed,
                    options=NewtonOptions(atol=1e-8,rtol=1e-10,linear=GMRESOptions(restart=70,rtol=1e-5)))
    torch.testing.assert_close(result.x,x,atol=2e-9,rtol=1e-9)


@pytest.mark.parametrize('factory,kwargs',[(GMRESOptions,dict(restart=0)),(GMRESOptions,dict(rtol=-1)),
    (NewtonOptions,dict(armijo=1)),(NewtonOptions,dict(linear=GMRESOptions(rtol=1.)))])
def test_invalid_controls(factory,kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)
