"""Small transferable P1 material/boundary example using the shared MAC driver.

No LV configuration, patient mesh, fibers or cavity assumptions are required.
Units: cm-g-s. This manufactured block is an API example, not a cardiac case.
"""
import argparse
from dataclasses import asdict
from types import SimpleNamespace
import torch
from afsi_torch.solids import P1Solid, BoundaryForce
from afsi_torch.mac.execution import build_driver
from afsi_torch.mac.implicit import MACCouplingOptions


def material(F, fields, time):
    """Replace this single-element PK1 function to supply another material."""
    mu, lame = fields
    I = torch.eye(3, device=F.device, dtype=F.dtype)
    E = .5*(F.T@F-I)
    return F@(2*mu*E+lame*torch.trace(E)*I)


def boundary_force(nodes, fields, time):
    """Integrated local nodal force; a spring and prescribed nodal load here."""
    reference, spring, load = fields
    return -spring*(nodes-reference)+load*(1+time)


def run(*, device='cpu', steps=3, dt=1e-4, fluid_cells=8, shear=5000., lame=1000., spring=100.):
    if type(steps) is not int or steps < 1:
        raise ValueError('positive integer steps required')
    X = torch.tensor([[5.,5.,5.],[5.3,5.,5.],[5.,5.3,5.],[5.,5.,5.3]],
                     device=device, dtype=torch.float64)
    cells = torch.tensor([[0,1,2,3]], device=device, dtype=torch.int64)
    mesh = SimpleNamespace(X=X, cells=cells)
    faces = cells[:,:3]
    boundary = BoundaryForce(faces, boundary_force, (
        X[faces].clone(), X.new_full((1,1,1),spring), X.new_tensor([[[1200.,100.,0.]]])))
    model = P1Solid(mesh, material, cell_fields=(X.new_full((1,),shear),X.new_full((1,),lame)),
                    boundaries=(boundary,))
    # Ordinary settings, shared by every 3D MAC solid adapter. A different demo
    # may instead build these from the existing serializable configuration API.
    settings = dict(dt=dt,fluid_shape=(fluid_cells,)*3,fluid_lengths=(15.,)*3,
                    fluid_origin=(0.,)*3,rho=1.,mu=1.,interaction_degree=2,
                    coupling=asdict(MACCouplingOptions(scheme='cnab-semiimplicit',
                                                       semiimplicit_solver='anderson-newton')),
                    execution_backend='fused',solid_backend='reference',mass_backend='pcg',
                    coupling_backend='optimized',pressure_backend='torch',warm_start=True)
    driver = build_driver(model,settings,device)
    state = driver.initialize(X)
    for _ in range(steps):
        state, info = driver.step(state)
    return dict(steps=state.step,time_s=state.time,**model.diagnostics(state.x),
                residual=info['nonlinear']['residual_norm'],tolerance=info['nonlinear']['tolerance'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--dt',type=float,default=1e-4)
    p.add_argument('--fluid-cells',type=int,default=8)
    p.add_argument('--shear',type=float,default=5000.)
    p.add_argument('--lame',type=float,default=1000.)
    p.add_argument('--spring',type=float,default=100.)
    args=p.parse_args()
    import json
    print(json.dumps(run(**vars(args)),indent=2))


if __name__ == '__main__':
    main()
