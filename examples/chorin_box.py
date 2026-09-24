"""Short forced-flow smoke test, not a coupled LV simulation or accuracy study."""
import argparse
import json
from pathlib import Path
import torch
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver


def run(device='cpu', cells=4, steps=5):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if cells < 2 or steps < 1:
        raise ValueError('cells>=2 and steps>=1 required')
    mesh = create_box((cells,)*3, device=device)
    op = prepare_operators(mesh)
    solver = ChorinSolver(op, dt=.01, rho=1., mu=.1)
    X = mesh.velocity_coordinates
    bubble = (4*X*(1-X)).prod(-1)
    density = bubble[:, None]*X.new_tensor([1., .3, -.2])
    u, p = torch.zeros_like(X), None
    history = []
    for step in range(steps):
        result = solver.step(u, density=density, pressure_initial=p)
        u, p = result.velocity, result.pressure
        assert torch.isfinite(u).all() and torch.isfinite(p).all()
        torch.testing.assert_close(u[mesh.velocity_boundary], torch.zeros_like(u[mesh.velocity_boundary]), atol=0, rtol=0)
        history.append(dict(step=step+1, time_s=(step+1)*solver.dt, **result.diagnostics))
    return dict(status='passed', device=str(device), torch=torch.__version__, cells=cells**3,
                velocity_nodes=len(X), pressure_nodes=len(p), dt_s=solver.dt,
                length_cm=1., rho_g_per_cm3=solver.rho, mu_g_per_cm_s=solver.mu,
                max_velocity_cm_per_s=u.abs().max().item(), history=history,
                preconditioner='Jacobi', coupled_to_solid=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cells', type=int, default=4, help='cells per axis')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--output')
    args = parser.parse_args()
    result = run(args.device, args.cells, args.steps)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
