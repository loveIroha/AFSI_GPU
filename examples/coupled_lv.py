"""First generated-LV IB/FEM time steps; a short numerical test, not a cycle."""
import argparse
from dataclasses import asdict
import json
import csv
from pathlib import Path
import numpy as np
import torch
from afsi_torch.geometry import LVConfig, generate_lv
from afsi_torch.geometry.output import write_lv
from afsi_torch.lv_model import LVSolid, RampLoads
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.coupling_output import CoupledWriter
from afsi_torch.units import CGS_UNITS


def run(device='cpu', steps=10, dt=1e-4, fluid_cells=6, mesh_size=1.2, output='results/coupled_lv',
        output_every=5):
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    if steps < 1 or output_every < 1:
        raise ValueError('positive steps and output_every required')
    config = LVConfig(mesh_size=mesh_size)
    mesh = generate_lv(config, device=device)
    model = LVSolid(mesh, loads=RampLoads())
    fluid_mesh = create_box((fluid_cells,)*3, (12.,)*3, (-6., -6., -8.), device=device)
    fluid = ChorinSolver(prepare_operators(fluid_mesh), dt=dt, rho=1., mu=1.)
    coupling = ExplicitIBStepper(fluid, model.force, model.validate)
    state = coupling.initialize(mesh.X)
    initial = dict(step=0, time_s=0., **model.diagnostics(state.x))
    writer = None
    if output:
        write_lv(Path(output)/'geometry', mesh, model.fibers)
        writer = CoupledWriter(output, mesh, fluid_mesh)
        writer.write(state)
    history = []
    for _ in range(steps):
        result = coupling.step(state)
        state = result.state
        history.append(dict(step=state.step, **result.diagnostics, **model.diagnostics(state.x)))
        if writer and (state.step % output_every == 0 or state.step == steps):
            writer.write(state)
    report = dict(status='passed', device=str(device), torch=torch.__version__, units=CGS_UNITS,
        solid_config=asdict(config), loads=asdict(model.loads), material=asdict(model.parameters), beta=model.beta,
        fluid_counts=fluid_mesh.counts, fluid_lengths_cm=fluid_mesh.lengths, fluid_origin_cm=fluid_mesh.origin,
        dt_s=dt, steps=steps, simulated_time_s=state.time, solid_nodes=len(state.x), solid_cells=len(mesh.cells),
        fluid_velocity_nodes=len(state.velocity), fluid_pressure_nodes=len(state.pressure),
        force_path='IB density followed by consistent fluid mass',
        ordering='zero bootstrap; flow; x += dt H(x_old) u_new; force(x_new,t_old)',
        initial=initial, history=history, coupled_time_stepping=True, physiological_cycle=False,
        convergence_study_completed=False)
    if output:
        path = Path(output)
        (path/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        keys = ['step', 'time_s', 'cavity_volume_ml', 'wall_volume_cm3', 'minimum_detF',
                'maximum_detF', 'passive_energy_erg', 'spring_energy_erg', 'max_total_displacement_cm']
        with (path/'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer_csv = csv.DictWriter(stream, fieldnames=keys)
            writer_csv.writeheader()
            writer_csv.writerows([{key: h[key] for key in keys} for h in [initial, *history]])
        np.savez_compressed(path/'final_state.npz', **{name: getattr(state, name).cpu().numpy()
            for name in ('x', 'velocity', 'pressure', 'force')}, time=state.time, step=state.step,
            force_time=state.force_time, metadata=json.dumps(dict(dt=dt, units=CGS_UNITS)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--dt', type=float, default=1e-4)
    parser.add_argument('--fluid-cells', type=int, default=6)
    parser.add_argument('--mesh-size', type=float, default=1.2)
    parser.add_argument('--output', default='results/coupled_lv')
    parser.add_argument('--output-every', type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.device, args.steps, args.dt, args.fluid_cells, args.mesh_size,
                         args.output, args.output_every), indent=2))
