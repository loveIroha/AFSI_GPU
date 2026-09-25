"""Controlled LV time/mesh/duration study, with persistent partial failures.

This is a numerical investigation, not a cardiac cycle or convergence proof.
Exit 1 for failed runs; a completed study can still require refinement.
"""
import argparse
import csv
from dataclasses import dataclass, asdict
import hashlib
import json
from math import isfinite, log
from pathlib import Path
import platform
import time
import numpy as np
import torch
from afsi_torch.geometry import LVConfig, generate_lv
from afsi_torch.lv_model import LVSolid, RampLoads
from afsi_torch.fluid import create_box, prepare_operators, ChorinSolver
from afsi_torch.coupling import ExplicitIBStepper
from afsi_torch.units import CGS_UNITS


@dataclass(frozen=True)
class StudyCase:
    name: str
    dt: float = 1e-4
    final_time: float = .001
    fluid_cells: int = 6
    mesh_size: float = 1.2

    def __post_init__(self):
        if not self.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in self.name):
            raise ValueError('case name must be a nonempty safe filename')
        if any(not isfinite(v) or v <= 0 for v in (self.dt, self.final_time, self.mesh_size)):
            raise ValueError('positive finite dt, final time and mesh size required')
        if not isinstance(self.fluid_cells, int) or isinstance(self.fluid_cells, bool) or self.fluid_cells < 2:
            raise ValueError('fluid_cells must be an integer >= 2')
        if self.steps < 1 or abs(self.steps*self.dt-self.final_time) > 1e-12*self.final_time:
            raise ValueError('final_time must be an integer multiple of dt')

    @property
    def steps(self):
        return round(self.final_time/self.dt)


def study_cases(profile='standard'):
    if profile not in ('standard', 'smoke'):
        raise ValueError('unknown study profile')
    T = .001 if profile == 'standard' else .0004
    cases = [StudyCase('baseline', final_time=T), StudyCase('time_half', dt=5e-5, final_time=T),
             StudyCase('time_quarter', dt=2.5e-5, final_time=T)]
    if profile == 'standard':
        cases += [StudyCase('fluid_8', fluid_cells=8), StudyCase('fluid_10', fluid_cells=10),
                  StudyCase('solid_coarse', mesh_size=1.4), StudyCase('solid_fine', mesh_size=.9),
                  StudyCase('fluid_8_small_dt', fluid_cells=8, dt=2.5e-5),
                  StudyCase('fluid_10_small_dt', fluid_cells=10, dt=2.5e-5),
                  StudyCase('duration_005', final_time=.005), StudyCase('duration_010', final_time=.01)]
    return cases


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reject NaN/Infinity rather than silently writing non-standard JSON.
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


class StudyRunner:
    def __init__(self, device='cpu', output='results/lv_study'):
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
        self.device, self.output = str(device), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.models = {}

    def model(self, mesh_size):
        # Time/fluid comparisons share exactly the same reference coordinates,
        # cells and fibers; no accidental Gmsh remeshing between dt levels.
        if mesh_size not in self.models:
            mesh = generate_lv(LVConfig(mesh_size=mesh_size), device=self.device)
            self.models[mesh_size] = LVSolid(mesh, loads=RampLoads())
        return self.models[mesh_size]

    @torch.no_grad()
    def run_case(self, case):
        start = time.perf_counter()
        history, failure, state, model = [], None, None, None
        phase, attempted_step = 'setup', 0
        report = dict(case=asdict(case), requested_steps=case.steps, device=self.device,
                      completed=False, initial=None, history=history)
        try:
            model = self.model(case.mesh_size)
            mesh = create_box((case.fluid_cells,)*3, (12.,)*3, (-6., -6., -8.), device=self.device)
            flow = ChorinSolver(prepare_operators(mesh), dt=case.dt, rho=1., mu=1.)
            driver = ExplicitIBStepper(flow, model.force, model.validate)
            state = driver.initialize(model.mesh.X)
            report['initial'] = model.diagnostics(state.x)
            report['solid_nodes'] = len(state.x)
            report['solid_cells'] = len(model.mesh.cells)
            report['gmsh'] = model.mesh.gmsh_version
            report['physical_parameters'] = dict(material=asdict(model.parameters), beta=model.beta,
                rho=flow.rho, mu=flow.mu, solver_options=asdict(flow.options))
            phase = 'time_step'
            for attempted_step in range(1, case.steps+1):
                result = driver.step(state)
                diagnostic = model.diagnostics(result.state.x)
                history.append(dict(step=attempted_step, **result.diagnostics, **diagnostic,
                    advective_courant=(case.dt*(result.state.velocity.abs()/
                        result.state.velocity.new_tensor(mesh.cell_sizes)).sum(-1).max()).item()))
                state = result.state
            report['completed'] = True
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            # Preserve actual failures; never lower loads, enlarge tolerances,
            # reduce dt or silently continue the same trajectory after failure.
            failure = dict(type=type(exc).__name__, message=str(exc), phase=phase, attempted_step=attempted_step)
        report.update(failure=failure, accepted_steps=0 if state is None else state.step,
                      reached_time_s=0. if state is None else state.time,
                      wall_seconds=time.perf_counter()-start)
        folder = self.output/case.name
        folder.mkdir(parents=True, exist_ok=True)
        if state is not None:
            report['final'] = model.diagnostics(state.x)
            np.savez_compressed(folder/'last_accepted.npz', X=model.mesh.X.cpu().numpy(),
                cells=model.mesh.cells.cpu().numpy(), x=state.x.cpu().numpy(),
                velocity=state.velocity.cpu().numpy(), pressure=state.pressure.cpu().numpy(),
                force=state.force.cpu().numpy(), time=state.time, step=state.step)
        else:
            # Reusing an output directory must not leave a previous successful
            # run's checkpoint beside a new failed setup report.
            (folder/'last_accepted.npz').unlink(missing_ok=True)
        report['summary'] = summarize(report)
        write_json(folder/'report.json', report)
        if history:
            keys = ['step', 'time_s', 'cavity_volume_ml', 'wall_volume_cm3', 'minimum_detF',
                'maximum_detF', 'max_total_displacement_cm', 'passive_energy_erg', 'spring_energy_erg',
                'advective_courant', 'fe_minus_solid_power', 'lattice_power_error']
            with (folder/'history.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows({k: row[k] for k in keys} for row in history)
        else:
            (folder/'history.csv').unlink(missing_ok=True)
        return report


def summarize(report):
    rows = report['history']
    if not rows:
        return {}
    initial, final = report['initial'], report['final']
    return dict(delta_cavity_ml=final['cavity_volume_ml']-initial['cavity_volume_ml'],
        relative_wall_volume_change=final['wall_volume_cm3']/initial['wall_volume_cm3']-1,
        maximum_abs_relative_wall_volume_change=max(abs(r['wall_volume_cm3']/initial['wall_volume_cm3']-1) for r in rows),
        min_detF=min(r['minimum_detF'] for r in rows), max_detF=max(r['maximum_detF'] for r in rows),
        max_displacement_cm=final['max_total_displacement_cm'],
        max_divergence_l2=max(r['fluid']['corrected_divergence_l2'] for r in rows),
        max_net_flux_abs=max(abs(r['fluid']['net_flux']) for r in rows),
        max_courant=max(r['advective_courant'] for r in rows),
        max_solver_residual_ratio=max(s['residual_norm']/s['tolerance'] for r in rows for s in r['fluid']['solves'].values()),
        max_pressure_iterations=max(r['fluid']['solves']['pressure']['iterations'] for r in rows),
        max_lattice_power_error=max(r['lattice_power_error'] for r in rows),
        accumulated_fe_minus_solid_work_erg=report['case']['dt']*sum(r['fe_minus_solid_power'] for r in rows),
        accumulated_absolute_power_gap_erg=report['case']['dt']*sum(abs(r['fe_minus_solid_power']) for r in rows))


def relative_difference(a, b, floor):
    """Relative to the RESPONSE, not the large initial cavity volume."""
    return None if abs(b) <= floor else abs(a-b)/abs(b)


def compare_pair(coarse, fine, folder, axis):
    a, b = coarse['case'], fine['case']
    if not coarse['completed'] or not fine['completed']:
        return dict(coarse=a['name'], fine=b['name'], available=False, reason='incomplete run')
    if a['final_time'] != b['final_time'] or coarse['reached_time_s'] != fine['reached_time_s']:
        raise ValueError('refinement comparisons require the same final physical time')
    permitted = {'time': 'dt', 'fluid': 'fluid_cells', 'solid': 'mesh_size'}
    if axis not in permitted:
        raise ValueError('unknown comparison axis')
    changed = permitted[axis]
    if any(a[k] != b[k] for k in ('dt', 'fluid_cells', 'mesh_size') if k != changed):
        raise ValueError('change only one refinement parameter at a time')
    sa, sb = coarse['summary'], fine['summary']
    result = dict(coarse=a['name'], fine=b['name'], available=True,
        delta_cavity_difference_ml=abs(sa['delta_cavity_ml']-sb['delta_cavity_ml']),
        delta_cavity_relative_difference=relative_difference(sa['delta_cavity_ml'], sb['delta_cavity_ml'], 1e-10),
        max_displacement_relative_difference=relative_difference(sa['max_displacement_cm'], sb['max_displacement_cm'], 1e-12))
    if axis != 'solid':
        with np.load(Path(folder)/a['name']/'last_accepted.npz', allow_pickle=False) as da, np.load(
                Path(folder)/b['name']/'last_accepted.npz', allow_pickle=False) as db:
            if not np.array_equal(da['X'], db['X']) or not np.array_equal(da['cells'], db['cells']):
                raise ValueError('nodal comparison requires identical reference mesh and numbering')
            ua, ub = da['x']-da['X'], db['x']-db['X']
            error, response = float(np.linalg.norm(ua-ub)), float(np.linalg.norm(ub))
            result['displacement_nodal_l2_difference_cm'] = error
            result['displacement_nodal_l2_relative_difference'] = None if response <= 1e-12 else error/response
    return result


def assess(reports, folder, profile, tolerance=.05):
    """A declared 5% response-change SCREEN, never a full-cycle acceptance."""
    if not isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('screen tolerance must lie between 0 and 1')
    groups = dict(time=('time', ['baseline', 'time_half', 'time_quarter']))
    if profile == 'standard':
        groups.update(fluid=('fluid', ['baseline', 'fluid_8', 'fluid_10']),
                      solid=('solid', ['solid_coarse', 'baseline', 'solid_fine']),
                      fluid_small_dt=('fluid', ['time_quarter', 'fluid_8_small_dt', 'fluid_10_small_dt']))
    output = {}
    for label, (axis, names) in groups.items():
        pairs = [compare_pair(reports[a], reports[b], folder, axis) for a, b in zip(names, names[1:])]
        last = pairs[-1]
        keys = ['delta_cavity_relative_difference', 'max_displacement_relative_difference']
        if axis != 'solid':
            keys.append('displacement_nodal_l2_relative_difference')
        metrics = [last.get(k) for k in keys]
        status = ('inconclusive' if not last['available'] or any(v is None for v in metrics) else
                  'within_screen' if max(metrics) <= tolerance else 'needs_refinement')
        output[label] = dict(screen=status, finest_pair_relative_metrics=dict(zip(keys, metrics)), pairs=pairs)
        if axis == 'time' and all(p.get('available') for p in pairs):
            d0, d1 = [p['displacement_nodal_l2_difference_cm'] for p in pairs]
            dts = [reports[name]['case']['dt'] for name in names]
            ratios = [dts[0]/dts[1], dts[1]/dts[2]]
            uniform = ratios[0] > 1 and abs(ratios[0]-ratios[1]) < 1e-12
            output[label]['observed_order_nodal_displacement'] = (
                log(d0/d1)/log(ratios[0]) if uniform and min(d0, d1) > 1e-14 else None)
    return dict(relative_response_screen_tolerance=tolerance, axes=output,
        all_requested_runs_completed=all(r['completed'] for r in reports.values()),
        all_refinement_screens_met=all(r['screen'] == 'within_screen' for r in output.values()),
        full_cycle_ready=False,
        limitations=['short prescribed-load ramp, no preloaded state or valve/circulation model',
                     'wall-volume drift is not by itself a cavity leakage measure',
                     'fluid refinement also changes Peskin support width',
                     'separate refinements are sensitivity screens, not joint/asymptotic convergence proof'])


def run_study(device='cpu', output='results/lv_study', profile='standard'):
    runner = StudyRunner(device, output)
    reports = {}
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), *sorted((root/'src'/'afsi_torch').rglob('*.py'))]
    fingerprint = hashlib.sha256()
    for p in files:
        fingerprint.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes())
    metadata = dict(profile=profile, device=str(device), torch=torch.__version__, python=platform.python_version(),
        numpy=np.__version__, units=CGS_UNITS, source_sha256=fingerprint.hexdigest(),
        gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
        loads=asdict(RampLoads()), fluid_domain=dict(lengths_cm=[12.]*3, origin_cm=[-6., -6., -8.]),
        physics_changed=False, notes='CPU timings include setup/output and are not GPU performance claims')
    write_json(Path(output)/'study.json', dict(metadata=metadata, cases={}, assessment=None))
    for case in study_cases(profile):
        print(f'Running {case.name}: dt={case.dt:g}, T={case.final_time:g}, fluid={case.fluid_cells}, solid_h={case.mesh_size:g}', flush=True)
        reports[case.name] = runner.run_case(case)
        print(f"  completed={reports[case.name]['completed']}, accepted_steps={reports[case.name]['accepted_steps']}", flush=True)
        write_json(Path(output)/'study.json', dict(metadata=metadata, cases=reports, assessment=None))
    assessment = assess(reports, output, profile)
    report = dict(metadata=metadata, cases=reports, assessment=assessment)
    write_json(Path(output)/'study.json', report)
    columns = ['case', 'completed', 'reached_time_s', 'dt', 'fluid_cells', 'mesh_size', 'delta_cavity_ml',
               'max_displacement_cm', 'min_detF', 'max_divergence_l2', 'max_pressure_iterations', 'wall_seconds']
    with (Path(output)/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for name, r in reports.items():
            writer.writerow(dict(r['case'], **r['summary'], case=name, completed=r['completed'],
                reached_time_s=r['reached_time_s'], wall_seconds=r['wall_seconds']))
    return report


def compact_report(report):
    """Auditable CI/log summary; full histories and arrays stay in artifacts."""
    keys = ('case', 'completed', 'accepted_steps', 'reached_time_s', 'failure', 'initial', 'final',
            'summary', 'solid_nodes', 'solid_cells', 'gmsh', 'physical_parameters')
    return dict(metadata=report['metadata'], assessment=report['assessment'],
                cases={name: {k: r[k] for k in keys if k in r} for name, r in report['cases'].items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='results/lv_study')
    parser.add_argument('--profile', choices=['standard', 'smoke'], default='standard')
    args = parser.parse_args()
    result = run_study(args.device, args.output, args.profile)
    print(json.dumps(compact_report(result), indent=2, allow_nan=False))
    raise SystemExit(0 if result['assessment']['all_requested_runs_completed'] else 1)
