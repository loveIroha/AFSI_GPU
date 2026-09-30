"""Short, read-only checkpoint replay to locate MAC/IB runtime costs.

Compare the original expanded IB stencil with the separable implementation.
No checkpoints, histories or physical settings are changed. CUDA events are
read after a whole replay, with no extra synchronization between phases.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from afsi_torch.cycle_checkpoint import atomic_json
from afsi_torch.ib import peskin4
from afsi_torch.mac import MACGrid, MACFlow, divergence
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.mac.coupling import MACIBStepper
from afsi_torch.mac.transfer import FETransfer, MACStencil
from afsi_torch.solid import prepare_p2
from afsi_torch.mac.multigrid import MGOptions
from afsi_torch.fluid.solvers import SolverOptions


class ExpandedTransfer(FETransfer):
    """Frozen pre-optimization prepare() for the A/B comparison only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = torch.cartesian_prod(self.axis_offsets, self.axis_offsets, self.axis_offsets)

    @torch.no_grad()
    def prepare(self, x):
        points = self.interaction_points(x)
        self.check_support(points)
        indices, weights = [], []
        for c in range(3):
            scaled = (points-points.new_tensor(self.grid.face_origin(c)))/points.new_tensor(self.grid.spacing)
            base = torch.floor(scaled-1).to(torch.int64)
            nodes = base[:,None,:]+self.offsets[None,:,:]
            shape = self.grid.face_shape(c)
            if (nodes < 0).any() or (nodes >= nodes.new_tensor(shape)).any():
                raise ValueError('incomplete MAC interaction support')
            indices.append((nodes[...,0]*shape[1]+nodes[...,1])*shape[2]+nodes[...,2])
            weights.append(peskin4(scaled[:,None,:]-nodes.to(points.dtype)).prod(-1))
        return MACStencil(tuple(indices), tuple(weights))


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class PhaseRecorder:
    def __init__(self, device):
        self.device = device
        self.records = defaultdict(list)

    def wrap(self, function, label):
        @wraps(function)
        def recorded(*args, **kwargs):
            if self.device.type == 'cuda':
                stream = torch.cuda.current_stream(self.device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                result = function(*args, **kwargs)
                end.record(stream)
                self.records[label].append((start, end))
            else:
                start = perf_counter()
                result = function(*args, **kwargs)
                self.records[label].append(1000*(perf_counter()-start))
            return result
        return recorded

    def summary(self, steps):
        result = {}
        for name, entries in self.records.items():
            samples = ([a.elapsed_time(b) for a,b in entries]
                       if self.device.type == 'cuda' else entries)
            result[name] = dict(calls=len(samples), ms_per_call=mean(samples),
                                ms_per_step=sum(samples)/steps)
        return result


@contextmanager
def record_phases(driver, recorder):
    # pressure_solve is a nested subset of fluid_total; do not sum both.
    targets = [(driver, 'validate', 'solid_validation'),
               (driver.transfer, 'prepare', 'ib_prepare'),
               (driver.transfer, 'spread', 'ib_spread_including_mass_solve'),
               (driver.flow, 'step', 'fluid_total'),
               (driver.flow.pressure_solver, 'solve', 'pressure_solve'),
               (driver.transfer, 'interpolate', 'ib_interpolate_including_mass_solve'),
               (driver, 'force', 'solid_force')]
    originals = [(obj, name, getattr(obj, name)) for obj,name,_ in targets]
    try:
        for (obj,name,label), (_,_,function) in zip(targets, originals):
            setattr(obj, name, recorder.wrap(function, label))
        yield
    finally:
        for obj,name,function in originals:
            setattr(obj, name, function)


@torch.no_grad()
def replay(driver, initial, steps, device):
    state = initial
    cycles, force_iterations, velocity_iterations = [], [], []
    synchronize(device)
    started = perf_counter()
    for _ in range(steps):
        state, info = driver.step(state, diagnostics=False)
        cycles.append(info['flow']['pressure']['cycles'])
        force_iterations.append(info['force_mass']['iterations'])
        velocity_iterations.append(info['velocity_mass']['iterations'])
    synchronize(device)
    return state, dict(wall_seconds=perf_counter()-started,
        pressure_cycles_mean=mean(cycles), pressure_cycles_max=max(cycles),
        force_mass_iterations_mean=mean(force_iterations),
        velocity_mass_iterations_mean=mean(velocity_iterations))


def compare_states(reference, candidate):
    differences = {}
    for name in ('x', 'force', 'pressure'):
        a, b = getattr(reference, name), getattr(candidate, name)
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-8)
        differences[name+'_max_abs'] = (a-b).abs().max().item()
    for c,(a,b) in enumerate(zip(reference.velocity, candidate.velocity)):
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)
        differences[f'velocity_{c}_max_abs'] = (a-b).abs().max().item()
    return differences


@torch.no_grad()
def benchmark(checkpoint, *, device='cuda', steps=20, warmup=3):
    if type(steps) is not int or steps < 1 or type(warmup) is not int or warmup < 1:
        raise ValueError('steps and warmup must be positive integers')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    model, initial, settings, _ = load_mac(checkpoint, device)
    from afsi_torch.config import lv_grid
    grid = lv_grid(settings)
    degree = settings['interaction_degree']
    geometry = (model.geometry if degree is None else
                prepare_p2(model.mesh.X, model.mesh.cells, degree=degree))
    variants, final_states = {}, {}
    for name, cls, warm_start in [('expanded', ExpandedTransfer, False),
                                   ('separable', FETransfer, False),
                                   ('separable_warm', FETransfer, True)]:
        flow = MACFlow(grid, dt=settings['dt'], rho=settings['rho'], mu=settings['mu'], device=device,
                       options=MGOptions(**settings.get('pressure_solver',{})))
        transfer = cls(grid, geometry, warm_start=warm_start,
                       options=SolverOptions(**settings['mass_solver']) if 'mass_solver' in settings else None)
        driver = MACIBStepper(flow, transfer, model.force, model.validate)
        replay(driver, initial, warmup, device)
        transfer.reset_warm_start()
        # All replays start from the same checkpoint, including the warmup.
        # The uninstrumented pass is the throughput result.
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        live_bytes = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
        end, plain = replay(driver, initial, steps, device)
        peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        final_states[name] = end
        transfer.reset_warm_start()
        recorder = PhaseRecorder(device)
        with record_phases(driver, recorder):
            profiled, measured = replay(driver, initial, steps, device)
        repeat_difference = compare_states(end, profiled)
        phase_ms = recorder.summary(steps)
        # Physical checks happen after timing, at the same final state.
        end_stencil = transfer.prepare(end.x)
        density, _ = transfer.spread(end.force, end_stencil)
        solid_velocity, _ = transfer.interpolate(end.velocity, end_stencil)
        solid_power = (solid_velocity*end.force).sum().item()
        fluid_power = (grid.volume*sum((u*f).sum() for u,f in zip(end.velocity,density))).item()
        variants[name] = dict(**plain, ms_per_step=1000*plain['wall_seconds']/steps,
            instrumented_wall_seconds=measured['wall_seconds'],
            instrumentation_ratio=measured['wall_seconds']/plain['wall_seconds'],
            phase_timings=phase_ms, peak_allocated_bytes=peak,
            peak_extra_allocated_bytes=peak-live_bytes if peak is not None else None,
            replay_repeat_max_abs=repeat_difference,
            final_solid=model.diagnostics(end.x),
            final_divergence_l2=(grid.volume*divergence(end.velocity,grid.spacing).square().sum()).sqrt().item(),
            final_power_abs_error=abs(solid_power-fluid_power),
            final_power_relative_error=abs(solid_power-fluid_power)/max(abs(solid_power),abs(fluid_power),1e-30))
        print(f"{name}: {variants[name]['ms_per_step']:.3f} ms/step; "
              f"pressure cycles mean={plain['pressure_cycles_mean']:.1f}", flush=True)
        for phase, values in phase_ms.items():
            print(f"  {phase}: {values['ms_per_step']:.3f} ms/step", flush=True)
        del driver, transfer, flow, profiled, end_stencil, density, solid_velocity
    differences = compare_states(final_states['expanded'], final_states['separable'])
    warm_differences = compare_states(final_states['separable'], final_states['separable_warm'])
    return dict(schema=1, benchmark='mac-checkpoint-short-replay',
        checkpoint=str(Path(checkpoint).resolve()), device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        cpu_threads=torch.get_num_threads(), dtype=str(initial.x.dtype),
        start_step=initial.step, start_time_s=initial.time, end_time_s=final_states['separable'].time,
        steps=steps, warmup_steps=warmup, settings=settings,
        solid_nodes=len(initial.x), solid_cells=len(geometry.cells),
        interaction_points=geometry.weights.numel(),
        interaction_links=geometry.weights.numel()*3*64,
        variants=variants, equivalence_max_abs=differences,
        warm_start_equivalence_max_abs=warm_differences,
        measured_speedup=variants['expanded']['wall_seconds']/variants['separable']['wall_seconds'],
        warm_start_speedup=variants['separable']['wall_seconds']/variants['separable_warm']['wall_seconds'],
        notes=[
            'No input checkpoint or simulation output is modified.',
            'Setup, final diagnostics, logging and checkpoint I/O are excluded from throughput.',
            'Both variants and timed passes replay the identical checkpoint state.',
            'pressure_solve is included in fluid_total; all other listed phases are disjoint.',
            'Phase events measure stream elapsed time, including possible host-launch gaps, not kernel busy time.',
            'Remaining step time includes additional guards, support checks and Python dispatch.',
            'A final 2 s checkpoint samples only the held-load tail, not the full loading trajectory.',
            'separable_warm uses the previous mass-solve result as the initial guess; it retains the same true residual tolerance.',
            'Short replay speedup is not a measured full-horizon speedup.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--output', default='results/mac_performance/report.json')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'choose a new benchmark output: {output}')
    report = benchmark(args.checkpoint, device=args.device, steps=args.steps, warmup=args.warmup)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps(dict(report=str(output), measured_speedup=report['measured_speedup'],
                         warm_start_speedup=report['warm_start_speedup'],
                         equivalence_max_abs=report['equivalence_max_abs'],
                         warm_start_equivalence_max_abs=report['warm_start_equivalence_max_abs']), indent=2))


if __name__ == '__main__':
    main()
