"""Native demo_337: MPI ghost synchronization, offline tracking and fixed output."""
import ast
import csv
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback


def offline_tree(source, output_path, experiment, output_every=400):
    if type(output_every) is not int or output_every < 1:
        raise ValueError('output_every must be a positive integer')

    class Replace(ast.NodeTransformer):
        def __init__(self):
            self.counts = dict(output=0, counter=0, interpolation=0, spread=0, cadence=0, speed=0)

        def visit_Assign(self, node):
            if len(node.targets) == 1:
                target = node.targets[0]
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == 'config' and isinstance(target.slice, ast.Constant)
                        and isinstance(node.value, ast.IfExp)):
                    key = target.slice.value
                    if key in ('output_path', 'experiment_name'):
                        kind = 'output' if key == 'output_path' else 'counter'
                        self.counts[kind] += 1
                        node.value = ast.Constant(output_path if kind == 'output' else experiment)
                if isinstance(target, ast.Name) and target.id == 'u_max':
                    self.counts['speed'] += 1
                    node.value = ast.parse('global_component_max(ns_solver.u_, mesh.comm)', mode='eval').body
            return self.generic_visit(node)

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == 'ib_interpolation'):
                names = dict(fluid_to_solid=('interpolation', 'solid_velocity'),
                             solid_to_fluid=('spread', 'solid_force'))
                if call.func.attr in names:
                    key, name = names[call.func.attr]
                    self.counts[key] += 1
                    return [node, ast.copy_location(ast.parse(f'{name}.x.scatter_forward()').body[0], node)]
            return node

        def visit_If(self, node):
            if 'time_manager.should_output(step)' in ast.unparse(node.test):
                self.counts['cadence'] += 1
                node.test = ast.parse(
                    f'(step + 1) % {output_every} == 0 or step + 1 == config["num_steps"]', mode='eval').body
            return self.generic_visit(node)

    replace = Replace()
    tree = replace.visit(ast.parse(source))
    expected = dict(output=1, counter=1, interpolation=1, spread=1, cadence=1, speed=1)
    if replace.counts != expected:
        raise ValueError(f'unexpected demo_337 layout: {replace.counts}; refusing ambiguous rewrite')
    return ast.fix_missing_locations(tree)


def global_component_max(function, comm):
    import numpy as np
    from mpi4py import MPI
    n = function.function_space.dofmap.index_map.size_local
    bs = function.function_space.dofmap.index_map_bs
    local = function.x.array[:n*bs]
    return comm.allreduce(float(np.max(local, initial=-np.inf)), op=MPI.MAX)


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    import afsic
    import dolfinx
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    demo = Path(os.environ.get('AFSI337_DEMO', '/root/afsi/afsic/demo/demo_337'))
    folder = Path(os.environ['AFSI337_LOGDIR'])
    every = int(os.environ.get('AFSI337_OUTPUT_EVERY', '400'))
    script = demo / 'fsi_paralell_fibers_contraction.py'
    folder.mkdir(parents=True, exist_ok=True)
    fields = folder / 'fields'
    fields.mkdir(exist_ok=True)
    tree = offline_tree(script.read_text(encoding='utf-8'), str(fields) + '/', folder.name, every)
    config = {}
    namespace = dict(__name__='__main__', __file__=str(script), global_component_max=global_component_max)

    def init(project, experiment, cfg):
        expected = dict(T=2., dt=1/20000, num_steps=40000, Nx=32, Ny=32, Nz=32,
                        velocity_order=2, force_order=2, pressure_order=1, rho=1., mu=1.,
                        beta=5e5, kappa=5e5, diastole_time=1.5, systole_pressure=150000.,
                        max_tension=600000., deviatoric=False, contraction=True)
        if any(cfg.get(k) != value for k, value in expected.items()):
            raise ValueError('container demo_337 parameters differ from comparison settings')
        config.update(cfg)
        if comm.rank == 0:
            (folder / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')

    def upload(t, data):
        # Native source calls this on rank 0 only. Do not add collectives here.
        row = dict(source_time_s=float(t), accepted_time_s=float(t)+config['dt'],
                   step=round(float(t)/config['dt'])+1, **data)
        path = folder / 'history.csv'
        exists = path.exists()
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    afsic.swanlab_init = init
    afsic.swanlab_upload = upload
    original_solver = afsic.ChorinSolver

    class CheckedChorin(original_solver):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for solver in (self.solver1, self.solver2, self.solver3):
                solver.setErrorIfNotConverged(True)

    afsic.ChorinSolver = CheckedChorin
    os.chdir(demo)
    sys.path.insert(0, str(demo))
    started, code = perf_counter(), 0
    try:
        if comm.rank == 0:
            print(f'Native AFSI demo_337: {comm.size} MPI ranks; native C++ IB; offline; output every {every} steps', flush=True)
        exec(compile(tree, str(script), 'exec'), namespace)
    except BaseException as exc:
        code = 130 if isinstance(exc, KeyboardInterrupt) else 1
        traceback.print_exc()
        if comm.size > 1:
            comm.Abort(code)
    finally:
        for name in ('file_velocity', 'file_solid'):
            file = namespace.get(name)
            if file is not None:
                try:
                    file.close()
                except Exception:
                    code = code or 1
                    traceback.print_exc()
        elapsed = comm.allreduce(perf_counter()-started, op=MPI.MAX)
        code = comm.allreduce(code, op=MPI.MAX)
        if comm.rank == 0:
            (folder / 'runtime.txt').write_text(
                f'elapsed_seconds={elapsed:.6f} exit_code={code} mpi_ranks={comm.size}\n', encoding='utf-8')
            report = dict(completed=code == 0 and namespace.get('step', -1)+1 == config.get('num_steps'),
                          elapsed_seconds=elapsed, exit_code=code, mpi_ranks=comm.size,
                          dolfinx=dolfinx.__version__, config=config, output_every=every,
                          input='/root/afsi-data/337_ideal_left_ventricle',
                          timing_scope='setup, native solve, diagnostics and XDMF output; excludes MPI preflight',
                          differences='native Q2/Q1 FEM and nodal IB versus GPU MAC and quadrature IB',
                          volume_metric='native volume is wall volume, not cavity volume',
                          velocity_metric='u_max is globally reduced signed component maximum')
            (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            result = demo / 'data/systole-afsi.txt'
            if code == 0 and result.is_file():
                (folder / 'systole-afsi.txt').write_bytes(result.read_bytes())
    return code


if __name__ == '__main__':
    sys.exit(main())
