"""Run native demo_340, replacing only input/output paths and online tracking.

Native MPI IB is retained; distributed ghost synchronization is added.
"""
import ast
import csv
import json
import os
import signal
from pathlib import Path
import sys
from time import perf_counter
import traceback


def offline_tree(source,input_path,output_path,experiment,*,parallel=False):
    class Replace(ast.NodeTransformer):
        def __init__(self):
            self.counts=dict(input=0,output=0,counter=0)
            self.syncs=dict(fluid_to_solid=0,solid_to_fluid=0)

        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.value,ast.IfExp):
                target=node.targets[0]
                if (isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and
                    target.value.id=='config' and isinstance(target.slice,ast.Constant)):
                    key=target.slice.value
                    if key in ('output_path','experiment_name'):
                        kind='output' if key=='output_path' else 'counter'
                        self.counts[kind]+=1
                        node.value=ast.Constant(output_path if kind=='output' else experiment)
            if parallel and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='u_max':
                node.value=ast.Call(ast.parse('mesh.comm.allreduce',mode='eval').body,[node.value],
                                    [ast.keyword('op',ast.parse('MPI.MAX',mode='eval').body)])
            return self.generic_visit(node)

        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value.endswith('/340-valve/mesh-340.xdmf'):
                self.counts['input']+=1
                return ast.copy_location(ast.Constant(input_path),node)
            return node
        def visit_Expr(self,node):
            node=self.generic_visit(node)
            if parallel and isinstance(node.value,ast.Call):
                call=node.value
                if (isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and
                    call.func.value.id=='ib_interpolation' and call.func.attr in ('fluid_to_solid','solid_to_fluid')):
                    name='solid_velocity' if call.func.attr=='fluid_to_solid' else 'solid_force'
                    self.syncs[call.func.attr]+=1
                    sync=ast.parse(f'{name}.x.scatter_forward()').body[0]
                    return [node,ast.copy_location(sync,node)]
            return node
    transform=Replace()
    tree=transform.visit(ast.parse(source))
    if transform.counts!=dict(input=1,output=1,counter=1):
        raise ValueError(f'unexpected demo_340 source layout: {transform.counts}; refusing ambiguous rewrite')
    if parallel and transform.syncs!=dict(fluid_to_solid=1,solid_to_fluid=1):
        raise ValueError(f'unexpected MPI IB call layout: {transform.syncs}')
    return ast.fix_missing_locations(tree)


def main():
    # Background shell jobs may inherit SIGINT ignored; allow the printed stop command.
    signal.signal(signal.SIGINT,signal.default_int_handler)
    import afsic
    import dolfinx
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    if comm.size>1:
        original_solver=afsic.ChorinSolver
        class CheckedChorin(original_solver):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                for ksp in (self.solver1,self.solver2,self.solver3):
                    ksp.setErrorIfNotConverged(True)
        afsic.ChorinSolver=CheckedChorin
    demo=Path(os.environ.get('AFSI340_DEMO','/root/afsi/afsic/demo/demo_340'))
    folder=Path(os.environ['AFSI340_LOGDIR'])
    input_path=Path(os.environ['AFSI340_INPUT'])
    script=demo/'fsi_paralell.py'
    folder.mkdir(parents=True,exist_ok=True)
    out=folder/'fields'; out.mkdir(exist_ok=True)
    source=script.read_text(encoding='utf-8')
    tree=offline_tree(source,str(input_path),str(out)+'/',folder.name,parallel=comm.size>1)
    config={}
    def init(project,experiment,cfg):
        config.update(cfg)
        expected=dict(T=3.,dt=1/16000,num_steps=48000,Nx=128,Ny=32,velocity_order=2,force_order=2,
                      pressure_order=1,rho=1.,mu=.1,C0=2e5,C1=1e6,kappa=4e5,beta=1e8)
        if any(cfg.get(k)!=v for k,v in expected.items()):
            raise ValueError('container demo defaults differ from the GPU comparison setup')
        if comm.rank==0:
            (folder/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    def upload(t,data):
        row=dict(source_time_s=float(t),accepted_time_s=float(t)+config['dt'],step=round(t/config['dt'])+1,**data)
        path=folder/'history.csv'
        exists=path.exists()
        with path.open('a',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    afsic.swanlab_init=init
    afsic.swanlab_upload=upload
    os.chdir(demo); sys.path.insert(0,str(demo))
    namespace=dict(__name__='__main__',__file__=str(script))
    started=perf_counter(); code=0
    try:
        if comm.rank==0:
            print(f'Native AFSI demo_340: {comm.size} MPI ranks; native C++ IB retained; online tracking disabled',flush=True)
        exec(compile(tree,str(script),'exec'),namespace)
    except BaseException as exc:
        code=130 if isinstance(exc,KeyboardInterrupt) else 1
        traceback.print_exc()
        if comm.size>1:
            # A rank-local failure must not leave other ranks hanging in a collective.
            comm.Abort(code)
    finally:
        for name in ('file_velocity','file_solid'):
            file=namespace.get(name)
            if file is not None:
                try:
                    file.close()
                except Exception:
                    code=code or 1
                    traceback.print_exc()
        elapsed=perf_counter()-started
        if comm.size>1:
            elapsed=comm.allreduce(elapsed,op=MPI.MAX)
            code=comm.allreduce(code,op=MPI.MAX)
        if comm.rank!=0:
            return code
        (folder/'runtime.txt').write_text(f'elapsed_seconds={elapsed:.6f} exit_code={code}\n',encoding='utf-8')
        report=dict(elapsed_seconds=elapsed,exit_code=code,
                    completed=code==0 and namespace.get('step',-1)+1==config.get('num_steps'),config=config,
                    dolfinx=dolfinx.__version__,mpi_ranks=MPI.COMM_WORLD.size,
                    threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                    input=str(input_path),source=str(script),
                    last_entered_step=namespace.get('step',-1)+1,
                    timing_scope='native source including setup, per-step diagnostics and XDMF output; excludes input conversion',
                    differences='native Q2/Q1 FEM + nodal IB versus GPU MAC + quadrature IB')
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'elapsed_seconds={elapsed:.3f} exit_code={code}',flush=True)
    return code


if __name__=='__main__':
    sys.exit(main())
