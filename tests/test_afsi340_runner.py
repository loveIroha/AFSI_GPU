"""Test only native-runner rewriting/data integrity; requires no Docker/DOLFINx."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


def module(path):
    spec=importlib.util.spec_from_file_location('native340',Path(__file__).parents[1]/path)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_offline_rewrite_is_limited_to_paths_and_counter():
    runner=module('scripts/afsi340_offline_runner.py')
    source='''
config={"T":3., "dt":1/16000}
config["output_path"] = unique_filename("demo", "tag") if rank == 0 else None
config["output_path"] = broadcast(config["output_path"])
config["experiment_name"] = requests.get("http://counter.example/demo").text if rank == 0 else None
config["experiment_name"] = broadcast(config["experiment_name"])
with XDMFFile(comm,f"/home/dolfinx/afsi/data/340-valve/mesh-340.xdmf","r") as file:
    mesh=file.read_mesh(name="mesh")
for step in range(48000):
    ns_solver.solve_one_step()
    solid_coords += dt*solid_velocity
'''
    tree=runner.offline_tree(source,'/tmp/input.xdmf','/tmp/out/','offline')
    before=ast.parse(source)
    for i in (0,2,4,6):
        assert ast.dump(before.body[i])==ast.dump(tree.body[i])
    namespace=dict(rank=0,broadcast=lambda x:x)
    exec(compile(ast.Module(tree.body[:5],type_ignores=[]),'<test>','exec'),namespace)
    assert namespace['config']['output_path']=='/tmp/out/'
    assert namespace['config']['experiment_name']=='offline'
    assert tree.body[5].items[0].context_expr.args[1].values[0].value=='/tmp/input.xdmf'
    with pytest.raises(ValueError,match='unexpected'):
        runner.offline_tree(source.replace('/340-valve/','/other/'),'input','output','offline')


def test_native_input_selects_reference_not_deformed_checkpoint(tmp_path):
    writer=module('validation/write_afsi340_native_inputs.py')
    config=dict(width=.0212,length=.7,x_right=2.,height=1.61,C0=2e5,C1=1e6,kappa=4e5,beta=1e8,mesh_size=.01)
    X=np.array([[1.9788,0],[2.,0],[2.,.7],[1.9788,.91],[2.,.91],[2.,1.61],
                [1.9894,0],[1.9894,.35],[2.,.35],[1.9894,.91],[1.9894,1.26],[2.,1.26]],dtype=np.float64)
    data=dict(X=X,x=X+.01,cells=np.array([[0,1,2,6,7,8],[3,4,5,9,10,11]],dtype=np.int64),cell_tags=np.array([1,11],dtype=np.int64))
    meta=dict(schema=1,producer='afsi-torch-valve-mac',units='cm-g-s-per-unit-thickness',vertex_count=6,
              config=config,settings=dict(dt=1/16000,nx=256,ny=64,rho=1.,mu=.1))
    digest=hashlib.sha256(json.dumps(meta,sort_keys=True,allow_nan=False).encode())
    for key in sorted(data):
        digest.update(key.encode()); digest.update(data[key].tobytes())
    meta['sha256']=digest.hexdigest()
    path=tmp_path/'source.npz'
    np.savez_compressed(path,metadata=json.dumps(meta),**data)
    vertices,cells,tags,_,_=writer.prepare(path)
    np.testing.assert_array_equal(vertices,X[:6])
    np.testing.assert_array_equal(cells,data['cells'][:,:3])
    data['X'][0,0]+=.001
    np.savez_compressed(path,metadata=json.dumps(meta),**data)
    with pytest.raises(ValueError,match='intact'):
        writer.prepare(path)


def test_runner_records_local_history_and_wall_time(tmp_path,monkeypatch):
    import sys
    import types
    runner=module('scripts/afsi340_offline_runner.py')
    expected=dict(T=3.,dt=1/16000,num_steps=48000,Nx=128,Ny=32,velocity_order=2,force_order=2,
                  pressure_order=1,rho=1.,mu=.1,C0=2e5,C1=1e6,kappa=4e5,beta=1e8)
    source=f'''import afsic
config={expected!r}
config["output_path"]=undefined_online_output() if True else None
config["experiment_name"]=undefined_online_counter() if True else None
input_path=f"/home/dolfinx/afsi/data/340-valve/mesh-340.xdmf"
afsic.swanlab_init("demo-340",config["experiment_name"],config)
for step in range(config["num_steps"]):
    pass
afsic.swanlab_upload(step*config["dt"],dict(x_displacement=.1,y_displacement=.01,volume=.02968))
'''
    demo=tmp_path/'demo'; demo.mkdir()
    (demo/'fsi_paralell.py').write_text(source,encoding='utf-8')
    folder=tmp_path/'run'
    monkeypatch.setenv('AFSI340_DEMO',str(demo))
    monkeypatch.setenv('AFSI340_LOGDIR',str(folder))
    monkeypatch.setenv('AFSI340_INPUT',str(tmp_path/'mesh.xdmf'))
    monkeypatch.setitem(sys.modules,'afsic',types.ModuleType('afsic'))
    monkeypatch.setitem(sys.modules,'dolfinx',types.SimpleNamespace(__version__='test'))
    monkeypatch.setitem(sys.modules,'mpi4py',types.SimpleNamespace(MPI=types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(size=1))))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys,'path',list(sys.path))
    # Do not change the test process's interrupt handler.
    monkeypatch.setattr(runner.signal,'signal',lambda *a:None)
    assert runner.main()==0
    report=json.loads((folder/'report.json').read_text())
    assert report['completed'] and report['last_entered_step']==48000 and report['elapsed_seconds']>0
    import csv
    with (folder/'history.csv').open() as file:
        row=next(csv.DictReader(file))
    assert float(row['accepted_time_s'])==3. and int(row['step'])==48000
    assert 'exit_code=0' in (folder/'runtime.txt').read_text()
