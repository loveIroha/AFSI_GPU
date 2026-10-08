"""MPI launch preparation tests without a Docker/DOLFINx dependency."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module(path):
    spec = importlib.util.spec_from_file_location('native337',Path(__file__).parents[1]/path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


SOURCE = '''
config={"dt":1/20000,"num_steps":40000}
config["output_path"]=online_path() if rank==0 else None
config["experiment_name"]=online_counter() if rank==0 else None
for step in range(config["num_steps"]):
    ns_solver.solve_one_step()
    ib_interpolation.fluid_to_solid(ns_solver.u_._cpp_object,solid_velocity._cpp_object)
    solid_coords.x.array[:]+=solid_velocity.x.array[:]*config['dt']
    u_max=ns_solver.u_.x.array.max()
    ib_interpolation.solid_to_fluid(ns_solver.f._cpp_object,solid_force._cpp_object)
    if step == 1 or time_manager.should_output(step):
        file_solid.write_function(solid_coords,step*config['dt'])
'''


def test_native_physics_retained_and_ghost_sync_precedes_coordinate_update():
    runner = module('scripts/afsi337_mpi_runner.py')
    tree = runner.offline_tree(SOURCE,'/tmp/fields/','offline')
    before = ast.parse(SOURCE)
    loop = tree.body[3]
    assert ast.dump(loop.body[0])==ast.dump(before.body[3].body[0])
    assert ast.dump(loop.body[3])==ast.dump(before.body[3].body[2])
    assert ast.unparse(loop.body[2])=='solid_velocity.x.scatter_forward()'
    assert ast.unparse(loop.body[6])=='solid_force.x.scatter_forward()'
    assert tree.body[1].value.value=='/tmp/fields/'
    assert tree.body[2].value.value=='offline'
    code = ast.unparse(loop.body[-1].test)
    sample = [n for n in range(40000) if eval(code,{'step':n,'config':{'num_steps':40000}})]
    assert len(sample)==100 and sample[0]==399 and sample[-1]==39999


def test_unrecognized_source_fails_before_launch():
    runner = module('scripts/afsi337_mpi_runner.py')
    with pytest.raises(ValueError,match='unexpected'):
        runner.offline_tree(SOURCE.replace('ib_interpolation.fluid_to_solid','other.fluid_to_solid'),'out','name')
    with pytest.raises(ValueError,match='positive'):
        runner.offline_tree(SOURCE,'out','name',0)


def test_3d_stencil_partition_unity_and_discrete_power():
    checker = module('validation/check_afsi337_mpi.py')
    points = np.array([[2.13,2.29,1.9],[3.85,2.52,2.7],[1.54,2.4,2.9]])
    ids,w,volume = checker.stencil(points)
    np.testing.assert_allclose(w.sum(1),1.,rtol=0,atol=1e-15)
    def phi(r):
        r=abs(r)
        return (3-2*r+np.sqrt(1+4*r-4*r*r))/8 if r<1 else (5-2*r-np.sqrt(-7+12*r-4*r*r))/8 if r<2 else 0.
    for q in range(len(points)):
        for entry, index in enumerate(ids[q]):
            node=np.array(np.unravel_index(index,(65,65,65)))
            expected=np.prod([phi(r) for r in points[q]/(5/64)-node])
            np.testing.assert_allclose(w[q,entry],expected,rtol=0,atol=1e-15)
    rng=np.random.default_rng(4)
    velocity=rng.normal(size=(65**3,3));force=rng.normal(size=(len(points),3))
    gathered=(velocity[ids]*w[...,None]).sum(1)
    spread=np.zeros_like(velocity)
    np.add.at(spread,ids.reshape(-1),(w[...,None]*force[:,None]/volume).reshape(-1,3))
    np.testing.assert_allclose((gathered*force).sum(),volume*(spread*velocity).sum(),rtol=1e-13,atol=1e-13)
