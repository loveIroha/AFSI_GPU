"""Source load values, geometry/frame conventions and exact field persistence."""
import json
import numpy as np
import pytest
import torch
from afsi_torch.afsi337 import AFSI337Loads, default_quadrature, generated_model, geometry_config
from afsi_torch.afsi337_io import load_native_solid
from afsi_torch.cycle_checkpoint import load_cycle
from afsi_torch.geometry import ENDO, EPI, BASE
from afsi_torch.geometry.ellipsoid_fibers import ellipsoidal_frame
from examples.lv_cycle import run

DEVICES = ['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='CUDA unavailable'))]


def test_source_load_is_simultaneous_ramp_and_hold():
    load = AFSI337Loads()
    for t,p,a in [(0,0,0),(.75,75000,300000),(1.5,150000,600000),
                   (2,150000,600000),(3.2,150000,600000)]:
        assert load.at(t) == pytest.approx((p,a))
    assert not hasattr(load,'period')
    for t in [-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):
            load.at(t)


def test_reference_quadrature_matches_basix():
    basix = pytest.importorskip('basix')
    for cell,(q,w) in zip((basix.CellType.tetrahedron,basix.CellType.triangle),default_quadrature()):
        expected_q,expected_w = basix.make_quadrature(cell,4)
        np.testing.assert_allclose(q.numpy(),expected_q,rtol=0,atol=2e-15)
        np.testing.assert_allclose(w.numpy(),expected_w,rtol=0,atol=2e-15)


def test_native_hash_mapping_uses_raw_coordinates_and_rejects_collisions(tmp_path):
    from validation.export_afsi337_solid import coordinate_key,load_fields
    xyz = np.array([[-17.,0.,0.],[5.,7.,1.],[2.,4.,3.]])
    fiber = np.eye(3)
    sheet = np.roll(fiber,1,axis=1)
    np.savetxt(tmp_path/'f0.txt',fiber.ravel())
    np.savetxt(tmp_path/'s0.txt',sheet.ravel())
    np.savetxt(tmp_path/'cdm.txt',[coordinate_key(x) for x in xyz],fmt='%d')
    f,s = load_fields(tmp_path,xyz[[2,0,1]])
    np.testing.assert_array_equal(f,fiber[[2,0,1]])
    np.testing.assert_array_equal(s,sheet[[2,0,1]])
    with pytest.raises(ValueError,match='missing'):
        load_fields(tmp_path,xyz/10+np.array([3.5,2.5,2.5]))
    with pytest.raises(ValueError,match='ambiguous'):
        load_fields(tmp_path,np.array([[0.,0.,0.],[1.,-10.,0.]]))


@pytest.mark.parametrize('device',DEVICES)
def test_ellipsoid_frame_against_parametric_derivatives(device):
    config = geometry_config()
    t = np.array([0.,.25,.7,1.])
    mu = np.array([1.7,2.,2.5,2.9])
    theta = np.array([.1,1.,3.,5.])
    rs,rl = .7+.3*t,1.7+.3*t
    X = np.column_stack((rl*np.cos(mu),rs*np.sin(mu)*np.cos(theta),rs*np.sin(mu)*np.sin(theta)))
    unit = lambda v:v/np.linalg.norm(v,axis=1)[:,None]
    em = unit(np.column_stack((-rl*np.sin(mu),rs*np.cos(mu)*np.cos(theta),rs*np.cos(mu)*np.sin(theta))))
    et = unit(np.column_stack((np.zeros(4),-rs*np.sin(mu)*np.sin(theta),rs*np.sin(mu)*np.cos(theta))))
    angle = (1-2*t)*np.pi/2
    f = unit(np.sin(angle)[:,None]*em+np.cos(angle)[:,None]*et)
    s = unit(np.cross(f,unit(np.cross(em,et))))
    actual = ellipsoidal_frame(torch.tensor(X+config.center,device=device),config,
                               torch.tensor(t,device=device))
    np.testing.assert_allclose(actual.fiber.cpu(),f,atol=2e-15)
    np.testing.assert_allclose(actual.sheet.cpu(),s,atol=2e-15)


@pytest.mark.parametrize('device',DEVICES)
def test_generated_geometry_harmonic_field_and_nonzero_force(device):
    pytest.importorskip('gmsh')
    model = generated_model(mesh_size=.4,device=device)
    mesh,t = model.mesh,model.fibers.transmural
    torch.testing.assert_close(mesh.X[mesh.surface(BASE),0],torch.full_like(mesh.X[mesh.surface(BASE),0],4.))
    for label,value in [(ENDO,0.),(EPI,1.)]:
        torch.testing.assert_close(t[mesh.surface(label)],torch.full_like(t[mesh.surface(label)],value))
    # Independent dense P1 assembly verifies interior equilibrium and natural base.
    cells = mesh.cells[:,:4].cpu().numpy()
    X = mesh.X.cpu().numpy()
    K = np.zeros((mesh.vertex_count,mesh.vertex_count))
    for ids in cells:
        d = (X[ids[1:]]-X[ids[0]]).T
        grad = np.array([[-1,-1,-1],[1,0,0],[0,1,0],[0,0,1]])@np.linalg.inv(d)
        K[np.ix_(ids,ids)] += grad@grad.T*abs(np.linalg.det(d))/6
    free = np.ones(mesh.vertex_count,dtype=bool)
    for label in (ENDO,EPI):
        free[mesh.surface(label)[:,:3].cpu().numpy().ravel()] = False
    assert np.linalg.norm((K@t[:mesh.vertex_count].cpu().numpy())[free]) < 1e-10
    assert model.geometry.weights.shape[1] == 14
    assert len(model.endo.quadrature_weights) == 6
    model.validate(mesh.X)
    assert torch.linalg.vector_norm(model.force(mesh.X,.75)) > 1
    assert abs(model.diagnostics(mesh.X)['wall_volume_cm3']/mesh.config.wall_volume-1) < .15


@pytest.mark.parametrize('device',DEVICES)
def test_aligned_driver_restart_preserves_load_frame_and_quadrature(tmp_path,device):
    pytest.importorskip('gmsh')
    common = dict(device=device,profile='afsi337',mesh_size=.4,fluid_cells=8,
                  write_vtk=False,history_every=1,log_every=100,checkpoint_every=2)
    whole = run(**common,output=tmp_path/'whole',end_time=.0003)
    run(**common,output=tmp_path/'split',end_time=.00015)
    resumed = run(device=device,profile='afsi337',resume=tmp_path/'split'/'checkpoint.npz',
                  end_time=.0003,write_vtk=False,history_every=1)
    assert resumed['load_protocol'] == 'ramp-and-hold'
    assert resumed['period_s'] is None and not resumed['full_cycle_completed']
    assert not resumed['reference_load_completed']
    assert resumed['settings']['origin'] == [0.,0.,0.]
    assert not whole['source_alignment']['fluid_mesh_matched']
    a,sa,*_ = load_cycle(tmp_path/'whole'/'checkpoint.npz',device)
    b,sb,*_ = load_cycle(tmp_path/'split'/'checkpoint.npz',device)
    assert isinstance(b.loads,AFSI337Loads)
    for name in ('fiber','sheet','transmural'):
        torch.testing.assert_close(getattr(a.fibers,name),getattr(b.fibers,name),atol=1e-12,rtol=1e-12)
    for name in ('x','velocity','pressure','force'):
        torch.testing.assert_close(getattr(sa,name),getattr(sb,name),rtol=1e-8,atol=1e-8)


def test_native_import_preserves_coefficients_and_rejects_bad_tags(tmp_path):
    pytest.importorskip('gmsh')
    model = generated_model(mesh_size=.4)
    mesh = model.mesh
    metadata = dict(schema=1,producer='afsi337-solid-export',units='cm')
    data = dict(X=mesh.X.numpy(),cells=mesh.cells.numpy(),tagged_vertices=mesh.faces[:,:3].numpy(),
        facet_tags=mesh.facet_tags.numpy(),fiber=model.fibers.fiber.numpy()*1.01,
        sheet=model.fibers.sheet.numpy()*.99,volume_points=model.volume_quadrature[0].numpy(),
        volume_weights=model.volume_quadrature[1].numpy(),surface_points=model.surface_quadrature[0].numpy(),
        surface_weights=model.surface_quadrature[1].numpy())
    path = tmp_path/'native.npz'
    np.savez_compressed(path,metadata=json.dumps(metadata),**data)
    imported = load_native_solid(path)
    np.testing.assert_array_equal(imported.fibers.fiber.numpy(),data['fiber'])
    np.testing.assert_array_equal(imported.fibers.sheet.numpy(),data['sheet'])
    data['tagged_vertices'][0] = data['tagged_vertices'][1]
    np.savez_compressed(path,metadata=json.dumps(metadata),**data)
    with pytest.raises(ValueError,match='exactly'):
        load_native_solid(path)

