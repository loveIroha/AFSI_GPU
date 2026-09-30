"""P2 ordering, VTK axis order, physical fields and resumed frame collections."""
import base64
from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zlib
import numpy as np
import pytest
import torch
import meshio
from afsi_torch.mac.output import MACWriter
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.coupling import MACState
from test_mac_execution import setup, DEVICES


def decode(node):
    text=node.text
    # Decode just enough header to determine its complete base64 length.
    blocks=int(np.frombuffer(base64.b64decode(text[:12])[:8],dtype='<u8')[0])
    chars=4*((8*(3+blocks)+2)//3)
    header=np.frombuffer(base64.b64decode(text[:chars]),dtype='<u8')
    payload=base64.b64decode(text[chars:])
    parts=[]; start=0
    for length in header[3:]:
        stop=start+int(length)
        parts.append(zlib.decompress(payload[start:stop])); start=stop
    data=b''.join(parts)
    assert len(data)==(blocks-1)*int(header[1])+int(header[2])
    return np.frombuffer(data,dtype='<f8')


def inputs(device):
    X, geometry, _=setup(device)
    grid=MACGrid((4,6,8),(2.,3.,4.),(-1.,-2.,-3.))
    model=SimpleNamespace(mesh=SimpleNamespace(X=X,cells=geometry.cells),geometry=geometry,
                          fibers=SimpleNamespace(fiber=torch.ones_like(X)))
    xyz=grid.coordinates(device=device)
    pressure=100*xyz[...,0]+10*xyz[...,1]+xyz[...,2]
    velocity=tuple((c+1)*grid.coordinates(c,device=device)[...,c]+.2
                   for c in range(3))
    state=MACState(5,.005,1.02*X+.1,velocity,pressure,torch.cos(X),.004)
    return model, grid, state


@pytest.mark.parametrize('device',DEVICES)
def test_vtk_fields_axis_order_and_deformed_p2_mesh(tmp_path,device):
    model,grid,state=inputs(device)
    saved=state.x.clone()
    writer=MACWriter(tmp_path,model,grid)
    writer.write(state)
    solid=meshio.read(tmp_path/'solid_000005.vtu')
    np.testing.assert_array_equal(solid.points,state.x.cpu().numpy())
    np.testing.assert_allclose(solid.point_data['displacement_cm'],(state.x-model.mesh.X).cpu().numpy())
    np.testing.assert_array_equal(solid.point_data['nodal_force_dyn'],state.force.cpu().numpy())
    np.testing.assert_array_equal(solid.cells_dict['tetra10'],model.mesh.cells.cpu().numpy()[:,[0,1,2,3,4,7,5,6,8,9]])
    np.testing.assert_allclose(solid.cell_data_dict['J_min']['tetra10'],1.02**3,rtol=1e-12)
    np.testing.assert_allclose(solid.cell_data_dict['J_max']['tetra10'],1.02**3,rtol=1e-12)
    root=ET.parse(tmp_path/'fluid_000005.vti').getroot()
    assert root.find('ImageData').attrib['WholeExtent']=='0 4 0 6 0 8'
    arrays={a.attrib['Name']:decode(a) for a in root.findall('.//CellData/DataArray')}
    pressure=arrays['pressure_dyn_per_cm2'].reshape(8,6,4).transpose(2,1,0)
    np.testing.assert_array_equal(pressure,state.pressure.cpu().numpy())
    velocity=arrays['velocity_cm_per_s'].reshape(8,6,4,3).transpose(2,1,0,3)
    expected=grid.coordinates(device=device).cpu().numpy()*np.array([1.,2.,3.])+.2
    np.testing.assert_allclose(velocity,expected,atol=1e-14)
    np.testing.assert_allclose(arrays['divergence_per_s'],6.,atol=1e-14)
    times={a.attrib['Name']:decode(a)[0] for a in root.findall('.//FieldData/DataArray')}
    assert times=={'TimeValue':.005,'force_time_s':.004}
    torch.testing.assert_close(state.x,saved,rtol=0,atol=0)
    # Read the exact VTI binary arrays through meshio's independent VTK XML
    # decoder, in a VTU envelope (meshio does not expose an ImageData reader).
    envelope=tmp_path/'binary_reader_check.vtu'
    meshio.write(envelope,meshio.Mesh(np.zeros((1,3)),
                 [('hexahedron',np.zeros((4*6*8,8),dtype=np.int64))]),header_type='UInt64')
    tree=ET.parse(envelope)
    piece=tree.getroot().find('./UnstructuredGrid/Piece')
    previous=piece.find('CellData')
    if previous is not None:
        piece.remove(previous)
    piece.append(root.find('./ImageData/Piece/CellData'))
    tree.write(envelope)
    decoded=meshio.read(envelope)
    for name,values in arrays.items():
        np.testing.assert_array_equal(decoded.cell_data_dict[name]['hexahedron'].reshape(-1),values)


def test_resume_replaces_future_manifest_and_recovers_only_complete_pairs(tmp_path):
    model,grid,state=inputs('cpu')
    writer=MACWriter(tmp_path,model,grid)
    for step in (0,2,4,6):
        writer.write(replace(state,step=step,time=step*.001))
    (tmp_path/'fluid_000004.vti').unlink()
    resumed=MACWriter(tmp_path,model,grid,resume_time=.004)
    assert [f[0] for f in resumed.frames]==[0.,.002]
    resumed.write(replace(state,step=4,time=.004))
    resumed.write(replace(state,step=8,time=.008))
    resumed.write(replace(state,step=8,time=.008))
    assert [f[0] for f in resumed.frames]==[0.,.002,.004,.008]
    for kind in ('solid','fluid'):
        frames=ET.parse(tmp_path/(kind+'.pvd')).findall('.//DataSet')
        assert [float(f.attrib['timestep']) for f in frames]==[0.,.002,.004,.008]
    # Unreferenced future files remain on disk; manifests select the accepted branch.
    assert (tmp_path/'solid_000006.vtu').exists()


def test_failed_frame_never_publishes_an_unpaired_time(tmp_path,monkeypatch):
    model,grid,state=inputs('cpu')
    writer=MACWriter(tmp_path,model,grid)
    writer.write(state)
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(writer,'_fluid',fail)
    with pytest.raises(OSError,match='disk failure'):
        writer.write(replace(state,step=6,time=.006))
    assert len(writer.frames)==1
    for kind in ('solid','fluid'):
        assert len(ET.parse(tmp_path/(kind+'.pvd')).findall('.//DataSet'))==1


@pytest.mark.parametrize('device',DEVICES)
def test_demo_output_resume_disable_and_numerical_equivalence(tmp_path,device):
    pytest.importorskip('gmsh')
    from demo.ideal_lv_fsi.run_mac import main
    from examples.lv_mac import run
    from afsi_torch.mac.checkpoint import load_mac
    from test_mac_demo_backends import assert_states
    folder=tmp_path/'series'
    report=main(['--device',device,'--mesh-size','.4','--fluid-cells','16',
                 '--end-time','.00015','--output',str(folder),'--output-every','2'])
    assert report['visualization']['frames']==3  # step 0,2,3 final partial frame
    report=main(['--device',device,'--resume',str(folder/'checkpoint.npz'),'--end-time','.0003'])
    frames=ET.parse(folder/'vtk'/'solid.pvd').findall('.//DataSet')
    np.testing.assert_allclose([float(f.attrib['timestep']) for f in frames],
                               [0.,.0001,.00015,.0002,.0003],atol=1e-18,rtol=0)
    assert report['visualization']['frames']==5
    report=main(['--device',device,'--resume',str(folder/'checkpoint.npz'),
                 '--end-time','.0004','--no-vtk'])
    assert report['visualization']['enabled'] is False
    assert len(ET.parse(folder/'vtk'/'solid.pvd').findall('.//DataSet'))==5
    run(device=device,mesh_size=.4,fluid_cells=16,end_time=.0004,output=tmp_path/'plain')
    _,a,settings,_=load_mac(folder/'checkpoint.npz',device)
    _,b,_,_=load_mac(tmp_path/'plain'/'checkpoint.npz',device)
    assert settings['write_vtk'] is False and settings['output_every']==2
    assert_states(a,b)
    from validation.export_mac_vtk import export
    checkpoint=folder/'checkpoint.npz'
    original=checkpoint.read_bytes()
    export(checkpoint,tmp_path/'snapshot',device)
    assert checkpoint.read_bytes()==original
    frames=ET.parse(tmp_path/'snapshot'/'solid.pvd').findall('.//DataSet')
    assert len(frames)==1 and float(frames[0].attrib['timestep'])==.0004
    with pytest.raises(FileExistsError):
        export(checkpoint,tmp_path/'snapshot',device)


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_output_interval_precedes_setup(tmp_path,value):
    from examples.lv_mac import run
    with pytest.raises(ValueError,match='output_every'):
        run(device='cpu',output=tmp_path,output_every=value)
