"""Public configuration reaches physics, solvers, CLI, checkpoints and export."""
from dataclasses import asdict, replace
import json
import numpy as np
import pytest
import torch
from afsi_torch.config import (LVSimulationConfig, ValveSimulationConfig, TimeConfig, FluidConfig,
    OutputConfig, LVExecutionConfig, InletConfig, MGOptions, mass_options,
    load_config, save_config, lv_grid, GuccioneParameters, AFSI337Loads)
from test_mac_execution import DEVICES


def test_partial_json_preserves_case_defaults_and_round_trips(tmp_path):
    path=tmp_path/'lv.json'
    path.write_text(json.dumps(dict(geometry=dict(mesh_size=.4),fluid=dict(mu=.9))))
    cfg=load_config(path)
    assert cfg.geometry.inner_axes==(.7,.7,1.7) and cfg.geometry.long_axis=='x'
    assert cfg.fluid.shape==(64,64,64) and cfg.fluid.mu==.9
    save_config(path,cfg)
    assert load_config(path)==cfg
    path.write_text(json.dumps(dict(fluid=dict(rho=1.2),solid=dict(mesh_size=.06))))
    cfg=load_config(path,ValveSimulationConfig)
    assert cfg.fluid.shape==(256,64) and cfg.fluid.mu==.1
    assert cfg.time.dt==1/16000 and cfg.solid.C1==1e6
    path.write_text(json.dumps(dict(fluid=dict(nxx=32))))
    with pytest.raises(ValueError,match='unknown config.fluid'):
        load_config(path)


@pytest.mark.parametrize('build,match',[
    (lambda: TimeConfig(0,2),'dt'),
    (lambda: TimeConfig(3e-5,.001),'integer multiple'),
    (lambda: FluidConfig((True,16,16)), 'integer cell'),
    (lambda: FluidConfig((64,64,63)), 'coarsen'),
    (lambda: LVSimulationConfig(fluid=FluidConfig((256,256,256))), 'viscous'),
    (lambda: ValveSimulationConfig(fluid=FluidConfig((32,8),(8.,2.),(0.,0.),mu=.1)), 'height'),
    (lambda: LVExecutionConfig(mass_backend='graph'), 'fused'),
    (lambda: LVSimulationConfig(output=OutputConfig(fluid_fields=False)), 'fluid_fields'),
    (lambda: ValveSimulationConfig(mass_solver=replace(mass_options(),check_every=17)), 'check_every'),
])
def test_invalid_config_fails_before_mesh_generation(build,match):
    with pytest.raises(ValueError,match=match):
        build()


def test_lv_grid_accepts_legacy_cubic_and_new_rectangular_settings():
    old=lv_grid(dict(fluid_cells=16,box_length=5.))
    assert old.shape==(16,)*3 and old.origin==(0.,)*3
    new=lv_grid(dict(fluid_shape=[16,32,16],fluid_lengths=[5.,6.,5.],fluid_origin=[.1,-.1,.2]))
    assert new.spacing==(5/16,6/32,5/16) and new.origin==(.1,-.1,.2)


@pytest.mark.parametrize('device',DEVICES)
def test_lv_custom_physics_grid_solvers_and_resume_match_uninterrupted(tmp_path,device):
    pytest.importorskip('gmsh')
    from afsi_torch.simulation.lv_mac import run
    from afsi_torch.mac.checkpoint import load_mac
    from afsi_torch.mac.execution import build_driver
    from test_mac_demo_backends import assert_states
    from validation.export_mac_vtk import export
    cfg=replace(LVSimulationConfig(),time=TimeConfig(1e-5,4e-5),
        fluid=FluidConfig((16,32,16),(5.,6.,5.),(.1,-.1,.2),rho=1.1,mu=.8),
        geometry=replace(LVSimulationConfig().geometry,mesh_size=.4),
        material=GuccioneParameters(C=15000.,kappa=400000.),
        loads=AFSI337Loads(pressure=6000.,tension=12000.,ramp_time=.002),beta=300000.,
        pressure_solver=MGOptions(rtol=1e-11,smooth=3,check_every=1),
        mass_solver=replace(mass_options(),rtol=5e-13,check_every=2),
        execution=LVExecutionConfig(execution_backend='fused',pressure_backend='graph' if device=='cuda' else 'workspace',
            solid_backend='pointwise',mass_backend='graph',coupling_backend='optimized',warm_start=True),
        output=OutputConfig(2,2,2,False))
    folder=tmp_path/'resumed'
    first=run(device=device,output=folder,case_config=cfg,end_time=2e-5)
    _,_,settings,_=load_mac(folder/'checkpoint.npz',device)
    assert first['fluid_spacing_cm']==cfg.fluid.spacing
    assert first['material']['C']==15000. and first['loads']['pressure']==6000.
    model,_,_,_=load_mac(folder/'checkpoint.npz',device)
    driver=build_driver(model,settings,device)
    assert driver.flow.pressure_solver.options==cfg.pressure_solver
    assert driver.transfer.options==cfg.mass_solver
    report=run(device=device,resume=folder/'checkpoint.npz',end_time=4e-5)
    run(device=device,output=tmp_path/'full',case_config=cfg)
    _,a,_,_=load_mac(folder/'checkpoint.npz',device)
    _,b,_,_=load_mac(tmp_path/'full'/'checkpoint.npz',device)
    assert_states(a,b)
    resolved=load_config(folder/'configuration.json')
    assert resolved.fluid==cfg.fluid and resolved.time==cfg.time
    assert report['configuration']['geometry']['center']==cfg.geometry.center
    with pytest.raises(ValueError,match='restores physical'):
        run(device=device,resume=folder/'checkpoint.npz',case_config=cfg)
    with pytest.raises(ValueError,match='resume restores'):
        run(device=device,resume=folder/'checkpoint.npz',rho=2.)
    export(folder/'checkpoint.npz',tmp_path/'snapshot',device)
    import xml.etree.ElementTree as ET
    image=ET.parse(tmp_path/'snapshot'/'fluid_000004.vti').getroot().find('ImageData')
    assert image.attrib['WholeExtent']=='0 16 0 32 0 16'
    assert image.attrib['Origin']=='0.1 -0.1 0.2'


@pytest.mark.parametrize('device',DEVICES)
def test_valve_custom_parameters_and_inlet_survive_checkpoint(tmp_path,device):
    pytest.importorskip('gmsh')
    from afsi_torch.simulation.valve_mac import run
    from afsi_torch.mac2d.checkpoint import load
    from afsi_torch.mac2d.execution import build_driver
    cfg=replace(ValveSimulationConfig(),time=TimeConfig(1/16000,2/16000),
        fluid=FluidConfig((32,8),(7.,1.8),(0.,0.),rho=1.2,mu=.15),
        solid=replace(ValveSimulationConfig().solid,height=1.8,mesh_size=.07,C0=180000.,beta=9e7),
        inlet=InletConfig(amplitude=3.,period=.8,offset=1.2),
        pressure_solver=MGOptions(smooth=3,check_every=1),
        mass_solver=replace(mass_options(),check_every=2),output=OutputConfig(1,2,2,False))
    report=run(device=device,output=tmp_path/'valve',case_config=cfg)
    solid,state,settings,_=load(tmp_path/'valve'/'checkpoint.npz',device)
    driver=build_driver(solid,settings,device)
    assert driver.flow.grid.lengths==(7.,1.8)
    assert driver.flow.pressure_solver.options==cfg.pressure_solver
    assert driver.transfer.solver.options==cfg.mass_solver
    expected=3.*(np.sin(2*np.pi*.2/.8)+1.2)*driver.flow.profile
    torch.testing.assert_close(driver.flow.inlet(.2),expected)
    restored=load_config(tmp_path/'valve'/'configuration.json',ValveSimulationConfig)
    assert restored==cfg
    assert report['config']['C0']==180000. and report['inlet']['period']==.8
    resumed=run(device=device,resume=tmp_path/'valve'/'checkpoint.npz',end_time=3/16000,field_every=0)
    assert resumed['settings']['inlet']['amplitude']==3.


def test_cli_json_priority_template_and_resume_guard(tmp_path):
    pytest.importorskip('gmsh')
    from demo.ideal_lv_fsi.run_mac import main
    path=tmp_path/'config.json'
    path.write_text(json.dumps(dict(time=dict(dt=1e-5,end_time=2e-5),geometry=dict(mesh_size=.4),
        fluid=dict(shape=[16,16,16],mu=.9),output=dict(write_vtk=False,log_every=2))))
    report=main(['--device','cpu','--ib-backend','reference','--pressure-backend','torch',
                 '--config',str(path),'--mu','.8','--output',str(tmp_path/'cli')])
    assert report['settings']['dt']==1e-5 and report['settings']['mu']==.8
    assert report['visualization']['enabled'] is False
    template=tmp_path/'template.json'
    assert main(['--write-config',str(template)]) is None
    assert load_config(template).time.end_time==2.
    with pytest.raises(SystemExit):
        main(['--resume',str(tmp_path/'cli'/'checkpoint.npz'),'--config',str(path)])


@pytest.mark.parametrize('device',DEVICES)
def test_fem_config_controls_rectangular_grid_material_and_cli(tmp_path,device):
    pytest.importorskip('gmsh')
    from afsi_torch.config import LVFEMSimulationConfig,FEMFluidConfig,SolverOptions
    from afsi_torch.simulation.lv_fem import run
    from afsi_torch.cycle_checkpoint import load_cycle
    from demo.ideal_lv_fsi.run_fem import main
    cfg=replace(LVFEMSimulationConfig(),time=TimeConfig(1e-5,2e-5),
        fluid=FEMFluidConfig((8,10,8),(5.,6.,5.),(.1,-.1,.2),rho=1.1,mu=.8),
        geometry=replace(LVFEMSimulationConfig().geometry,mesh_size=.4),
        material=GuccioneParameters(C=15000.),loads=AFSI337Loads(pressure=6000.,tension=12000.,ramp_time=.002),
        solver=SolverOptions(max_iterations=4000,recompute_every=200,check_every=2),
        history_every=1,output=OutputConfig(2,2,2,False))
    folder=tmp_path/'fem'
    report=run(device=device,case_config=cfg,output=folder)
    assert report['fluid_element_sizes_cm']==cfg.fluid.spacing
    assert report['settings']['solver']['check_every']==2
    assert report['material']['C']==15000. and report['settings']['rho']==1.1
    assert load_config(folder/'configuration.json',LVFEMSimulationConfig)==cfg
    model,state,_,settings,_=load_cycle(folder/'checkpoint.npz',device)
    assert tuple(settings['fluid_shape'])==(8,10,8)
    resumed=main(['--device',device,'--resume',str(folder/'checkpoint.npz'),
                  '--end-time','0.00003','--no-vtk','--check-every','2'])
    assert resumed['accepted_steps']==3 and resumed['material']['C']==15000.
    template=tmp_path/'fem_template.json'
    assert main(['--write-config',str(template)]) is None
    assert load_config(template,LVFEMSimulationConfig).fluid.shape==(32,)*3
