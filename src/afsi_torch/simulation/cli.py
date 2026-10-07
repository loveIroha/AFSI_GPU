"""Shared CLI wiring: demo defaults < JSON config < explicit CLI overrides."""
import argparse
from . import lv_mac, valve_mac
from ..config import LVSimulationConfig, ValveSimulationConfig, LVFEMSimulationConfig, load_config, save_config


def _common(parser, *, mac=True):
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ib-backend',choices=('reference','cuda'),help='IB transfer execution; cuda requires the native extension')
    parser.add_argument('--output')
    parser.add_argument('--resume', help='restore geometry, physical settings and dt from checkpoint')
    parser.add_argument('--config', help='partial JSON case configuration')
    parser.add_argument('--write-config', help='write the selected configuration template and exit')
    parser.add_argument('--end-time', type=float)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--mesh-size', type=float)
    parser.add_argument('--fluid-lengths', type=float, nargs='+', help='domain lengths in cm, in axis order')
    parser.add_argument('--rho', type=float, help='fluid density in g/cm^3')
    parser.add_argument('--mu', type=float, help='dynamic viscosity in g/(cm s)')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--checkpoint-every', type=int)
    if mac:
        parser.add_argument('--warm-start', action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument('--mass-backend', choices=('pcg','graph'))


def _options(parser, argv, defaults, default_output):
    options = vars(parser.parse_args(argv))
    path = options.pop('config')
    write = options.pop('write_config')
    if options['resume'] and path:
        parser.error('--resume restores its physical settings; omit --config')
    config = load_config(path, defaults) if path else defaults
    if write:
        save_config(write, config)
        print(f'Configuration template: {write}')
        return None
    if options['resume']:
        options['end_time'] = config.time.end_time if options['end_time'] is None else options['end_time']
        options['log_every'] = config.output.log_every if options['log_every'] is None else options['log_every']
        options['checkpoint_every'] = config.output.checkpoint_every if options['checkpoint_every'] is None else options['checkpoint_every']
    else:
        options['case_config'] = config
        if options['output'] is None:
            options['output'] = default_output
    return options


def _finish(report):
    print(f"{report['status']}: step={report['accepted_steps']}, t={report['reached_time_s']:.6f} s, "
          f"elapsed_seconds={report['elapsed_seconds']:.3f}", flush=True)
    return report


def lv_main(argv=None, *, defaults=None, default_output='results/demo_ideal_lv/mac'):
    defaults = LVSimulationConfig() if defaults is None else defaults
    parser = argparse.ArgumentParser(description='Generated P2 left ventricle / 3D MAC IB-FSI (cm-g-s)')
    _common(parser)
    parser.add_argument('--fluid-cells', type=int, help='equal cell count on all three axes')
    parser.add_argument('--fluid-shape', type=int, nargs=3, metavar=('NX','NY','NZ'))
    parser.add_argument('--fluid-origin', type=float, nargs=3)
    parser.add_argument('--interaction-degree', type=int)
    parser.add_argument('--vtk', dest='write_vtk', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--output-every', type=int)
    parser.add_argument('--execution-backend', choices=('torch','fused'))
    parser.add_argument('--pressure-backend', choices=('torch','fused','workspace','graph'))
    parser.add_argument('--solid-backend', choices=('reference','pointwise'))
    parser.add_argument('--coupling-backend', choices=('reference','optimized'))
    options = _options(parser, argv, defaults, default_output)
    return None if options is None else _finish(lv_mac.run(**options))


def valve_main(argv=None, *, defaults=None, default_output='results/demo_ideal_valve/mac'):
    defaults = ValveSimulationConfig() if defaults is None else defaults
    parser = argparse.ArgumentParser(description='Generated P2 valve / 2D MAC IB-FSI (cm-g-s)')
    _common(parser)
    parser.add_argument('--nx', type=int)
    parser.add_argument('--ny', type=int)
    parser.add_argument('--field-every', type=int, help='VTK frame interval in steps; 0 disables output')
    parser.add_argument('--fluid-fields', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fused', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--execution-backend', choices=('reference','optimized'))
    parser.add_argument('--pressure-backend', choices=('auto','reference','workspace','graph'))
    options = _options(parser, argv, defaults, default_output)
    return None if options is None else _finish(valve_mac.run(**options))


def fem_main(argv=None, *, defaults=None, default_output='results/demo_ideal_lv/fem'):
    from dataclasses import replace
    from .lv_fem import run
    defaults=LVFEMSimulationConfig() if defaults is None else defaults
    p=argparse.ArgumentParser(description='Generated P2 left ventricle / Q2-Q1 FEM IB-FSI (cm-g-s)')
    _common(p,mac=False)
    p.add_argument('--fluid-cells',type=int)
    p.add_argument('--fluid-shape',type=int,nargs=3)
    p.add_argument('--fluid-origin',type=float,nargs=3)
    p.add_argument('--box-length',type=float)
    p.add_argument('--history-every',type=int)
    p.add_argument('--output-every',type=int)
    p.add_argument('--vtk',dest='write_vtk',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--backend',choices=('csr','quadrature'))
    p.add_argument('--check-every',type=int)
    options=_options(p,argv,defaults,default_output)
    if options is None:
        return None
    fluid_options={name:options.pop(name) for name in ('fluid_shape','fluid_lengths','fluid_origin','rho','mu')}
    if options['resume']:
        if any(value is not None for value in fluid_options.values()):
            p.error('--resume restores fluid grid and physical settings')
    else:
        if options['fluid_cells'] is not None and fluid_options['fluid_shape'] is not None:
            p.error('choose --fluid-cells or --fluid-shape')
        if options['box_length'] is not None and fluid_options['fluid_lengths'] is not None:
            p.error('choose --box-length or --fluid-lengths')
        cfg=options['case_config']
        changes={({'fluid_shape':'shape','fluid_lengths':'lengths','fluid_origin':'origin'}.get(name,name)):value
                 for name,value in fluid_options.items() if value is not None}
        options['case_config']=replace(cfg,fluid=replace(cfg.fluid,**changes))
    return _finish(run(**options,profile='afsi337'))
