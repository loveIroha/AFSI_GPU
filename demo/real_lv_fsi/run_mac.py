"""Imported real LV, user H-O UFL and three 0.8 s cycles. Edit CONFIG here."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsi_torch.real_lv import RealLVConfig
from afsi_torch.config import TimeConfig, FluidConfig, OutputConfig, LVExecutionConfig, load_config, save_config
from afsi_torch.holzapfel_ogden import HOParameters, RealLVLoads
from afsi_torch.mac.implicit import MACCouplingOptions
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
from afsi_torch.simulation.real_lv_mac import run

# Confirmed user parameters. Stress: dyn/cm²; basal beta: dyn/cm³.
# Input files stay outside the repository. The shared fluid defaults rho=mu=1.
CONFIG = RealLVConfig(
    source_dir='/mnt/large2/gjh/realistic_left_ventricle', source_units='cm',
    endo_tag=2, epi_tag=1, base_tag=3,
    time=TimeConfig(dt=1e-4, end_time=2.4),
    fluid=FluidConfig(shape=(128,128,128), lengths=(15.,15.,15.), origin=(0.,0.,0.), rho=1., mu=1.),
    material=HOParameters(a=2244.87, b=1.6215, a_f=24267., b_f=1.8268,
                          a_s=5562.38, b_s=.7746, a_fs=3905.16, b_fs=1.695,
                          kappa=5e6, active_stretch_slope=4.9),
    beta=5e6, basal_center_cm=(7.5,7.5),
    loads=RealLVLoads(period=.8, pressure_kpa=1.067, pressure_increment_kpa=13.46, tension_kpa=84.26),
    solid_degree=5, interaction_degree=2,
    interaction_quadrature=InteractionQuadratureOptions(mode='adaptive',point_density=2.,rule_family='xiao-gimbutas'),
    coupling=MACCouplingOptions(scheme='cnab-semiimplicit',semiimplicit_solver='anderson-newton'),
    output=OutputConfig(log_every=100, checkpoint_every=1000, output_every=200, write_vtk=True),
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='partial JSON configuration')
    parser.add_argument('--write-config', help='write effective settings and exit')
    parser.add_argument('--mesh-dir', help='directory containing mesh, boundary, fiber and sheet files')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--resume', help='resume a real-LV checkpoint in its original directory')
    parser.add_argument('--resume-dt', type=float,
                        help='subdivide saved dt by an integer; requires --resume and a new --output directory')
    parser.add_argument('--coupling', choices=('explicit-lagged', 'explicit-rk3', 'implicit-newton', 'cnab-midpoint', 'cnab-semiimplicit'),
                        help='coupling time scheme; switching a resumed case requires a new output directory')
    parser.add_argument('--end-time', type=float)
    parser.add_argument('--nonlinear-solver',choices=('newton','anderson-newton'),
                        help='midpoint residual solver; changing a resumed solver requires a new output directory')
    parser.add_argument('--cycles', type=int)
    parser.add_argument('--dt', type=float)
    parser.add_argument('--fluid-cells', type=int)
    parser.add_argument('--fluid-lengths', nargs=3, type=float)
    parser.add_argument('--fluid-origin', nargs=3, type=float)
    parser.add_argument('--rho', type=float)
    parser.add_argument('--mu', type=float)
    parser.add_argument('--kappa', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--interaction-quadrature',choices=('fixed','adaptive'))
    parser.add_argument('--ib-rule-family',choices=('conical','xiao-gimbutas'))
    parser.add_argument('--ib-transfer-backend',choices=('reference','fused','cell'),
                        help='adaptive FE/IB execution; fused/cell are experimental GPU paths')
    parser.add_argument('--ib-stencil-backend',choices=('component','shared'),
                        help='adaptive reference transfer: per-component or shared face/center tables')
    parser.add_argument('--stokes-warm-start',action=argparse.BooleanOptionalAction,default=None,
                        help='reuse successful nonlinear pressures within each midpoint solve')
    parser.add_argument('--ib-prepare-backend',choices=('torch','triton'),
                        help='shared adaptive templates: reference tensors or direct P1/Peskin CUDA preparation')
    parser.add_argument('--ib-point-density',type=float,help='adaptive Gaussian density parameter, >=2')
    parser.add_argument('--reference', action='store_true', help='torch/PCG reference execution for small CPU tests')
    parser.add_argument('--no-vtk', action='store_true')
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--output-every', type=int)
    parser.add_argument('--checkpoint-every', type=int)
    args = parser.parse_args(argv)
    if args.end_time is not None and args.cycles is not None:
        parser.error('choose --end-time or --cycles')
    if args.cycles is not None and args.cycles < 1:
        parser.error('--cycles must be a positive integer')
    end_time = args.end_time if args.cycles is None else .8*args.cycles
    if args.resume_dt is not None and not args.resume:
        parser.error('--resume-dt requires --resume')
    if args.resume:
        overrides = (args.config, args.mesh_dir, args.dt, args.fluid_cells, args.fluid_lengths,
                     args.fluid_origin, args.rho, args.mu, args.kappa, args.beta,
                     args.log_every, args.output_every, args.checkpoint_every, args.write_config)
        overrides += (args.interaction_quadrature,args.ib_point_density,args.ib_rule_family,args.ib_transfer_backend,
                      args.ib_stencil_backend,args.stokes_warm_start,args.ib_prepare_backend)
        if any(v is not None for v in overrides) or args.reference or args.no_vtk:
            parser.error('resume restores settings; only device, output, end time/cycles, --resume-dt and --coupling may change')
        return run(device=args.device, output=args.output, resume=args.resume,
                   end_time=end_time, resume_dt=args.resume_dt, coupling_scheme=args.coupling,
                   nonlinear_solver=args.nonlinear_solver)
    config = load_config(args.config, CONFIG) if args.config else CONFIG
    config = replace(config,
        interaction_quadrature=replace(config.interaction_quadrature,
            **{k:v for k,v in dict(mode=args.interaction_quadrature,point_density=args.ib_point_density,
                                 rule_family=args.ib_rule_family,transfer_backend=args.ib_transfer_backend,
                                 stencil_backend=args.ib_stencil_backend,prepare_backend=args.ib_prepare_backend).items() if v is not None}),
        coupling=replace(config.coupling,scheme=config.coupling.scheme if args.coupling is None else args.coupling,
            semiimplicit_solver=config.coupling.semiimplicit_solver if args.nonlinear_solver is None else args.nonlinear_solver,
            stokes_warm_start=config.coupling.stokes_warm_start if args.stokes_warm_start is None else args.stokes_warm_start),
        source_dir=config.source_dir if args.mesh_dir is None else args.mesh_dir,
        time=TimeConfig(config.time.dt if args.dt is None else args.dt,
                        config.time.end_time if end_time is None else end_time),
        material=replace(config.material, kappa=config.material.kappa if args.kappa is None else args.kappa),
        beta=config.beta if args.beta is None else args.beta,
        fluid=replace(config.fluid, **{k: v for k, v in dict(
            shape=(args.fluid_cells,)*3 if args.fluid_cells is not None else None,
            lengths=args.fluid_lengths, origin=args.fluid_origin, rho=args.rho, mu=args.mu).items() if v is not None}),
        execution=LVExecutionConfig(warm_start=True) if args.reference else config.execution,
        output=replace(config.output, **{k: v for k, v in dict(
            write_vtk=False if args.no_vtk else None, log_every=args.log_every,
            output_every=args.output_every, checkpoint_every=args.checkpoint_every).items() if v is not None}))
    if config.coupling.stokes_warm_start and config.coupling.scheme!='cnab-semiimplicit':
        parser.error('--stokes-warm-start requires cnab-semiimplicit')
    if args.write_config:
        save_config(args.write_config, config)
        return
    if args.nonlinear_solver is not None and config.coupling.scheme!='cnab-semiimplicit':
        parser.error('--nonlinear-solver requires cnab-semiimplicit')
    report = run(case_config=config, device=args.device, output=args.output)
    print(f'completed: step={report["accepted_steps"]}, t={report["reached_time_s"]:.6f} s, '
          f'elapsed_seconds={report["elapsed_seconds"]:.3f}', flush=True)
    return report


if __name__ == '__main__':
    main()
