"""Real-LV BE-BE: paper passive inflation or prescribed active cycles."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from afsi_torch.paper_lv import PaperLVConfig
from afsi_torch.config import TimeConfig, LVExecutionConfig, load_config, save_config
from afsi_torch.mac.adaptive_transfer import InteractionQuadratureOptions
from afsi_torch.simulation.paper_lv_mac import run

CONFIG = PaperLVConfig()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', help='partial paper configuration JSON')
    p.add_argument('--write-config', help='write effective configuration and exit')
    p.add_argument('--mesh-dir')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output')
    p.add_argument('--resume', help='paper BE-BE checkpoint only; old CNAB checkpoints are incompatible')
    p.add_argument('--dt', type=float)
    p.add_argument('--end-time', type=float)
    p.add_argument('--load-protocol', choices=('inflation', 'active-cycle'))
    p.add_argument('--cycles', type=int, help='number of 0.8 s active cycles; alternative to --end-time')
    p.add_argument('--fluid-cells', type=int)
    p.add_argument('--fluid-lengths', nargs=3, type=float)
    p.add_argument('--fluid-origin', nargs=3, type=float)
    p.add_argument('--rho', type=float)
    p.add_argument('--mu', type=float)
    p.add_argument('--kappa', type=float)
    p.add_argument('--beta', type=float)
    p.add_argument('--nonlinear-solver', choices=('jfnk', 'newton', 'anderson-newton'),
                   help='solver-only selection; allowed on resume')
    p.add_argument('--anderson-policy', choices=('legacy','adaptive'), help='solver-only policy; allowed on resume')
    p.add_argument('--anderson-max-iterations', type=int,
                   help='Anderson trial budget before Newton; adaptive policy may extend it; allowed on resume')
    p.add_argument('--newton-preconditioner', choices=('none','solid-block'), help='solver-only option; allowed on resume')
    p.add_argument('--linear-policy', choices=('reference','estimated','inexact'), help='solver-only option; allowed on resume')
    p.add_argument('--ib-response-backend', choices=('quadrature','csr'), help='equivalent frozen IB execution; allowed on resume')
    p.add_argument('--ib-csr-assembly-backend', choices=('coalesce','hash'), help='GPU CSR construction; allowed on resume')
    p.add_argument('--support-backend', choices=('points', 'vertices'))
    p.add_argument('--helmholtz-backend', choices=('reference', 'workspace', 'graph'))
    p.add_argument('--ib-point-density', type=float)
    p.add_argument('--ib-rule-family', choices=('conical', 'xiao-gimbutas'))
    p.add_argument('--ib-transfer-backend', choices=('reference', 'fused', 'cell'))
    p.add_argument('--ib-stencil-backend', choices=('component', 'shared'))
    p.add_argument('--ib-prepare-backend', choices=('torch', 'triton'))
    p.add_argument('--ib-shared-execution', choices=('reference', 'vector', 'reduced'))
    p.add_argument('--reuse-ib-buffers', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--interaction-quadrature', choices=('fixed', 'adaptive'))
    p.add_argument('--pressure-backend', choices=('torch', 'fused', 'workspace', 'graph'))
    p.add_argument('--no-convection', action='store_true', help='test eq43 without convection; not the cardiac reproduction')
    p.add_argument('--reference', action='store_true', help='eager tensor/PCG execution, useful for small CPU tests')
    p.add_argument('--no-vtk', action='store_true')
    for key in ('log-every', 'output-every', 'checkpoint-every'):
        p.add_argument('--'+key, type=int)
    args = p.parse_args(argv)
    if args.anderson_max_iterations is not None and args.anderson_max_iterations<1:
        p.error('--anderson-max-iterations must be a positive integer')
    if args.resume:
        forbidden = (args.config, args.write_config, args.mesh_dir, args.dt, args.fluid_cells,
            args.fluid_lengths, args.fluid_origin, args.rho, args.mu, args.kappa, args.beta,
            args.load_protocol, args.cycles,
            args.support_backend, args.helmholtz_backend,
            args.ib_point_density, args.interaction_quadrature, args.pressure_backend,
            args.ib_rule_family, args.ib_transfer_backend, args.ib_stencil_backend, args.ib_prepare_backend,
            args.ib_shared_execution, args.reuse_ib_buffers,
            args.log_every, args.output_every, args.checkpoint_every)
        if any(v is not None for v in forbidden) or args.reference or args.no_vtk or args.no_convection:
            p.error('resume restores paper physics/execution; only end time, original output and solver-only overrides may be supplied')
        report = run(device=args.device, output=args.output, resume=args.resume, end_time=args.end_time,
                     anderson_policy=args.anderson_policy,newton_preconditioner=args.newton_preconditioner,
                     linear_policy=args.linear_policy,ib_response_backend=args.ib_response_backend,
                     ib_csr_assembly_backend=args.ib_csr_assembly_backend,
                     anderson_max_iterations=args.anderson_max_iterations,
                     nonlinear_solver=args.nonlinear_solver)
    else:
        config = load_config(args.config, CONFIG) if args.config else CONFIG
        protocol = config.load_protocol if args.load_protocol is None else args.load_protocol
        end_time = config.time.end_time if args.end_time is None else args.end_time
        if args.cycles is not None:
            if protocol!='active-cycle' or args.cycles<1 or args.end_time is not None:
                p.error('--cycles requires active-cycle, a positive integer, and no --end-time')
            end_time = args.cycles*config.cyclic_loads.period
        execution = LVExecutionConfig(warm_start=True) if args.reference else config.execution
        if args.pressure_backend is not None:
            execution = replace(execution, pressure_backend=args.pressure_backend)
        quadrature = config.interaction_quadrature
        if args.reference:
            quadrature = replace(quadrature, transfer_backend='reference', prepare_backend='torch', reuse_stencil_buffers=False)
        if args.interaction_quadrature == 'fixed':
            quadrature = InteractionQuadratureOptions(mode='fixed')
        else:
            quadrature = replace(quadrature, **{k:v for k,v in dict(
                mode=args.interaction_quadrature, point_density=args.ib_point_density,
                rule_family=args.ib_rule_family, transfer_backend=args.ib_transfer_backend,
                stencil_backend=args.ib_stencil_backend, prepare_backend=args.ib_prepare_backend,
                shared_execution=args.ib_shared_execution, reuse_stencil_buffers=args.reuse_ib_buffers).items() if v is not None})
        config = replace(config,
            source_dir=config.source_dir if args.mesh_dir is None else args.mesh_dir,
            load_protocol=protocol,
            time=TimeConfig(config.time.dt if args.dt is None else args.dt,end_time),
            fluid=replace(config.fluid, **{k:v for k,v in dict(
                shape=(args.fluid_cells,)*3 if args.fluid_cells else None, lengths=args.fluid_lengths,
                origin=args.fluid_origin, rho=args.rho, mu=args.mu).items() if v is not None}),
            material=replace(config.material, kappa=config.material.kappa if args.kappa is None else args.kappa),
            beta=config.beta if args.beta is None else args.beta,
            nonlinear_solver=config.nonlinear_solver if args.nonlinear_solver is None else args.nonlinear_solver,
            support_backend=config.support_backend if args.support_backend is None else args.support_backend,
            flow=replace(config.flow, **{k:v for k,v in dict(
                convection=False if args.no_convection else None,
                helmholtz_backend=args.helmholtz_backend).items() if v is not None}),
            execution=execution, interaction_quadrature=quadrature,
            output=replace(config.output, **{k:v for k,v in dict(
                log_every=args.log_every, output_every=args.output_every, checkpoint_every=args.checkpoint_every,
                write_vtk=False if args.no_vtk else None).items() if v is not None}))
        if args.anderson_policy is not None:
            from afsi_torch.mac.midpoint_solver import anderson_policy
            config = replace(config,anderson=anderson_policy(config.anderson,args.anderson_policy))
        if args.anderson_max_iterations is not None:
            config = replace(config,anderson=replace(config.anderson,max_iterations=args.anderson_max_iterations))
        if args.newton_preconditioner is not None:
            config = replace(config,newton_preconditioner=args.newton_preconditioner)
        if args.linear_policy is not None:
            from afsi_torch.nonlinear import coupled_linear_policy
            config = replace(config,nonlinear=coupled_linear_policy(config.nonlinear,args.linear_policy))
        if args.ib_response_backend is not None:
            config = replace(config,ib_response_backend=args.ib_response_backend)
        if args.ib_csr_assembly_backend is not None:
            config = replace(config,ib_csr_assembly_backend=args.ib_csr_assembly_backend)
        if args.write_config:
            save_config(args.write_config, config)
            return
        report = run(case_config=config, device=args.device, output=args.output)
    print(f'completed: step={report["accepted_steps"]}, t={report["reached_time_s"]:.6f} s, '
          f'elapsed_seconds={report["elapsed_seconds"]:.3f}', flush=True)
    return report


if __name__ == '__main__':
    main()
