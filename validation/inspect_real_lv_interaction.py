"""Read the real P1 LV and estimate adaptive IB quadrature; no fluid simulation."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.real_lv_fsi.run_mac import CONFIG
from afsi_torch.config import load_config
from afsi_torch.real_lv import imported_model
from afsi_torch.mac.grid import MACGrid
from afsi_torch.mac.adaptive_transfer import interaction_quadrature_plan
from afsi_torch.cycle_checkpoint import atomic_json


def inspect(config,device='cpu'):
    model = imported_model(config,device)
    grid = MACGrid(config.fluid.shape,config.fluid.lengths,config.fluid.origin)
    plan = interaction_quadrature_plan(model.mesh.X,model.mesh.cells,grid,config.interaction_quadrature)
    return dict(plan,material_model='user-HO-iso-I1-DG0',material_unchanged=True,
                fixed_degree_2_point_count=4*len(model.mesh.cells),simulation_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-dir')
    parser.add_argument('--config')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',default='results/real_lv_adaptive_plan.json')
    args = parser.parse_args()
    config = load_config(args.config,CONFIG) if args.config else CONFIG
    if args.mesh_dir is not None:
        config = replace(config,source_dir=args.mesh_dir)
    report = inspect(config,args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,report)
    print(json.dumps(report,indent=2))
    if not report['within_budget']:
        raise SystemExit('Required adaptive quadrature exceeds the configured budget. Inspect the report before changing limits.')


if __name__=='__main__':
    main()
