"""Export one saved 3D MAC state to ParaView without changing its checkpoint."""
import argparse
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from afsi_torch.mac.checkpoint import load_mac
from afsi_torch.config import lv_grid
from afsi_torch.mac.output import MACWriter


def export(checkpoint,output,device='cpu'):
    model,state,settings,_=load_mac(checkpoint,device)
    grid=lv_grid(settings)
    output=Path(output)
    if any(output.glob('*.pvd')) or any(output.glob('*.vtu')) or any(output.glob('*.vti')):
        raise FileExistsError('choose an empty visualization output directory')
    writer=MACWriter(output,model,grid)
    writer.write(state)
    print(f'Exported step={state.step}, t={state.time:g} s: {output.resolve()}/solid.pvd and fluid.pvd')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    export(args.checkpoint,args.output,args.device)


if __name__=='__main__':
    main()
