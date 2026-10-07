"""Run a declared demo JSON: python demo/run.py path/to/config.json [overrides]."""
import argparse
import json
from pathlib import Path
import runpy

ENTRIES={'ideal-lv-mac':'ideal_lv_fsi/run_mac.py',
         'ideal-lv-fem':'ideal_lv_fsi/run_fem.py',
         'ideal-valve-mac':'ideal_valve_fsi/run_mac.py',
         'real-lv':'real_lv_fsi/run_mac.py'}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',help='JSON containing demo and case parameters')
    args,overrides=p.parse_known_args(argv)
    path=Path(args.config).resolve()
    with path.open(encoding='utf-8') as stream:
        config=json.load(stream)
    if not isinstance(config,dict) or config.get('demo') not in ENTRIES:
        p.error('JSON requires demo: '+', '.join(ENTRIES))
    option_names={arg.split('=',1)[0] for arg in overrides if arg.startswith('--')}
    if option_names & {'--config','--resume'}:
        p.error('do not override the config; use the case entry point for checkpoint resume')
    entry=Path(__file__).parent/ENTRIES[config['demo']]
    if '--output' not in option_names:
        # Distinct fresh run even when the same JSON is launched twice.
        from datetime import datetime
        destination=Path('results')/f'{path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}'
        overrides.extend(['--output',str(destination)])
        print(f'Output directory: {destination}',flush=True)
    return runpy.run_path(str(entry),run_name='afsi_demo')['main'](['--config',str(path),*overrides])

if __name__=='__main__':
    main()
