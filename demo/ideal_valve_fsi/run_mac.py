"""Generated 2D ideal valve aligned with AFSI demo_340; default horizon 3 s."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from examples.valve_mac import main

if __name__=='__main__':
    main(default_end_time=3.,default_output='results/demo_ideal_valve/mac')
