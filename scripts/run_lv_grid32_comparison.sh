#!/usr/bin/env bash
# New MAC32 on one GPU; FEM32 then grad-div FEM32 on a second GPU.
set -euo pipefail

run_one() {
    local name=$1 gpu=$2 entry=$3
    local out="$batch/$name"
    printf 'running\n' > "$out/status.txt"
    date -Iseconds > "$out/started_at.txt"
    local code=0
    CUDA_VISIBLE_DEVICES="$gpu" /usr/bin/time \
        -f 'elapsed_seconds=%e exit_code=%x' -o "$out/runtime.txt" \
        "$python" -u "$entry" --config "$out/config.json" \
        --device cuda --output "$out/simulation" \
        > "$out/run.log" 2>&1 < /dev/null || code=$?
    date -Iseconds > "$out/finished_at.txt"
    if (( code == 0 )); then
        printf 'completed\n' > "$out/status.txt"
    else
        printf 'failed exit_code=%s\n' "$code" > "$out/status.txt"
    fi
    # A failed FEM case must not prevent the independent grad-div case.
    return 0
}

if [[ ${1:-} == --worker ]]; then
    repo=$2 batch=$3 queue=$4 gpu=$5 python=$6
    cd "$repo"
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
    if [[ $queue == mac ]]; then
        run_one mac32 "$gpu" demo/ideal_lv_fsi/run_mac.py
    else
        run_one fem32 "$gpu" demo/ideal_lv_fsi/run_fem.py
        run_one fem32_grad_div "$gpu" demo/AFSI337_stable/run.py
    fi
    exit 0
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
python=$(command -v python)
mac_gpu=${AFSI_MAC_GPU:-0}
fem_gpu=${AFSI_FEM_GPU:-1}
[[ $mac_gpu != "$fem_gpu" ]] || { echo 'Choose two different GPUs.' >&2; exit 1; }
[[ -x /usr/bin/time ]] || { echo '/usr/bin/time is required.' >&2; exit 1; }

# Snapshot and validate every case before detaching either worker.
batch=$("$python" - "$repo" "${1:-}" "$mac_gpu" "$fem_gpu" <<'PY'
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import torch
from afsi_torch.config import LVSimulationConfig, LVFEMSimulationConfig, load_config

repo = Path(sys.argv[1])
source = repo / os.environ.get('AFSI_SOLID_INPUT', 'results/afsi337_reference/solid.npz')
if not source.is_file():
    raise SystemExit(f'Missing shared solid input: {source}')
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the selected Python environment.')
gpus = [sys.argv[3], sys.argv[4]]
available = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).split()
if any(gpu not in available for gpu in gpus):
    raise SystemExit(f'GPU indices {gpus} are not in {available}')
batch = Path(sys.argv[2] or f'results/lv_grid32_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
batch.mkdir(parents=True, exist_ok=False)
cases = [
    ('mac32', 'demo/ideal_lv_fsi/configs/mac_gl2017_32.json', LVSimulationConfig),
    ('fem32', 'demo/ideal_lv_fsi/configs/afsi337_fem.json', LVFEMSimulationConfig),
    ('fem32_grad_div', 'demo/AFSI337_stable/config.json', LVFEMSimulationConfig),
]
physical = None
for name, filename, cls in cases:
    data = json.loads((repo / filename).read_text())
    data['solid_input'] = str(source.resolve())
    data['output'].update(log_every=200, checkpoint_every=1000, output_every=400,
                          write_vtk=True, fluid_fields=True)
    fields = {k: copy.deepcopy(data[k]) for k in ('time', 'geometry', 'material', 'loads', 'beta', 'basal_constraint')}
    fields['fluid'] = {k: data['fluid'][k] for k in ('shape', 'lengths', 'origin', 'rho', 'mu')}
    if physical is None:
        physical = fields
    elif fields != physical:
        raise SystemExit(f'Physical settings differ: {name}')
    assert data['fluid']['shape'] == [32, 32, 32]
    out = batch / name
    out.mkdir()
    path = out / 'config.json'
    path.write_text(json.dumps(data, indent=2) + '\n')
    load_config(path, cls)
    (out / 'status.txt').write_text('queued\n')
manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    solid_input=str(source.resolve()), solid_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    mac_gpu=gpus[0], fem_gpu=gpus[1], physical_settings=physical,
    fem_grad_div_alpha=1.0, queue=dict(mac=['mac32'], fem=['fem32', 'fem32_grad_div']),
    note='32^3 MAC cells align element counts; spacing is twice the 32^3 Q2 FEM velocity-node spacing.')
(batch / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(batch)
PY
)

script="$repo/scripts/run_lv_grid32_comparison.sh"
nohup bash "$script" --worker "$repo" "$batch" mac "$mac_gpu" "$python" \
    > "$batch/mac_queue.log" 2>&1 < /dev/null &
echo $! > "$batch/mac_worker.pid"
nohup bash "$script" --worker "$repo" "$batch" fem "$fem_gpu" "$python" \
    > "$batch/fem_queue.log" 2>&1 < /dev/null &
echo $! > "$batch/fem_worker.pid"
echo "Batch: $batch"
echo "GPU $mac_gpu: mac32; GPU $fem_gpu: fem32 -> fem32_grad_div"
echo "Status: cat $batch/{mac32,fem32,fem32_grad_div}/status.txt"
echo 'Workers are detached; closing the terminal does not stop the jobs.'
