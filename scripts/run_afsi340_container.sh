#!/usr/bin/env bash
# Run from the Linux HOST, not from inside Docker or a FEniCS environment.
set -euo pipefail
container=afsi_dev_ljy
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source="${1:-$repo_root/results/demo_ideal_valve/mac/checkpoint.npz}"
workdir=/root/afsi/afsic/demo/demo_340
run_id="$(date +%Y%m%d-%H%M%S)-$$"
logdir="/root/afsi-data/afsi340_runlog/$run_id"
inputdir="$logdir/input"
test -f "$source" || { echo "Missing GPU checkpoint: $source" >&2; exit 1; }
docker start "$container" >/dev/null
docker exec "$container" python -c 'import dolfinx, afsic, numpy, basix, ufl, mpi4py; print("AFSI environment ready")'
for file in fsi_paralell.py FRH.py NeoHookean.py; do
  docker exec "$container" test -f "$workdir/$file" || { echo "Missing native demo: $workdir/$file" >&2; exit 1; }
done
docker exec "$container" mkdir -p "$logdir"
docker cp "$source" "$container:$logdir/source.npz"
docker cp "$repo_root/validation/write_afsi340_native_inputs.py" "$container:$logdir/write_inputs.py"
docker cp "$repo_root/scripts/afsi340_offline_runner.py" "$container:$logdir/offline_runner.py"
docker exec "$container" python "$logdir/write_inputs.py" --source "$logdir/source.npz" --output "$inputdir"
# Check source rewriting before detaching, so path/version errors remain visible.
docker exec -e AFSI340_LOGDIR="$logdir" -e AFSI340_DEMO="$workdir" -e AFSI340_INPUT="$inputdir/mesh-340.xdmf" "$container" python -c '
import importlib.util, os
from pathlib import Path
p=Path(os.environ["AFSI340_LOGDIR"])
s=importlib.util.spec_from_file_location("runner",p/"offline_runner.py")
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.offline_tree((Path(os.environ["AFSI340_DEMO"])/"fsi_paralell.py").read_text(),os.environ["AFSI340_INPUT"],str(p/"fields")+"/",p.name)
print("Native mesh and offline runner ready")'
docker exec -d -w "$workdir" -e AFSI340_LOGDIR="$logdir" -e AFSI340_DEMO="$workdir" \
  -e AFSI340_INPUT="$inputdir/mesh-340.xdmf" -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  "$container" bash -lc '
  echo $$ > "$AFSI340_LOGDIR/shell.pid"
  python -u "$AFSI340_LOGDIR/offline_runner.py" > "$AFSI340_LOGDIR/run.log" 2>&1 &
  task_pid=$!
  echo "$task_pid" > "$AFSI340_LOGDIR/python.pid"
  wait "$task_pid"
  '
echo "Started native AFSI demo_340 in $container (one MPI rank, 3 s / 48000 steps)"
echo "Log: $logdir/run.log"
echo "Runtime: $logdir/runtime.txt (written on exit)"
echo "Monitor: docker exec $container tail -f $logdir/run.log"
echo "Stop: docker exec $container bash -c 'kill -INT \$(cat $logdir/python.pid)'"
