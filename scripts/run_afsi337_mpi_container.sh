#!/usr/bin/env bash
# Run on the Linux host. Docker detach survives logout.
set -euo pipefail
container=afsi_dev_ljy
ranks="${AFSI337_NP:-8}"
every="${AFSI337_OUTPUT_EVERY:-400}"
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || { echo 'AFSI337_NP must be positive' >&2; exit 1; }
[[ "$every" =~ ^[1-9][0-9]*$ ]] || { echo 'AFSI337_OUTPUT_EVERY must be positive' >&2; exit 1; }
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir=/root/afsi/afsic/demo/demo_337
input=/root/afsi-data/337_ideal_left_ventricle
run_id="$(date +%Y%m%d-%H%M%S)-np$ranks-$$"
logdir="/root/afsi-data/afsi337_runlog/$run_id"
docker start "$container" >/dev/null
docker exec "$container" python -c 'import dolfinx, afsic, mpi4py; print("AFSI environment ready")'
for path in "$workdir/fsi_paralell_fibers_contraction.py" "$workdir/data/ideal_middle_wall.txt" \
  "$input/f0.txt" "$input/s0.txt" "$input/cdm.txt" \
  "$input/lv_ellipsoid/geometry/mesh.xdmf" "$input/lv_ellipsoid/geometry/markers.json"; do
  docker exec "$container" test -f "$path" || { echo "Missing AFSI input: $path" >&2; exit 1; }
done
docker exec "$container" mkdir -p "$logdir"
docker cp "$repo_root/scripts/afsi337_mpi_runner.py" "$container:$logdir/offline_runner.py"
docker cp "$repo_root/validation/check_afsi337_mpi.py" "$container:$logdir/check_mpi.py"
docker exec -e AFSI337_DEMO="$workdir" -e AFSI337_LOGDIR="$logdir" -e AFSI337_OUTPUT_EVERY="$every" \
  "$container" python -c '
import importlib.util, os
from pathlib import Path
p=Path(os.environ["AFSI337_LOGDIR"])/"offline_runner.py"
s=importlib.util.spec_from_file_location("runner",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.offline_tree((Path(os.environ["AFSI337_DEMO"])/"fsi_paralell_fibers_contraction.py").read_text(),
              str(p.parent/"fields")+"/",p.parent.name,int(os.environ["AFSI337_OUTPUT_EVERY"]))
print("Native source checked; running 3D MPI IB preflight")'
docker exec -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  "$container" mpirun -np "$ranks" python -u "$logdir/check_mpi.py" \
  --input "$input/lv_ellipsoid/geometry/mesh.xdmf" --output "$logdir/mpi_check.json"
docker exec -d -w "$workdir" -e AFSI337_LOGDIR="$logdir" -e AFSI337_DEMO="$workdir" \
  -e AFSI337_NP="$ranks" -e AFSI337_OUTPUT_EVERY="$every" \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  "$container" bash -lc '
  echo $$ > "$AFSI337_LOGDIR/shell.pid"
  start=$(date +%s)
  mpirun -np "$AFSI337_NP" python -u "$AFSI337_LOGDIR/offline_runner.py" > "$AFSI337_LOGDIR/run.log" 2>&1 &
  task_pid=$!
  echo "$task_pid" > "$AFSI337_LOGDIR/launcher.pid"
  set +e
  wait "$task_pid"
  code=$?
  end=$(date +%s)
  printf "elapsed_seconds=%s exit_code=%s mpi_ranks=%s\n" "$((end-start))" "$code" "$AFSI337_NP" > "$AFSI337_LOGDIR/mpi_runtime.txt"
  exit "$code"
'
echo "Started AFSI demo_337: $ranks MPI ranks; 1 library thread each; 2 s / 40000 steps"
echo "Results: $logdir/fields"
echo "Log: $logdir/run.log"
echo "Runtime: $logdir/runtime.txt (written on normal exit); launcher status: mpi_runtime.txt"
echo "Monitor: docker exec $container tail -f $logdir/run.log"
echo "Stop: docker exec $container bash -c 'kill -INT \$(cat $logdir/launcher.pid)'"
