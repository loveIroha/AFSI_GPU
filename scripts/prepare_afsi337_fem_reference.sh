#!/usr/bin/env bash
# Host-side read-only native export: never starts or changes an AFSI simulation.
set -euo pipefail
container="${AFSI_CONTAINER:-afsi_dev_ljy}"
output="${1:-results/afsi337_reference}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$output"
for name in solid.npz force.npz zero.npz advecting.npz; do
  [[ ! -e "$output/$name" ]] || { echo "Existing reference: $output/$name; choose a new directory." >&2; exit 1; }
done
work="$(docker exec "$container" mktemp -d /tmp/afsi337-fem-reference.XXXXXX)"
for name in export_afsi337_solid export_afsi337_force_reference export_afsi_native_step; do
  docker cp "$root/validation/$name.py" "$container:$work/$name.py"
done
docker exec "$container" python "$work/export_afsi337_solid.py" --output "$work/solid.npz"
docker exec "$container" python "$work/export_afsi337_force_reference.py" --output "$work/force.npz"
docker exec "$container" python "$work/export_afsi_native_step.py" --output "$work/zero.npz"
docker exec "$container" python "$work/export_afsi_native_step.py" --initial-velocity-amplitude 2 --output "$work/advecting.npz"
for name in solid force zero advecting; do
  docker cp "$container:$work/$name.npz" "$output/$name.npz"
done
echo "Native solid, fiber/sheet, boundary tags, forces and Chorin/IB references: $output"
