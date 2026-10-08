# AFSI demo_337 CPU / MPI reference run

Run these commands on the Linux **host**, using the existing `afsi_dev_ljy`
container. No host Conda environment is needed. The AFSI input directory must
already exist at `/root/afsi-data/337_ideal_left_ventricle`, as in the previous
single-process run. Its mesh and fibers should be the same generated reference
used by the GPU case; this launch script does not replace them.

```bash
cd /mnt/large2/gjh/AFSI_GPU
git pull --ff-only
AFSI337_NP=8 AFSI337_OUTPUT_EVERY=400 bash scripts/run_afsi337_mpi_container.sh
```

The script first checks native source layout and the live 3D C++ MPI IB transfer
against independent interpolation/spreading formulas. A failed check prevents
the full run. After `Started AFSI demo_337`, the solve runs detached in Docker;
closing the terminal does not stop it. It uses 8 MPI ranks and 1 numerical-library
thread per rank. DOLFINx/PETSc FEM operations are distributed; native IB gathers
owned values to rank 0, applies its C++ kernel, then scatters owned results.
This is not an 8-way parallel IB kernel.

Physics follows the native demo: 2 s / 40000 steps, dt=5e-5 s, Q2/Q1 fluid on
32^3 hexahedra in a 5 cm box, P2 solid, Guccione, beta=kappa=5e5, pressure ramp
to 150000 dyn/cm^2 and active tension ramp to 600000 dyn/cm^2 over 1.5 s.
The runner adds forward ghost synchronization after IB interpolation/spreading,
global reduction of the native maximum signed velocity component, and local
CSV reporting in place of SwanLab. It checks PETSc convergence.

Output is every 400 accepted steps (0.02 s) and at the final step, matching the
GPU output cadence rather than the native 10000 fps. Native per-step scalar
diagnostics are retained. Native XDMF timestamps use force/source time t_n:
the corresponding accepted state is t_n+dt; history records both clocks.
XDMF stores P1 interpolated visualization fields from P2 solutions; `solid_coords_io`
is current position and must not be treated as displacement or added to the
mesh through Warp By Vector. Native `volume` is myocardial wall volume, not
LV cavity volume. Native `u_max` is a signed component maximum, not speed magnitude.

All output is in a unique directory:

```text
/root/afsi-data/afsi337_runlog/<timestamp>-np8-<pid>/
    run.log
    runtime.txt         # Python setup/solve/output wall time, written on exit
    mpi_runtime.txt     # launcher exit status and wall time, including startup
    history.csv
    config.json
    report.json
    mpi_check.json
    systole-afsi.txt    # original final section postprocessing
    fields/
        velocity.xdmf / velocity.h5
        solid_force.xdmf / solid_force.h5
```

With the existing `/mnt/large2/qwer -> /root` bind mount, this is also visible
on the host under `/mnt/large2/qwer/afsi-data/afsi337_runlog/<run-id>/`.

Monitor the newest MPI run (or replace with the exact printed directory):

```bash
run_dir=$(docker exec afsi_dev_ljy python -c 'from pathlib import Path; p=Path("/root/afsi-data/afsi337_runlog"); print(max((d for d in p.iterdir() if d.is_dir() and "-np8-" in d.name), key=lambda d:d.name))')
docker exec afsi_dev_ljy tail -f "$run_dir/run.log"
docker top afsi_dev_ljy
```

After completion, copy the entire run, including both XDMF and HDF5:

```bash
run_name=${run_dir##*/}
destination="/mnt/large2/gjh/afsi_result/ideal_lv_${run_name}"
mkdir -p "$destination"
docker cp "afsi_dev_ljy:${run_dir}/." "$destination/"
cat "$destination/runtime.txt"
cat "$destination/mpi_runtime.txt"
```

Do not copy changing HDF5 output before the solve exits. Check `exit_code=0`
and `report.json` completion before treating the run as a completed reference.
The native source directory is left unchanged. Input conversion, preflight,
and result copying are outside the reported solve time.
