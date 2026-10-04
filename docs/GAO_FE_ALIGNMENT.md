# Real-LV finite element interaction quadrature

The real-LV demo keeps the supplied H–O UFL, active stretch factor 4.9,
fiber/sheet fields, radial basal penalty, pressure/tension waveforms and
material parameters. It does **not** substitute Gao's constitutive stress
or activation model. The GPU pressure multigrid is unchanged.

## Numerical reference and scope

[Gao et al., Dynamic finite-strain modelling of the human left ventricle](https://pmc.ncbi.nlm.nih.gov/articles/PMC4816497/)
uses Gaussian interaction quadrature chosen with respect to the deformed
element and Eulerian spacing. We adopt this approach to address possible
undersampling of the regularized delta kernel by a fixed four-point rule.
This is a coupling integration change, not structural remeshing.

The order selector follows the first-order element criterion in
[IBTK getQuadratureKey](https://github.com/IBAMR/IBAMR/blob/master/ibtk/src/utilities/libmesh_utilities.cpp):

```text
n = max(2, ceil(point_density * maximum_deformed_edge_length / min(grid_spacing)))
quadrature_degree = 2*n - 1
```

The default density parameter is 2. Each affine P1 cell uses its own order.
Fresh demos use positive Xiao–Gimbutas tetrahedral rules from Basix 0.10
at degree 2*n−1. They use fewer points than the original conical construction:

| Order n | Degree | Conical/Keast points | Compact points |
| --- | --- | --- | --- |
| 2 | 3 | 8 | 6 |
| 3 | 5 | 15 | 14 |
| 4 | 7 | 64 | 31 |
| 5 | 9 | 125 | 57 |
| 6 | 11 | 216 | 95 |
| 7 | 13 | 343 | 146 |
| 8 | 15 | 512 | 214 |

Above degree 15 the compact family explicitly falls back to the original
positive conical rule at the full requested degree. No degree/density clipping
is introduced. The user's histogram (1634/123506/10268/22 cells at orders
3/4/5/6) requires 4,438,928 compact points versus 9,217,146 old points.
This 51.8% point reduction is not a measured whole-solver speedup.

Basix generates only small reference tables during preparation; the numerical
FE/IB operations remain PyTorch/GPU. Install `.[quadrature]` to enable this
family. The conical family has no Basix dependency. The rules have the same
polynomial exactness, but the regularized IB kernel is not a single polynomial
over a cell, so compact sampling changes the discrete IB operators. Compare
deformation, velocity and coupling errors as well as timing. The criterion
does not prove FSI stability or grid convergence.

## GPU finite element implementation

Solid weak-form forces and CSR tangents remain those of the supplied UFL.
For affine P1 cells with DG0 directions, stress and shape gradients are
constant within each cell: summing degree-5 reference weights gives the
same volume weak form without duplicating constitutive calculations.

Cell edge lengths and order selection use PyTorch on the tensor device.
Cells with equal order are grouped there. Reference Gaussian tables are
small CPU setup data. FE interpolation, reference weighting, nodal assembly,
Peskin stencils and spread/gather execute on the tensor device, with the
existing CUDA kernels for GPU spread/gather. Variable group sizes use
dynamic-shape compiled kernels. An order comparison and changed-rule
histogram require host synchronization; this feature adds work and is not
a promised speed improvement.

The consistent reference mass matrix M is still assembled once as CSR.
Degree 2 already integrates P1 N_a*N_b exactly; it need not be rebuilt when
interaction quadrature changes. For the selected B, W and K:

```text
M F = b
f = Kᵀ W B F / cell_volume
M U = Bᵀ W K u
```

W contains **reference** cell weights, not current-volume weights. Spread
and interpolation share the same quadrature and stencil, giving
b·U = cell_volume * f·u up to mass-solve errors. Each stencil owns its rule,
so a later preparation cannot alter an existing Newton action. The rule
and kernel are frozen at the predicted midpoint during that step's nonlinear
solve; they are reselected on subsequent preparations.

## Configuration, inspection and comparison

The fresh `demo/real_lv_fsi/run_mac.py` defaults to adaptive compact quadrature.
`RealLVConfig()` and legacy checkpoints retain fixed quadrature for backward
compatibility. New configuration/checkpoints store the choice explicitly.

```json
{
  "interaction_quadrature": {
    "mode": "adaptive",
    "point_density": 2.0,
    "max_order": 8,
    "max_points": 12000000,
    "rule_family": "xiao-gimbutas"
  }
}
```

Caps are resource guards, not permission to reduce the required density.
The run stops if a required order or point count exceeds them. Inspect the
reference mesh first, without allocating IB stencils or advancing fluid:

```bash
python -u validation/inspect_real_lv_interaction.py \
  --mesh-dir /mnt/large2/gjh/realistic_left_ventricle \
  --output results/real_lv_adaptive_plan.json
```

This reports the initial rule histogram and exact initial point count.
Deformation can increase both. If a cap is exceeded, assess its memory/work
implications before raising it through `--config`; do not silently clip
density. `max_order` supports up to 22 (degree 43).

```bash
python -m pip install -e ".[test,mesh,fused,quadrature]"
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_mac_compact_quadrature.py

CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --interaction-quadrature adaptive --ib-point-density 2 \
  --ib-rule-family xiao-gimbutas \
  --dt 1e-4 --end-time 0.005 --output results/real_lv_compact_early
```

For a controlled comparison, start a separate fresh run with identical
material, fluid, load and time settings and `--interaction-quadrature fixed`.
Compare point count, power error, deformation, velocity and wall-clock time.
An old fixed-rule checkpoint resumes its fixed rule; it is not automatically
converted to the new trajectory. The CSV reports interaction point counts;
the JSON reports the last prepared stencil's order histogram.

An existing adaptive checkpoint without `rule_family` restores its original
conical rule. A normal resume does not silently convert its quadrature.
Fresh demos choose compact explicitly; use `--ib-rule-family conical` for
the previous family. Compare both using the embedded mesh in the existing
checkpoint, without modifying it or producing a long simulation:

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint results/real_lv_adaptive_early/checkpoint.npz \
  --schemes cnab-semiimplicit --nonlinear-solvers anderson-newton \
  --execution-variants baseline reuse compact \
  --device cuda --warmup 5 --steps 20 \
  --output results/real_lv_compact_performance/report.json
```

The variants respectively use conical/recompute, conical/reuse and compact/reuse.
All start from the same saved x/u/p and AB2 history. Reports include point
counts, final response reuse counts, warmed timings, solver counts, measured
speedups and endpoint differences. The benchmark does not save a continuation.

## Reusing the converged coupled response

`coupling.reuse_final_evaluation` defaults to true. The most recent nonlinear
residual evaluation retains its complete Stokes response, FE force, propagated
force density, interpolated velocity and actual residual. Final acceptance
can reuse that response only when the nodal unknown is exactly equal to a
copied snapshot and no intervening Stokes/linear action has invalidated it.
Different points, in-place mutations and overwritten workspaces cause a fresh
evaluation. A linear Jacobian action always invalidates eligibility.

Geometry validation, nonlinear tolerance, flow momentum/divergence checks,
endpoint force and CFL screens remain active. No approximate-point reuse or
tolerance relaxation is used. Reuse usually saves one Stokes solve and two
mass solves per step. For the measured AA=2 non-startup pattern, the intended
counts are Stokes 4→3, pressure 8→6 and mass 9→7; counts can differ with
convergence and startup. Set `reuse_final_evaluation:false` in JSON for a
recompute control. The CSV/nonlinear report records whether final reuse occurred.

## Remaining differences and validation limits

Gao's reported outer-box normal-traction/tangential-velocity condition is
different from our closed no-slip box. The default `cnab-semiimplicit`
solves nonlinear midpoint elasticity; it is an implicit elastic extension
of CN–AB2 and predicted-midpoint geometry, rather than a verbatim IBAMR
implementation. `cnab-midpoint` retains the explicit-force variant for
comparison. Our convection reconstruction is not claimed identical to
IBAMR's PPM implementation. These differences were not changed here.

The user's physical parameters, geometry and prescribed activation also
differ from Gao's research cases by design. Local tests cover Gaussian
polynomial exactness, reference mass invariance, force/torque/power balance,
affine velocity reproduction, CSR coupled Jacobian, independent supplied-UFL
stress, resource guards and restart. The real-mesh three-cycle GPU result
remains unverified; neither this change nor dt=1e-4 guarantees its stability.
