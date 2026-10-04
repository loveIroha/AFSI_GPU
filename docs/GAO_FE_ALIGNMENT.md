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
Positive Gauss–Jacobi conical rules provide degree 2*n−1; order 3 uses the
existing positive degree-5 Keast rule. These are valid degree-matched rules,
not a copy of libMesh's exact point tables. Order 2 has eight points rather
than the old degree-2 four-point rule; order 3 has 15 points; higher orders
have n³ points. This criterion controls sampling but does not prove FSI
stability or grid convergence.

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

The fresh `demo/real_lv_fsi/run_mac.py` defaults to adaptive quadrature.
`RealLVConfig()` and legacy checkpoints retain fixed quadrature for backward
compatibility. New configuration/checkpoints store the choice explicitly.

```json
{
  "interaction_quadrature": {
    "mode": "adaptive",
    "point_density": 2.0,
    "max_order": 8,
    "max_points": 12000000
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
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_adaptive_p1_transfer.py

CUDA_VISIBLE_DEVICES=0 python -u demo/real_lv_fsi/run_mac.py \
  --device cuda --interaction-quadrature adaptive --ib-point-density 2 \
  --dt 1e-4 --end-time 0.005 --output results/real_lv_adaptive_early
```

For a controlled comparison, start a separate fresh run with identical
material, fluid, load and time settings and `--interaction-quadrature fixed`.
Compare point count, power error, deformation, velocity and wall-clock time.
An old fixed-rule checkpoint resumes its fixed rule; it is not automatically
converted to the new trajectory. The CSV reports interaction point counts;
the JSON reports the last prepared stencil's order histogram.

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
