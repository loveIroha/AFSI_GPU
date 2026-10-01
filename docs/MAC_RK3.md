# Explicit FE/IB with projected RK3 fluid

The real-LV demo accepts `--coupling explicit-rk3`. This is a lower-cost
alternative to `implicit-newton`: one solid-force update, two consistent FE
mass solves and three fluid pressure solves per step when diagnostics are off.
Diagnostic power sampling may add work in other drivers. There is no coupled
Newton iteration or solid tangent assembly in this mode.

## Method and order

Keep the existing AFSI-style lagged force and FE kinematics. At x_n, prepare
the adjoint IB pair, spread the stored FE force once, and hold its Eulerian
density f fixed during the fluid step. With

```text
E(u) = P[u + dt*(-N_h(u) + nu*L_h(u) + f/rho)],
u1 = E(u_n),
u2 = 3/4*u_n + 1/4*E(u1),
u_new = 1/3*u_n + 2/3*E(u2),
x_new = x_n + dt*J_n*u_new,
b_new = b(x_new,t_n).
```

Each E includes a converged pressure Poisson solve. The pressure stored for
output and warm starts is the RK-weighted average of the three multipliers
with weights (1/6,1/6,2/3), which satisfies the integrated momentum equation.
It is not an independently solved endpoint pressure. Reports retain every
stage's true pressure residual and tolerance. `pressure.cycles` is the sum
over all three solves, and the combined pressure residual is reported as a
weighted upper bound, not as an uncomputed true residual.

The autonomous fluid subproblem uses the standard
[SSPRK(3,3) method](https://ketch.github.io/numipedia/methods/SSPRK33.html).
The complete FE/IB scheme remains first order because force and transfer
geometry are frozen and the solid uses end-velocity kinematics. No third-order
FSI claim is made. Centered conservative MAC fluxes, physical viscosity, the
Peskin kernel, CSR consistent mass, H-O law and prescribed loading remain the
same. GPU fusion, compact IB, mass graphs and pressure graphs are reused.

## Stability screen

Forward Euler with centered advection imposed the old A=dt*sum(U_c^2)/(2*nu)
screen. RK3 has stability polynomial 1+z+z^2/2+z^3/6 and a nonzero imaginary
stability interval; it does not require that Euler A bound. This is a change
in the actual time integration, not simply suppressing an exception.

Every stage checks CFL=sum(dt*max|u_c|/h_c)<=0.25 and
D=nu*dt*sum(h_c^-2)<=0.25. For frozen periodic scalar advection/diffusion,
these conservative limits place eigenvalues inside the rectangle
Re(z) in [-1,0], Im(z) in [-0.25,0.25] contained in the RK3 stability region.
See also [Gkeyll's RK stability discussion](https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html).
A and cell Reynolds number remain diagnostics. The SSP name alone does not
establish nonlinear monotonicity for centered spatial differences.

This screen does not certify the explicit elastic/IB update, nonlinear flow,
or the spatial resolution of the real mesh. Positive J, finite fields, IB
support and the existing per-step displacement bound remain enforced.
If strong contraction exposes an elastic time-step restriction, dt may still
need reduction. A three-cycle real-mesh GPU run has not yet validated this mode.

## Performance experiment

Use the same saved real-LV state for both methods. The benchmark performs no
per-step file output, excludes setup/warmup from the measured wall time, and
counts **all** pressure/mass solves (including those nested inside GMRES).
It performs CUDA synchronization only at timing boundaries in addition to the
solvers' existing convergence checks. The source checkpoint is read-only.

```bash
CUDA_VISIBLE_DEVICES=0 python -u validation/benchmark_real_lv_schemes.py \
  --checkpoint /path/to/real_lv/checkpoint.npz \
  --device cuda --warmup 2 --steps 10 \
  --output results/real_lv_scheme_performance/report.json
```

Run without another job competing for the same GPU. Ratios are measured only
for the checkpoint phase; early filling costs do not predict contraction costs.
This compares methods with different temporal errors, not algebraically
equivalent backends. No full-run speedup is asserted before GPU measurements.

To switch an existing run, specify a new output directory:

```bash
python -u demo/real_lv_fsi/run_mac.py --device cuda \
  --resume /path/to/real_lv/checkpoint.npz --coupling explicit-rk3 \
  --cycles 3 --output results/real_lv_rk3_branch
```

The checkpoint state and dt are retained; stored force is recomputed at the
lagged sampling time. Ordinary subsequent resume restores RK3 automatically.
For a uniform-method trajectory, start a new run from the reference mesh with
`--coupling explicit-rk3 --dt 1e-4 --cycles 3` instead.
