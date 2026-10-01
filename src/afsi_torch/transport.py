"""Host-side policy for explicit centered advection plus physical viscosity.

The frozen, constant-coefficient periodic model has the sufficient bounds
D=sum(nu*dt/h_i**2)<=1/2 and A=dt*sum(U_i**2)/(2*nu)<=1.
We use D<=1/4 for the closed-wall stencil and A<=1/4 as a safety margin,
alongside CFL<=1/4. This is a screening policy, not a nonlinear FSI theorem.
Cell Reynolds number remains a spatial-resolution diagnostic, not a veto.
"""
COURANT_LIMIT = .25
VISCOUS_LIMIT = .25
ADVECTION_DIFFUSION_LIMIT = .25
TRANSPORT_POLICY = 'centered-explicit-advection-diffusion-v1'


def implicit_policy():
    return dict(name='backward-euler-frozen-ib-newton-v1',
                courant_monitor_only=True, advection_diffusion_monitor_only=True,
                cell_reynolds_monitor_only=True,
                scope='implicit transport/solid force; step-frozen IB geometry; no unconditional nonlinear FSI claim')


def transport_policy():
    return dict(name=TRANSPORT_POLICY, courant_limit=COURANT_LIMIT,
                viscous_number_limit=VISCOUS_LIMIT,
                advection_diffusion_limit=ADVECTION_DIFFUSION_LIMIT,
                advection_diffusion_formula='dt*sum(component_max_abs_velocity**2)/(2*mu/rho)',
                cell_reynolds_monitor_only=True,
                scope='frozen-coefficient transport screen; not full nonlinear IB-FSI stability or spatial accuracy')


def transport_numbers(speeds, spacing, dt, nu):
    return dict(courant=sum(dt*u/h for u, h in zip(speeds, spacing)),
                cell_reynolds=max(u*h/nu for u, h in zip(speeds, spacing)),
                advection_diffusion_number=dt*sum(u*u for u in speeds)/(2*nu),
                viscous_number=dt*nu*sum(1/h**2 for h in spacing))


def transport_violations(courant, advection_diffusion_number, viscous_number):
    return [name for name, value, limit in
            (('courant', courant, COURANT_LIMIT),
             ('advection_diffusion', advection_diffusion_number, ADVECTION_DIFFUSION_LIMIT),
             ('viscous', viscous_number, VISCOUS_LIMIT)) if value > limit]
