"""Failure-only local force/velocity diagnostics; no extra per-step kernels."""
import torch
from . import boundary as bd
from .holzapfel_ogden import ho_pk1


def velocity_peaks(grid, velocity):
    peaks = []
    for c,u in enumerate(velocity):
        flat = u.abs().argmax().item()
        shape = u.shape
        i, rem = divmod(flat, shape[1]*shape[2])
        j, k = divmod(rem, shape[2])
        index = (i,j,k)
        peaks.append(dict(component='xyz'[c], value_cm_per_s=u[index].item(),
            max_abs_cm_per_s=u[index].abs().item(), rms_cm_per_s=u.square().mean().sqrt().item(),
            index=list(index), location_cm=[o+h*n for o,h,n in zip(grid.face_origin(c),grid.spacing,index)]))
    return peaks


@torch.no_grad()
def coupled_failure_diagnostics(model, grid, state, problem):
    evaluation = problem.last_evaluation
    report = dict(accepted_step=state.step, accepted_time_s=state.time,
        accepted_solid=model.diagnostics(state.x), accepted_velocity_peaks=velocity_peaks(grid,state.velocity),
        trial_status='last residual evaluation; may be rejected; not a checkpoint state',
        residual_evaluations=problem.evaluations, jacobian_actions=problem.actions,
        tangent_assemblies=problem.assemblies)
    if evaluation is None:
        return report
    r, flow, force, density, _, _, _ = evaluation
    x = problem.last_midpoint
    F = model.element_gradient(x)
    J = torch.linalg.det(F)
    pressure, tension = model.loads.at(problem.half_time)
    def assemble(P):
        local = -model.volumes[:,None,None]*torch.einsum('eiJ,eaJ->eai',P,model.gradients)
        return torch.zeros_like(x).index_add(0,model.mesh.cells.reshape(-1),local.reshape(-1,3))
    P = ho_pk1(F,model.mesh.fiber,model.mesh.sheet,model.parameters,0.)
    Pvol = 2*model.parameters.kappa*torch.log(J)[:,None,None]*torch.linalg.inv(F).transpose(-1,-2)
    Ff = torch.einsum('eij,ej->ei',F,model.mesh.fiber)
    factor = tension*(1+model.parameters.active_stretch_slope*(torch.linalg.vector_norm(Ff,dim=-1)-1))
    Pactive = factor[:,None,None]*Ff[:,:,None]*model.mesh.fiber[:,None,:]
    components = dict(total=force, passive_without_volume=assemble(P-Pvol), volume=assemble(Pvol),
        active=assemble(Pactive), follower=bd.pressure_force(x,model.endo,pressure), basal=model.basal_force(x))
    report.update(trial_time_s=problem.half_time, trial_midpoint_solid=model.diagnostics(x),
        trial_velocity_peaks=velocity_peaks(grid,flow.velocity),
        trial_kinematic_residual_cm=torch.linalg.vector_norm(r).item(),
        trial_stokes=flow.diagnostics, force_components={})
    for name,values in components.items():
        norms = torch.linalg.vector_norm(values,dim=-1)
        node = norms.argmax().item()
        report['force_components'][name] = dict(norm_dyn=torch.linalg.vector_norm(values).item(),
            maximum_nodal_norm_dyn=norms[node].item(), reference_node=node,
            location_cm=x[node].tolist(), resultant_dyn=values.sum(0).tolist())
    mismatch = force-sum(values for name,values in components.items() if name!='total')
    report['force_decomposition_max_abs_dyn'] = mismatch.abs().max().item()
    peaks = torch.tensor([p['location_cm'] for p in report['trial_velocity_peaks']],device=x.device,dtype=x.dtype)
    centers = x[model.mesh.cells].mean(1)
    report['cells_near_velocity_peaks'] = []
    for point in peaks:
        cell = (centers-point).square().sum(-1).argmin().item()
        report['cells_near_velocity_peaks'].append(dict(cell=cell, center_cm=centers[cell].tolist(),
            detF=J[cell].item(), reference_volume_cm3=model.volumes[cell].item(),
            distance_cm=torch.linalg.vector_norm(centers[cell]-point).item()))
    return report
