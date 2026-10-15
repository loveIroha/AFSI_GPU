"""P2 adaptive execution: identical points, Peskin4 transfer, weights and mass."""
import torch
from copy import deepcopy
from .cuda_ib import require_cuda


def execution_settings(saved,backend):
    """Explicit execution-only override; preserve saved discretization/history."""
    if backend not in ('reference','cuda'):
        raise ValueError('P2 execution must be reference or cuda')
    settings=deepcopy(saved)
    if (settings.get('coupling',{}).get('scheme')!='cnab-midpoint' or
            settings.get('interaction_quadrature',{}).get('mode')!='adaptive'):
        raise ValueError('P2 execution override requires adaptive CN-AB2 midpoint IB')
    q=settings['interaction_quadrature']
    q.update(transfer_backend='reference',stencil_backend='component',prepare_backend='torch',
        reuse_stencil_buffers=False,shared_execution='reference')
    if backend=='cuda':
        if settings.get('ib_backend')!='cuda' or settings.get('execution_backend')!='fused':
            raise ValueError('P2 CUDA execution requires ib_backend=cuda and fused execution')
        q.update(transfer_backend='fused',stencil_backend='shared',prepare_backend='cuda',reuse_stencil_buffers=True)
    return settings


def prepare(transfer,x,rule,*,buffers=None):
    require_cuda(x)
    total=rule.point_count
    if buffers is None:
        base=torch.empty((2,total,3),device=x.device,dtype=torch.int64)
        phi=x.new_empty((2,total,3,4))
    else:
        base,phi=buffers
        if (base.shape!=(2,total,3) or phi.shape!=(2,total,3,4) or
                base.dtype!=torch.int64 or phi.dtype!=x.dtype or
                base.device!=x.device or phi.device!=x.device or
                not base.is_contiguous() or not phi.is_contiguous()):
            raise ValueError('invalid P2 stencil buffers')
    invalid=torch.zeros((),device=x.device,dtype=torch.int32)
    offset=0
    for group in rule.groups:
        torch.ops.afsi_ib_cuda.p2_prepare_(x.contiguous(),group.cells,group.values,
            transfer.origin,transfer.spacing,transfer.limits,base,phi,invalid,offset)
        offset+=group.weights.numel()
    if invalid.item():
        raise ValueError('MAC IB support reaches a wall or is nonfinite; enlarge/refine the fluid box')
    return base,phi


def spread(transfer,coefficient,stencil):
    fields=transfer.grid.zeros(device=coefficient.device,dtype=coefficient.dtype)
    offset=0
    for group in stencil.rule.groups:
        torch.ops.afsi_ib_cuda.p2_spread_(coefficient.contiguous(),group.cells,group.values,group.weights,
            stencil.base,stencil.phi,*fields,offset,*transfer.grid.shape,float(transfer.grid.volume))
        offset+=group.weights.numel()
    return fields


def assemble_velocity(transfer,velocity,stencil):
    rhs=torch.zeros_like(transfer.diagonal)
    velocity=tuple(v.contiguous() for v in velocity)
    offset=0
    for group in stencil.rule.groups:
        torch.ops.afsi_ib_cuda.p2_gather_assemble_(*velocity,group.cells,group.values,group.weights,
            stencil.base,stencil.phi,rhs,offset,*transfer.grid.shape)
        offset+=group.weights.numel()
    return rhs
