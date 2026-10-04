"""Execution-only FE/IB fusion; same quadrature, kernels and consistent mass.

CPU implementations exercise the cell reductions independently of CUDA.
CUDA implementations avoid full point-force/point-velocity intermediates.
"""
from math import ceil
import torch


def support_extent(group, density):
    # hmax <= order*dx_min/density. Thus each coordinate's integer base
    # spans at most ceil(order/density), plus four Peskin support nodes.
    return 4+ceil(group.order/density)


def spread(transfer, coefficient, stencil, *, reduced):
    if coefficient.is_cuda:
        from ._triton_adaptive import spread as cuda_spread
        return cuda_spread(transfer,coefficient,stencil,reduced=reduced)
    result = transfer.grid.zeros(device=coefficient.device,dtype=coefficient.dtype)
    offset = 0
    for group in stencil.rule.groups:
        count = group.weights.numel()
        force = torch.einsum('qa,eai->eqi',group.values,coefficient[group.cells])*group.weights[...,None]
        if not reduced:
            from .compact_transfer import CompactStencil
            part = CompactStencil(stencil.base[:,offset:offset+count],stencil.phi[:,offset:offset+count])
            fields = transfer.spread_grid(force.reshape(-1,3),part)
            for out,value in zip(result,fields):
                out.add_(value)
        else:
            extent = support_extent(group,transfer.quadrature_options.point_density)
            axis = torch.arange(extent,device=coefficient.device)
            nodes = torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
            for c,out in enumerate(result):
                base = stencil.base[c,offset:offset+count].reshape(len(group.cells),-1,3)
                phi = stencil.phi[c,offset:offset+count].reshape(len(group.cells),-1,3,4)
                # Chunk cells so the CPU validation path also has bounded memory.
                for start in range(0,len(group.cells),16):
                    b,p = base[start:start+16],phi[start:start+16]
                    ids = b.amin(1)[:,None]+nodes[None]
                    delta = ids[:,:,None]-b[:,None]
                    valid = ((delta>=0)&(delta<4)).all(-1)
                    weights = torch.ones_like(valid,dtype=coefficient.dtype)
                    for a in range(3):
                        table = p[:,:,a,:][:,None].expand(-1,len(nodes),-1,-1)
                        weights *= torch.gather(table,3,delta[...,a,None].clamp(0,3)).squeeze(-1)
                    local = (weights*valid*force[start:start+16,None,:,c]).sum(-1)/transfer.grid.volume
                    _,ny,nz = transfer.grid.face_shape(c)
                    linear = (ids[...,0]*ny+ids[...,1])*nz+ids[...,2]
                    # Empty nodes can lie outside the face array; never access them.
                    active = valid.any(-1)
                    out.reshape(-1).index_add_(0,linear[active],local[active])
        offset += count
    return result


def assemble_velocity(transfer, velocity, stencil):
    if velocity[0].is_cuda:
        from ._triton_adaptive import assemble_velocity as cuda_assemble
        return cuda_assemble(transfer,velocity,stencil)
    from .compact_transfer import CompactStencil
    rhs = torch.zeros_like(transfer.diagonal)
    offset = 0
    for group in stencil.rule.groups:
        count = group.weights.numel()
        part = CompactStencil(stencil.base[:,offset:offset+count],stencil.phi[:,offset:offset+count])
        point_velocity = transfer.gather_grid(velocity,part).reshape(*group.weights.shape,3)
        local = torch.einsum('qa,eq,eqi->eai',group.values,group.weights,point_velocity)
        rhs.index_add_(0,group.cells.reshape(-1),local.reshape(-1,3))
        offset += count
    return rhs
