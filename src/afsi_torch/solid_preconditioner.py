"""Nodal 3x3 blocks for Guccione/active solid and reference-area springs.

Only a Newton preconditioner. The actual JVP includes all residual terms;
follower-pressure derivatives are deliberately omitted from these blocks.
"""
import torch
from . import solid
from .materials import guccione_pk1, active_pk1


@torch.no_grad()
def guccione_blocks(x,geometry,fields,parameters,*,base=None,beta=0.):
    F = solid.deformation_gradient(x,geometry)
    def stress(F,f,s,n,t):
        return guccione_pk1(F,f,s,n,parameters)+active_pk1(F,f,t)
    tangent = torch.vmap(torch.func.jacfwd(stress,argnums=0))(
        F.reshape(-1,3,3),fields.fiber.reshape(-1,3),fields.sheet.reshape(-1,3),
        fields.normal.reshape(-1,3),fields.tension.reshape(-1)).reshape(*F.shape[:2],3,3,3,3)
    local = torch.einsum('eq,eqaJ,eqiJkL,eqaL->eaik',geometry.weights,geometry.gradients,tangent,geometry.gradients)
    blocks = x.new_zeros((len(x),3,3)).index_add(0,geometry.cells.reshape(-1),local.reshape(-1,3,3))
    if base is not None:
        weights = base.reference_weights*beta
        spring = torch.einsum('bq,qa,qa->ba',weights,base.values,base.values)
        diagonal = x.new_zeros(len(x)).index_add(0,base.faces.reshape(-1),spring.reshape(-1))
        blocks = blocks+diagonal[:,None,None]*torch.eye(3,device=x.device,dtype=x.dtype)
    if not torch.isfinite(blocks).all():
        raise FloatingPointError('nonfinite solid preconditioner')
    return blocks


def block_inverse(blocks):
    inverse = torch.linalg.inv(blocks)
    if not torch.isfinite(inverse).all():
        raise FloatingPointError('nonfinite block inverse')
    return lambda v: (inverse@v[...,None]).squeeze(-1)
