"""Pointwise affine P1 contractions, without library batched small GEMMs."""
import torch


def affine_gradient(x,cells,gradients):
    nodes=x[cells]
    return (nodes[:,0,:,None]*gradients[:,0,None,:]
            +nodes[:,1,:,None]*gradients[:,1,None,:]
            +nodes[:,2,:,None]*gradients[:,2,None,:]
            +nodes[:,3,:,None]*gradients[:,3,None,:])


def affine_area_vectors(x,surface):
    nodes=x[surface.faces]
    # A P1 triangle has the same two tangents at every original quadrature
    # point. Keep the original Q axis as an expanded view for force assembly.
    area=torch.linalg.cross(nodes[:,1]-nodes[:,0],nodes[:,2]-nodes[:,0])
    return area[:,None,:].expand(-1,len(surface.values),-1)
