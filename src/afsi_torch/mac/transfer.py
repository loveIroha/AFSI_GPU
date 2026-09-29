"""Griffith--Luo unified weak-form, quadrature-based FE/MAC transfer.

b is an integrated FE nodal force, not a force-density coefficient.
M F=b; f=h^-3 K^T W B F; M U=B^T W K u. Reference weights W are used
exactly once. Identical interaction points and kernels are used both ways.
"""
from dataclasses import dataclass
import torch
from ..ib import peskin4
from ..fluid.solvers import SolverOptions, pcg


@dataclass(frozen=True)
class MACStencil:
    indices: tuple
    weights: tuple


class FETransfer:
    def __init__(self, grid, geometry, *, options=None):
        self.grid, self.geometry = grid, geometry
        if (geometry.weights <= 0).any():
            raise ValueError('positive interaction quadrature weights required')
        self.options = (SolverOptions(rtol=1e-12, atol=1e-13, max_iterations=500,
                        check_every=4) if options is None else options)
        cells, N, W = geometry.cells, geometry.values, geometry.weights
        if torch.linalg.matrix_rank(N) < N.shape[1]:
            raise ValueError('interaction quadrature underintegrates the consistent P2 mass')
        local = torch.einsum('eq,qa,qb->eab', W, N, N)
        row = cells[:,:,None].expand_as(local).reshape(-1)
        col = cells[:,None,:].expand_as(local).reshape(-1)
        size = (geometry.node_count,)*2
        self.mass = torch.sparse_coo_tensor(torch.stack((row,col)), local.reshape(-1),
            size, device=W.device, dtype=W.dtype, check_invariants=True).coalesce().to_sparse_csr()
        self.diagonal = W.new_zeros(geometry.node_count).index_add(0, cells.reshape(-1),
            local.diagonal(dim1=-2,dim2=-1).reshape(-1))[:,None].expand(-1,3)
        self.axis_offsets = torch.arange(4, device=W.device)

    def _nodal(self, value):
        g = self.geometry
        if (value.shape != (g.node_count,3) or value.device != g.weights.device or
                value.dtype != g.weights.dtype or not torch.isfinite(value).all()):
            raise ValueError('invalid FE nodal field')

    def evaluate(self, value):
        return torch.einsum('qa,eai->eqi', self.geometry.values, value[self.geometry.cells])

    def mass_action(self, value):
        return torch.sparse.mm(self.mass, value)

    def interaction_points(self, x):
        self._nodal(x)
        return self.evaluate(x).reshape(-1,3)

    def check_support(self, points):
        h = points.new_tensor(self.grid.spacing)
        scaled = (points-points.new_tensor(self.grid.origin))/h
        # All supports stay away from fixed normal faces. Thus every interacting
        # fluid DOF has full h^3 volume and no clipped/renormalized kernel.
        if (not torch.isfinite(points).all() or (scaled < 2).any() or
                (scaled >= points.new_tensor(self.grid.shape)-2).any()):
            raise ValueError('MAC IB support reaches a wall; enlarge/refine the fluid box')

    @torch.no_grad()
    def prepare(self, x):
        points = self.interaction_points(x)
        self.check_support(points)
        indices, weights = [], []
        for c in range(3):
            scaled = (points-points.new_tensor(self.grid.face_origin(c)))/points.new_tensor(self.grid.spacing)
            base = torch.floor(scaled-1).to(torch.int64)
            # Evaluate each one-dimensional kernel only four times per axis.
            # The previous (points,64,3) construction repeated every phi value
            # 16 times and materialized large coordinate/sqrt temporaries.
            nodes = base[:,:,None]+self.axis_offsets[None,None,:]
            shape = self.grid.face_shape(c)
            if (nodes < 0).any() or (nodes >= nodes.new_tensor(shape)[None,:,None]).any():
                raise ValueError('incomplete MAC interaction support')
            phi = peskin4(scaled[:,:,None]-nodes.to(points.dtype))
            # Cartesian order is unchanged: z varies fastest, then y, then x.
            ids = ((nodes[:,0,:,None,None]*shape[1]+nodes[:,1,None,:,None])
                   *shape[2]+nodes[:,2,None,None,:])
            kernel = (phi[:,0,:,None,None]*phi[:,1,None,:,None])*phi[:,2,None,None,:]
            indices.append(ids.reshape(-1,64))
            weights.append(kernel.reshape(-1,64))
        return MACStencil(tuple(indices), tuple(weights))

    @torch.no_grad()
    def spread(self, nodal_force, stencil):
        self._nodal(nodal_force)
        coefficient, info = pcg(self.mass_action, nodal_force, self.diagonal, options=self.options)
        force_q = (self.evaluate(coefficient)*self.geometry.weights[...,None]).reshape(-1,3)
        fields = []
        for c,(ids,weights) in enumerate(zip(stencil.indices,stencil.weights)):
            out = nodal_force.new_zeros(self.grid.face_shape(c))
            out.reshape(-1).index_add_(0,ids.reshape(-1),
                (weights*force_q[:,c,None]/self.grid.volume).reshape(-1))
            fields.append(out)
        return tuple(fields), info

    @torch.no_grad()
    def interpolate(self, velocity, stencil):
        self.grid.check_velocity(velocity)
        if any(u.device != self.geometry.weights.device or u.dtype != self.geometry.weights.dtype
               or not torch.isfinite(u).all() for u in velocity):
            raise ValueError('invalid MAC interpolation field')
        point_velocity = torch.stack([(u.reshape(-1)[ids]*weights).sum(-1)
            for u,ids,weights in zip(velocity,stencil.indices,stencil.weights)],-1)
        W,N,cells = self.geometry.weights,self.geometry.values,self.geometry.cells
        local = torch.einsum('qa,eq,eqi->eai',N,W,point_velocity.reshape(*W.shape,3))
        rhs = W.new_zeros((self.geometry.node_count,3)).index_add(0,cells.reshape(-1),local.reshape(-1,3))
        return pcg(self.mass_action,rhs,self.diagonal,options=self.options)
