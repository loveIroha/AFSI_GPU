"""Gao/IBTK-style adaptive Gaussian interaction quadrature for affine P1 FE.

Select n= max(2, ceil(point_density*deformed_hmax/dx_min)), degree=2*n-1
per cell. Only small reference rule tables are generated on the CPU. Cell
selection, FE evaluation/assembly and IB kernels use the tensor device.
Each stencil owns its quadrature: later prepares cannot change an existing
linear/nonlinear action. The reference consistent CSR mass remains fixed.
"""
from dataclasses import dataclass
from math import isfinite
import numpy as np
import torch
from .compact_transfer import CompactFETransfer, CompactStencil, _prepare, _evaluate, _weighted, _assemble
from ..p1 import tetra_rule
from .shared_stencil import SharedStencil


@dataclass(frozen=True)
class InteractionQuadratureOptions:
    mode: str = 'fixed'
    point_density: float = 2.
    max_order: int = 8
    max_points: int = 12000000
    rule_family: str = 'conical'
    transfer_backend: str = 'reference'
    stencil_backend: str = 'component'
    prepare_backend: str = 'torch'
    reuse_stencil_buffers: bool = False
    shared_execution: str = 'reference'

    def __post_init__(self):
        if self.shared_execution not in ('reference','vector','reduced'):
            raise ValueError('shared_execution must be reference, vector or reduced')
        if self.shared_execution!='reference' and (self.mode!='adaptive' or
                self.stencil_backend!='shared' or self.transfer_backend!='fused'):
            raise ValueError('vector/reduced shared execution requires adaptive/shared/fused transfer')
        if type(self.reuse_stencil_buffers) is not bool:
            raise ValueError('reuse_stencil_buffers must be a bool')
        if self.reuse_stencil_buffers and (self.stencil_backend!='shared' or self.prepare_backend!='triton'):
            raise ValueError('stencil buffer reuse requires shared/triton preparation')
        if self.mode not in ('fixed','adaptive'):
            raise ValueError('interaction quadrature mode must be fixed or adaptive')
        if self.rule_family not in ('conical','xiao-gimbutas'):
            raise ValueError('interaction rule_family must be conical or xiao-gimbutas')
        if self.transfer_backend not in ('reference','fused','cell'):
            raise ValueError('adaptive transfer_backend must be reference, fused or cell')
        if self.stencil_backend not in ('component','shared'):
            raise ValueError('stencil_backend must be component or shared')
        if self.stencil_backend=='shared' and (self.mode!='adaptive' or self.transfer_backend=='cell'):
            raise ValueError('shared stencil requires adaptive quadrature and reference/fused transfer execution')
        if self.prepare_backend not in ('torch','triton'):
            raise ValueError('prepare_backend must be torch or triton')
        if self.prepare_backend=='triton' and self.stencil_backend!='shared':
            raise ValueError('triton preparation requires shared adaptive stencils')
        if isinstance(self.point_density,bool) or not isfinite(self.point_density) or self.point_density < 2:
            raise ValueError('interaction point_density must be finite and >=2')
        if type(self.max_order) is not int or not 2 <= self.max_order <= 22:
            raise ValueError('interaction max_order must be an integer in [2,22]')
        if type(self.max_points) is not int or self.max_points < 8:
            raise ValueError('interaction max_points must be an integer >=8')


def _gauss_jacobi_unit(n, alpha):
    """Positive Gauss rule for integral_0^1 f(u)*(1-u)^alpha du."""
    k = np.arange(n,dtype=float)
    d = 2*k+alpha
    diagonal = np.zeros(n) if alpha==0 else -alpha*alpha/(d*(d+2))
    j = np.arange(1,n,dtype=float)
    a = 2*j+alpha
    off = np.sqrt(4*j*j*(j+alpha)**2/(a*a*(a*a-1)))
    nodes,vectors = np.linalg.eigh(np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1))
    return (nodes+1)/2, vectors[0]**2/(alpha+1)


def gaussian_tetra_rule(order, like, family='conical'):
    """Positive degree 2*order-1 rule, with selectable compact reference tables.

    This matches the IBTK degree criterion, not libMesh's exact point tables.
    Compact Xiao--Gimbutas tables cover order<=8; higher orders fall back to
    conical Gauss. The conical family retains its Keast degree-5 table.
    The Jacobi factors include the Duffy Jacobian, so P1 mass is integrated
    exactly even for order=2. Plain 2x2x2 Legendre/Duffy would not suffice.
    """
    if type(order) is not int or order < 2:
        raise ValueError('Gaussian order must be an integer >=2')
    if family not in ('conical','xiao-gimbutas'):
        raise ValueError('unknown Gaussian rule family')
    if family=='xiao-gimbutas' and order<=8:
        try:
            import basix
        except ImportError as exc:
            raise ImportError('Compact quadrature needs: pip install -e ".[quadrature]"') from exc
        q,w = basix.make_quadrature(basix.CellType.tetrahedron,2*order-1,
                                  rule=basix.QuadratureType.xiao_gimbutas)
        if (np.any(w<=0) or np.any(q<0) or np.any(q.sum(-1)>1+1e-13)
                or not np.isfinite(q).all() or not np.isfinite(w).all()
                or abs(w.sum()-1/6)>1e-13):
            raise ValueError('invalid compact Gaussian quadrature table')
        return like.new_tensor(q),like.new_tensor(w)
    # Basix 0.10 tetrahedral XG tables stop at degree 15. Higher orders use
    # the original positive conical rule at the full requested degree.
    if order==3:
        return tetra_rule(5,like)
    nodes,weights = zip(*(_gauss_jacobi_unit(order,a) for a in (2,1,0)))
    u,v,w = torch.meshgrid(*(like.new_tensor(z) for z in nodes),indexing='ij')
    a,b,c = torch.meshgrid(*(like.new_tensor(z) for z in weights),indexing='ij')
    return torch.stack((u,(1-u)*v,(1-u)*(1-v)*w),-1).reshape(-1,3),(a*b*c).reshape(-1)


@dataclass(frozen=True)
class QuadratureGroup:
    cells: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor
    order: int = 0


@dataclass(frozen=True)
class InteractionRule:
    groups: tuple
    orders: torch.Tensor
    point_count: int


@dataclass(frozen=True)
class AdaptiveStencil(CompactStencil):
    rule: InteractionRule


@dataclass(frozen=True)
class SharedAdaptiveStencil(SharedStencil):
    rule: InteractionRule


def _orders(x,cells,edges,dx,density):
    nodes = x[cells]
    hmax = torch.linalg.vector_norm(nodes[:,edges[:,0]]-nodes[:,edges[:,1]],dim=-1).amax(-1)
    return torch.ceil(density*hmax/dx).clamp_min(2).to(torch.int64)


@torch.no_grad()
def interaction_quadrature_plan(x,cells,grid,options):
    """Inspect required adaptive rules without mass, stencils or fluid solves."""
    edges = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],device=x.device)
    orders = _orders(x,cells,edges,min(grid.spacing),options.point_density)
    distinct,counts = torch.unique(orders,return_counts=True)
    histogram = {int(n):int(c) for n,c in zip(distinct.cpu().tolist(),counts.cpu().tolist())}
    table_sizes = {n:len(gaussian_tetra_rule(n,x,options.rule_family)[1]) for n in histogram}
    total = sum(count*table_sizes[n] for n,count in histogram.items())
    return dict(point_density=options.point_density,point_count=total,
        rule_family=options.rule_family,points_per_order=table_sizes,
        gaussian_order_cell_counts=histogram,maximum_order=max(histogram),
        max_order=options.max_order,max_points=options.max_points,
        within_budget=max(histogram)<=options.max_order and total<=options.max_points,
        mesh_cells=len(cells),fluid_spacing_cm=grid.spacing,
        meaning='initial geometry only; selection updates with deformation')


class AdaptiveP1Transfer(CompactFETransfer):
    def __init__(self,*args,quadrature_options=None,fused=True,**kwargs):
        super().__init__(*args,**kwargs)
        if self.geometry.cells.shape[1]!=4:
            raise ValueError('adaptive interaction quadrature currently requires affine P1 tetrahedra')
        self.quadrature_options = quadrature_options or InteractionQuadratureOptions(mode='adaptive')
        if self.quadrature_options.stencil_backend=='shared':
            # Use identical subtraction/division as the component-specific
            # origins, computing each face/center axis table only once.
            self.origins = self.geometry.weights.new_tensor([self.grid.origin,
                tuple(o+.5*h for o,h in zip(self.grid.origin,self.grid.spacing))])
        self._rules,self._last_rule = {},None
        self._last_prepared_rule = None
        self.quadrature_builds = 0
        self._last_stencil_bytes = None
        from .stencil_workspace import StencilWorkspace
        self.stencil_workspace = StencilWorkspace(self.quadrature_options.max_points)
        device = self.geometry.weights.device
        # Groups change size as elements deform. Static-shape compilation here
        # would repeatedly compile and eventually hit Dynamo's cache limit.
        def compile_kernel(function):
            if fused and device.type=='cuda':
                return torch.compile(function,fullgraph=True,dynamic=True,
                                     options={'triton.cudagraphs':False})
            return function
        self._order_kernel = compile_kernel(_orders)
        self._prepare_kernel = compile_kernel(_prepare)
        self._evaluate_kernel = compile_kernel(_evaluate)
        self._weighted_kernel = compile_kernel(_weighted)
        self._assemble_kernel = compile_kernel(_assemble)
        self._reference_support_kernel = compile_kernel(self._support_flags)
        self.set_validation_backend('blocked' if fused else 'reference')
        if not fused:
            self._finite_kernel = lambda a,b,c:torch.isfinite(a).all() & torch.isfinite(b).all() & torch.isfinite(c).all()
            self._nodal_finite_kernel = lambda a:torch.isfinite(a).all()
        self.edges = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]],device=device)
        # det(reference edge matrix), never det(F) or current cell volume.
        self.reference_determinants = 6*self.geometry.weights.sum(-1)
        self._dx = min(self.grid.spacing)
        self._validation_vertices=torch.unique(self.geometry.cells)
        self._all_vertices_used=len(self._validation_vertices)==self.geometry.node_count

    def _rule(self,x):
        opt = self.quadrature_options
        orders = self._order_kernel(x,self.geometry.cells,self.edges,self._dx,opt.point_density)
        if self._last_rule is not None and torch.equal(orders,self._last_rule.orders):
            return self._last_rule
        maximum = int(orders.max().item())
        if maximum>opt.max_order:
            raise ValueError(f'adaptive IB needs Gaussian order {maximum}, exceeds max_order={opt.max_order}; no density clipping')
        histogram = torch.bincount(orders,minlength=maximum+1).cpu().tolist()
        tables,total = {},0
        for n,count in enumerate(histogram):
            if not count:
                continue
            if n not in self._rules:
                q,w = gaussian_tetra_rule(n,x,opt.rule_family)
                self._rules[n] = torch.cat((1-q.sum(-1,keepdim=True),q),-1),w
            tables[n] = self._rules[n]
            total += count*len(tables[n][1])
        if total>opt.max_points:
            raise ValueError(f'adaptive IB needs {total} points, exceeds max_points={opt.max_points}; no density clipping')
        groups = []
        for n,(N,w) in tables.items():
            ids = torch.where(orders==n)[0]
            groups.append(QuadratureGroup(self.geometry.cells[ids],N,self.reference_determinants[ids,None]*w,n))
        rule = InteractionRule(tuple(groups),orders,total)
        self._last_rule = rule
        self.quadrature_builds += 1
        return rule

    def _points(self,x,rule):
        return torch.cat([self._evaluate_kernel(x,g.values,g.cells).reshape(-1,3) for g in rule.groups])

    def interaction_points(self,x):
        self._nodal(x)
        return self._points(x,self._rule(x))

    def validation_points(self,x):
        # Every affine P1 quadrature point lies in this convex vertex hull.
        if self.validation_backend=='reference':
            return x[self.geometry.cells].reshape(-1,3)
        # Test exactly the same set of vertices, without duplicate cell-corner
        # reads. Do not add orphan vertices to the original support predicate.
        return x if self._all_vertices_used else x[self._validation_vertices]

    def check_configuration_support(self,x):
        """Fast sufficient P1 vertex screen; exact point predicate on fallback.

        A point is a convex combination of its four vertices. An interior
        vertex hull guarantees complete support. A rejected hull does NOT
        reject the configuration: test the original quadrature points then.
        Retain the original adaptive-order and point-count limit checks.
        """
        self._nodal(x)
        rule = self._rule(x)
        if self._vertex_support_kernel(self.validation_points(x)):
            return
        self.check_support(self._points(x,rule))

    def _direct_prepare(self,x,rule,*,buffers=None):
        from ._triton_prepare import prepare
        return prepare(self,x,rule,buffers=buffers)

    @torch.no_grad()
    def prepare(self,x):
        self._nodal(x)
        rule = self._rule(x)
        slot = None
        if self.quadrature_options.reuse_stencil_buffers:
            base,phi,slot = self.stencil_workspace.acquire(x,rule.point_count)
        if self.quadrature_options.prepare_backend=='triton' and x.is_cuda:
            base,phi = self._direct_prepare(x,rule,buffers=None if slot is None else (base,phi))
        else:
            points = self._points(x,rule)
            self.check_support(points)
            kernel = self.from_points(points)
            if slot is None:
                base,phi = kernel.base,kernel.phi
            else:
                base.copy_(kernel.base); phi.copy_(kernel.phi)
        self._last_prepared_rule = rule
        cls = SharedAdaptiveStencil if self.quadrature_options.stencil_backend=='shared' else AdaptiveStencil
        stencil = cls(base,phi,rule)
        if slot is not None:
            self.stencil_workspace.retain(slot,stencil)
        self._last_stencil_bytes = stencil.storage_bytes
        return stencil

    def spread_grid(self,force_q,stencil):
        if isinstance(stencil,SharedStencil) and not force_q.is_cuda:
            stencil = stencil.expanded()
        return super().spread_grid(force_q,stencil)

    def gather_grid(self,velocity,stencil):
        if isinstance(stencil,SharedStencil) and not velocity[0].is_cuda:
            stencil = stencil.expanded()
        return super().gather_grid(velocity,stencil)

    @torch.no_grad()
    def spread(self,nodal_force,stencil):
        self._nodal(nodal_force)
        if not isinstance(stencil,(AdaptiveStencil,SharedAdaptiveStencil)):
            raise ValueError('adaptive transfer requires a stencil with its paired quadrature')
        coefficient,info = self.solve_mass(nodal_force,self._force_coefficient if self.warm_start else None)
        if self.warm_start:
            self._force_coefficient = coefficient
        if self.quadrature_options.transfer_backend!='reference':
            from .adaptive_cell import spread
            return spread(self,coefficient,stencil,reduced=self.quadrature_options.transfer_backend=='cell'),info
        force_q = torch.cat([self._weighted_kernel(coefficient,g.values,g.cells,g.weights) for g in stencil.rule.groups])
        return self.spread_grid(force_q,stencil),info

    @torch.no_grad()
    def interpolate(self,velocity,stencil):
        self.grid.check_velocity(velocity)
        g = self.geometry
        if (any(u.device!=g.weights.device or u.dtype!=g.weights.dtype for u in velocity)
                or not self._finite_kernel(*velocity)):
            raise ValueError('invalid MAC interpolation field')
        if not isinstance(stencil,(AdaptiveStencil,SharedAdaptiveStencil)):
            raise ValueError('adaptive transfer requires a stencil with its paired quadrature')
        if self.quadrature_options.transfer_backend!='reference':
            from .adaptive_cell import assemble_velocity
            rhs = assemble_velocity(self,velocity,stencil)
        else:
            gathered = self.gather_grid(velocity,stencil)
            rhs,offset = torch.zeros_like(self.diagonal),0
            for group in stencil.rule.groups:
                count = group.weights.numel()
                rhs += self._assemble_kernel(gathered[offset:offset+count],group.values,group.cells,group.weights,self.diagonal)
                offset += count
        result,info = self.solve_mass(rhs,self._velocity_coefficient if self.warm_start else None)
        if self.warm_start:
            self._velocity_coefficient = result
        return result,info

    def quadrature_summary(self):
        rule = self._last_prepared_rule or self._last_rule
        return dict(mode='adaptive',point_density=self.quadrature_options.point_density,
                    transfer_backend=self.quadrature_options.transfer_backend,
                    shared_execution=self.quadrature_options.shared_execution,
                    shared_execution_device='cuda' if self.geometry.weights.is_cuda else 'cpu-oracle',
                    shared_gather_nodal_atomic_updates=(12*sum(len(g.cells) for g in rule.groups)
                        if rule is not None and self.geometry.weights.is_cuda and self.quadrature_options.shared_execution!='reference'
                        else None),
                    shared_spread_tile_points=4 if self.quadrature_options.shared_execution=='reduced' else None,
                    stencil_backend=self.quadrature_options.stencil_backend,
                    prepare_backend=self.quadrature_options.prepare_backend,
                    prepare_execution='triton' if self.quadrature_options.prepare_backend=='triton' and self.geometry.weights.is_cuda else 'torch',
                    point_intermediates_materialized=(False if self.geometry.weights.is_cuda and self.quadrature_options.transfer_backend=='fused' and self.quadrature_options.stencil_backend=='shared'
                        else True if not self.geometry.weights.is_cuda or self.quadrature_options.transfer_backend=='reference' else None),
                    stencil_storage_bytes=self._last_stencil_bytes,
                    stencil_workspace=self.stencil_workspace.summary(),
                    rule_family=self.quadrature_options.rule_family,
                    points_per_order={n:len(table[1]) for n,table in self._rules.items()},
                    high_order_fallback='positive conical above order 8' if self.quadrature_options.rule_family=='xiao-gimbutas' else None,
                    point_count=rule.point_count if rule else None,
                    gaussian_order_cell_counts=torch.bincount(rule.orders).cpu().tolist() if rule else [],
                    degree_selection='2*max(2,ceil(point_density*hmax/dx_min))-1',
                    sample='last prepared stencil, or initialization before the first prepare',
                    weights='reference volume',consistent_mass='fixed reference CSR',
                    quadrature_builds=self.quadrature_builds)
