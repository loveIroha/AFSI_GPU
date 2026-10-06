"""Frozen quadrature IB contracted to paired PyTorch CSR operators.

For component c, B[a,i] = sum_q W[q]*N[a,q]*phi_c(i,X[q]).
Spread is B.T*(M^-1 force)/cell_volume; interpolation solves M*U=B*u.
The same B values determine both directions; the consistent M is unchanged.
CUDA construction contracts cells in parallel, then coalesces bounded
device-side batches. Only allocation sizes/counters reach the host.
"""
from dataclasses import dataclass
from math import prod
import torch
from .adaptive_transfer import AdaptiveP1Transfer


@dataclass(frozen=True)
class AssembledStencil:
    gather: tuple
    spread: tuple
    rule: object
    assembly: dict

    @property
    def storage_bytes(self):
        return sum(a.crow_indices().numel()*a.crow_indices().element_size()
                   +a.col_indices().numel()*a.col_indices().element_size()
                   +a.values().numel()*a.values().element_size()
                   for a in self.gather+self.spread)


def _bases(stencil,component,offset,count):
    if getattr(stencil,'layout',None)=='shared':
        return torch.stack([stencil.base[int(a!=component),offset:offset+count,a] for a in range(3)],-1)
    return stencil.base[component,offset:offset+count]


def _tables(stencil,component,offset,count):
    base = _bases(stencil,component,offset,count)
    if getattr(stencil,'layout',None)=='shared':
        phi = torch.stack([stencil.phi[int(a!=component),offset:offset+count,a] for a in range(3)],1)
    else:
        phi = stencil.phi[component,offset:offset+count]
    return base,phi


def _cpu_entries(stencil,group,c,offset,low,width,prefix,start,count,shape):
    Q = len(group.values)
    base,phi = _tables(stencil,c,offset,group.weights.numel())
    sites = torch.arange(start,start+count,device=low.device)
    cell = torch.searchsorted(prefix[1:],sites,right=True)
    relative = sites-prefix[cell]
    sizes = width[cell]
    xyz = torch.stack((relative//(sizes[:,1]*sizes[:,2]),relative//sizes[:,2]%sizes[:,1],relative%sizes[:,2]),-1)+low[cell]
    point = cell[:,None]*Q+torch.arange(Q,device=low.device)[None,:]
    distance = xyz[:,None,:]-base[point]
    kernels = []
    for a in range(3):
        d = distance[:,:,a]
        kernels.append(torch.gather(phi[point,:, :][:,:,a,:],2,d.clamp(0,3)[...,None]).squeeze(-1)*((d>=0)&(d<4)))
    weight = (kernels[0]*kernels[1])*kernels[2]*group.weights[cell]
    values = torch.einsum('sq,qa->sa',weight,group.values)
    fluid = (xyz[:,0]*shape[1]+xyz[:,1])*shape[2]+xyz[:,2]
    keys = group.cells[cell]*prod(shape)+fluid[:,None]
    return keys.reshape(-1),values.reshape(-1)


def _coalesce(keys,values):
    keep = values!=0
    keys,values = keys[keep],values[keep]
    unique,inverse = torch.unique(keys,sorted=True,return_inverse=True)
    return unique,values.new_zeros(len(unique)).index_add_(0,inverse,values)


def assemble_component(stencil,c,grid,node_count,*,chunk_entries,max_entries):
    partials = []
    offset = raw_entries = batches = peak_batch = 0
    shape = grid.face_shape(c)
    for group in stencil.rule.groups:
        Q,E = len(group.values),len(group.cells)
        # Reductions stay on the stencil device; no coordinate download.
        base = _bases(stencil,c,offset,group.weights.numel())
        base = base.reshape(E,Q,3)
        low = base.amin(1)
        width = base.amax(1)-low+4
        sizes = width.prod(-1)
        prefix = torch.cat((sizes.new_zeros(1),sizes.cumsum(0)))
        count = int(prefix[-1].item())
        for start in range(0,count,chunk_entries//4):
            size = min(chunk_entries//4,count-start)
            arguments = (stencil,group,c,offset,low,width,prefix,start,size,shape)
            if group.weights.is_cuda:
                from ._triton_transfer_assembly import entries
                keys,values = entries(*arguments)
            else:
                keys,values = _cpu_entries(*arguments)
            raw_entries += len(keys); batches += 1; peak_batch = max(peak_batch,len(keys))
            item = _coalesce(keys,values)
            del keys,values
            level = 0
            # Balanced merges avoid repeatedly sorting a growing global matrix.
            while level<len(partials) and partials[level] is not None:
                previous = partials[level]; partials[level] = None
                item = _coalesce(torch.cat((previous[0],item[0])),torch.cat((previous[1],item[1])))
                level += 1
            if len(item[0])>max_entries:
                raise RuntimeError('assembled IB entry budget exceeded; use quadrature backend or raise ib_csr_max_entries')
            if level==len(partials): partials.append(item)
            else: partials[level] = item
            if sum(len(v[0]) for v in partials if v is not None)>2*max_entries:
                raise RuntimeError('assembled IB partial storage budget exceeded; use quadrature backend')
        offset += group.weights.numel()
    active = [v for v in partials if v is not None]
    keys,values = active[0]
    for other in active[1:]:
        keys,values = _coalesce(torch.cat((keys,other[0])),torch.cat((values,other[1])))
        if len(keys)>max_entries:
            raise RuntimeError('assembled IB entry budget exceeded')
    rows,columns = keys//prod(shape),keys%prod(shape)
    matrix = torch.sparse_coo_tensor(torch.stack((rows,columns)),values,
        (node_count,prod(shape)),device=values.device,dtype=values.dtype,
        is_coalesced=True,check_invariants=False).to_sparse_csr()
    # Materialize the transpose once; every response uses CSR SpMV both ways.
    transpose = matrix.transpose(0,1).to_sparse_csr()
    return matrix,transpose,dict(raw_cell_entries=raw_entries,batches=batches,
        maximum_batch_entries=peak_batch,nnz=len(values))


class AssembledP1Transfer(AdaptiveP1Transfer):
    def __init__(self,*args,chunk_entries=1048576,max_entries=32000000,assembly_backend='coalesce',**kwargs):
        if type(chunk_entries) is not int or chunk_entries<4 or type(max_entries) is not int or max_entries<1:
            raise ValueError('positive CSR entry budget and chunk_entries>=4 required')
        if assembly_backend not in ('coalesce','hash'):
            raise ValueError('CSR assembly_backend must be coalesce or hash')
        super().__init__(*args,**kwargs)
        self.chunk_entries,self.max_entries = chunk_entries,max_entries
        self.assembly_backend = assembly_backend
        from .hash_transfer_assembly import HashWorkspace
        self.hash_workspaces = tuple(HashWorkspace() for _ in range(3))
        self._last_assembly = None
        self.csr_builds = 0

    @torch.no_grad()
    def assemble_stencil(self,stencil):
        gather,spread,statistics = [],[],[]
        remaining = self.max_entries
        for c in range(3):
            if self.assembly_backend=='hash':
                from .hash_transfer_assembly import assemble_component_hash
                B,BT,info = assemble_component_hash(stencil,c,self.grid,self.geometry.node_count,
                    chunk_entries=self.chunk_entries,max_entries=remaining,workspace=self.hash_workspaces[c])
            else:
                B,BT,info = assemble_component(stencil,c,self.grid,self.geometry.node_count,
                    chunk_entries=self.chunk_entries,max_entries=remaining)
            remaining -= info['nnz']
            if remaining<0:
                raise RuntimeError('assembled IB total entry budget exceeded')
            gather.append(B); spread.append(BT); statistics.append(info)
        result = AssembledStencil(tuple(gather),tuple(spread),stencil.rule,
            dict(components=statistics,total_nnz=sum(v['nnz'] for v in statistics),
                 chunk_entries=self.chunk_entries,max_entries=self.max_entries,
                 assembly_backend=self.assembly_backend,
                 hash_workspace_bytes=sum(w.storage_bytes for w in self.hash_workspaces),
                 builder=('triton-cell+device-hash+unique-sort' if self.assembly_backend=='hash'
                          else 'triton-cell+torch-coalesce') if self.geometry.weights.is_cuda else 'torch-cpu-oracle'))
        self._last_assembly = dict(result.assembly,csr_storage_bytes=result.storage_bytes)
        self.csr_builds += 1
        return result

    def prepare(self,x):
        return self.assemble_stencil(super().prepare(x))

    @torch.no_grad()
    def spread(self,nodal_force,stencil):
        self._nodal(nodal_force)
        coefficient,info = self.solve_mass(nodal_force,self._force_coefficient if self.warm_start else None)
        if self.warm_start: self._force_coefficient = coefficient
        density = tuple((torch.sparse.mm(B,coefficient[:,c:c+1])/self.grid.volume).reshape(self.grid.face_shape(c))
                        for c,B in enumerate(stencil.spread))
        return density,info

    @torch.no_grad()
    def interpolate(self,velocity,stencil):
        self.grid.check_velocity(velocity)
        if any(u.device!=self.geometry.weights.device or u.dtype!=self.geometry.weights.dtype for u in velocity) or not self._finite_kernel(*velocity):
            raise ValueError('invalid MAC interpolation field')
        rhs = torch.cat([torch.sparse.mm(B,u.reshape(-1,1)) for B,u in zip(stencil.gather,velocity)],1)
        result,info = self.solve_mass(rhs,self._velocity_coefficient if self.warm_start else None)
        if self.warm_start: self._velocity_coefficient = result
        return result,info

    def quadrature_summary(self):
        return dict(super().quadrature_summary(),response_backend='csr',assembly=self._last_assembly,
                    csr_builds=self.csr_builds)
