"""Direct bounded hash reduction of cell contributions into PyTorch CSR.

CUDA never emits the full cell/node/lattice entry stream. Each parallel
cell quadrature reduction adds its four nodal values to device buckets.
Only unique keys are compacted/sorted once, then converted to paired CSR.
CPU insertion below is a small independent oracle, not a CUDA fallback.
"""
from dataclasses import dataclass
from math import prod
import torch


def power_of_two(n):
    return 1 << max(0,(int(n)-1).bit_length())


def hash_slot(key,capacity):
    """Independent unsigned SplitMix64 transcription for CPU tests."""
    mask = (1<<64)-1
    h = int(key) & mask
    h = ((h^(h>>30))*0xbf58476d1ce4e5b9) & mask
    h = ((h^(h>>27))*0x94d049bb133111eb) & mask
    return (h^(h>>31)) & (capacity-1)


@dataclass
class HashWorkspace:
    capacity: int = 0
    keys: torch.Tensor | None = None
    values: torch.Tensor | None = None
    flag: torch.Tensor | None = None
    allocations: int = 0
    reuses: int = 0

    def reset(self,capacity,like):
        if capacity<1 or capacity & (capacity-1):
            raise ValueError('power-of-two hash capacity required')
        # Scratch is bounded and reused, but never owns returned CSR values.
        if self.capacity!=capacity or self.values is None or self.values.device!=like.device or self.values.dtype!=like.dtype:
            self.keys = torch.empty(capacity,device=like.device,dtype=torch.int64)
            self.values = torch.empty(capacity,device=like.device,dtype=like.dtype)
            self.flag = torch.empty((),device=like.device,dtype=torch.int32)
            self.capacity = capacity
            self.allocations += 1
        else:
            self.reuses += 1
        self.keys.fill_(-1); self.values.zero_(); self.flag.zero_()

    @property
    def storage_bytes(self):
        return self.capacity*(8+self.values.element_size())+4 if self.values is not None else 0


def cpu_accumulate(keys,values,workspace):
    if keys.is_cuda or values.is_cuda:
        raise ValueError('CPU hash oracle cannot accept CUDA tensors')
    for key,value in zip(keys.tolist(),values.tolist()):
        if value==0:
            continue
        slot = hash_slot(key,workspace.capacity)
        for _ in range(128):
            previous = int(workspace.keys[slot])
            if previous in (-1,key):
                workspace.keys[slot] = key
                workspace.values[slot] += value
                break
            slot = (slot+1) & (workspace.capacity-1)
        else:
            workspace.flag.fill_(1)


def sorted_entries(workspace):
    # One compaction and unique-key sort. No sort of raw cell contributions.
    occupied = workspace.keys>=0
    keys,values = workspace.keys[occupied],workspace.values[occupied]
    keys,permutation = torch.sort(keys)
    return keys,values[permutation]


def accumulate_plans(stencil,c,plans,shape,workspace,chunk_entries):
    from .assembled_transfer import _cpu_entries
    launches = peak_batch = 0
    for group,offset,low,width,prefix,count in plans:
        if group.weights.is_cuda:
            from ._triton_transfer_assembly import hash_entries
            hash_entries(stencil,group,c,offset,low,width,prefix,count,shape,workspace)
            launches += 1
        else:
            for start in range(0,count,chunk_entries//4):
                size = min(chunk_entries//4,count-start)
                keys,values = _cpu_entries(stencil,group,c,offset,low,width,prefix,start,size,shape)
                peak_batch = max(peak_batch,len(keys))
                cpu_accumulate(keys,values,workspace)
                launches += 1
    return launches,peak_batch


def finish_component(workspace,node_count,shape,max_entries):
    keys,values = sorted_entries(workspace)
    if len(keys)>max_entries:
        raise RuntimeError('assembled IB entry budget exceeded; no hash entries were truncated')
    # Fail at the builder rather than letting corrupt data reach cuSPARSE
    # or a later fluid residual. One scalar transfer per completed component.
    valid = (keys>=0).all() & (keys<node_count*prod(shape)).all() & torch.isfinite(values).all()
    if not valid.item():
        raise FloatingPointError('invalid hash IB entries: nonfinite values or out-of-range keys')
    rows,columns = keys//prod(shape),keys%prod(shape)
    matrix = torch.sparse_coo_tensor(torch.stack((rows,columns)),values,
        (node_count,prod(shape)),device=values.device,dtype=values.dtype,
        is_coalesced=True,check_invariants=False).to_sparse_csr()
    transpose = matrix.transpose(0,1).to_sparse_csr()
    return matrix,transpose,len(keys)


def assemble_component_hash(stencil,c,grid,node_count,*,chunk_entries,max_entries,workspace):
    from .assembled_transfer import _bases
    shape = grid.face_shape(c)
    plans = []
    offset = raw = 0
    for group in stencil.rule.groups:
        E,Q = len(group.cells),len(group.values)
        base = _bases(stencil,c,offset,group.weights.numel()).reshape(E,Q,3)
        low = base.amin(1)
        width = base.amax(1)-low+4
        sizes = width.prod(-1)
        prefix = torch.cat((sizes.new_zeros(1),sizes.cumsum(0)))
        count = int(prefix[-1].item())
        plans.append((group,offset,low,width,prefix,count))
        raw += 4*count
        offset += group.weights.numel()
    limit = power_of_two(2*max_entries)
    # A heuristic affects capacity/retries only, never the accepted matrix.
    capacity = min(limit,max(workspace.capacity,power_of_two(max(128,raw//8))))
    attempts = launches = peak_batch = 0
    while True:
        workspace.reset(capacity,stencil.rule.groups[0].weights)
        attempts += 1
        calls,peak = accumulate_plans(stencil,c,plans,shape,workspace,chunk_entries)
        launches += calls; peak_batch = max(peak_batch,peak)
        if not workspace.flag.item():
            break
        if capacity>=limit:
            raise RuntimeError('assembled IB hash probe/budget exceeded; use coalesce backend or raise ib_csr_max_entries')
        # Failed accumulation is discarded in full. Retry on a larger table.
        capacity = min(limit,2*capacity)
    matrix,transpose,nnz = finish_component(workspace,node_count,shape,max_entries)
    return matrix,transpose,dict(raw_cell_entries=raw,batches=launches,
        maximum_batch_entries=peak_batch,nnz=nnz,hash_capacity=workspace.capacity,
        hash_attempts=attempts,hash_workspace_bytes=workspace.storage_bytes,
        hash_workspace_allocations=workspace.allocations,hash_workspace_reuses=workspace.reuses,
        raw_entries_materialized=not workspace.values.is_cuda,unique_key_sorts=1,
        duplicate_merge_sorts=0)
