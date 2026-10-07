"""Exact numeric IB updates with bounded persistent hash/CSR structure.

Existing keys are read without atomic CAS. New contributions go to a small
separate stream, merged after the read-only phase finishes. A support change
updates the pattern; a quadrature change simply integrates the new rule.
No weights, forces or coupling responses are reused between time steps.
"""
from dataclasses import dataclass
from math import prod
import torch
from .hash_transfer_assembly import (component_plans,accumulate_plans,
    power_of_two,cpu_accumulate,hash_slot)


class PatternBudgetExceeded(RuntimeError):
    pass


@dataclass
class CSRPatternCache:
    slots: torch.Tensor | None = None
    transpose_order: torch.Tensor | None = None
    crow: torch.Tensor | None = None
    columns: torch.Tensor | None = None
    transpose_crow: torch.Tensor | None = None
    transpose_columns: torch.Tensor | None = None
    missing_keys: torch.Tensor | None = None
    missing_values: torch.Tensor | None = None
    missing_count: torch.Tensor | None = None
    missing_capacity: int = 0
    rebuilds: int = 0
    reuses: int = 0
    extensions: int = 0
    resets: int = 0
    overflow_fallbacks: int = 0
    pattern_shape: tuple | None = None
    last_sorted_keys: int = 0
    total_sorted_keys: int = 0
    total_missing_entries: int = 0

    @property
    def nnz(self):
        return 0 if self.slots is None else len(self.slots)

    @property
    def storage_bytes(self):
        # Includes immutable CSR indices also shared by returned snapshots.
        return sum(v.numel()*v.element_size() for v in (self.slots,self.transpose_order,
            self.crow,self.columns,self.transpose_crow,self.transpose_columns,
            self.missing_keys,self.missing_values,self.missing_count) if v is not None)

    def scratch(self,capacity,like):
        if (self.missing_keys is None or self.missing_capacity!=capacity or
                self.missing_values.device!=like.device or self.missing_values.dtype!=like.dtype):
            self.missing_keys = torch.empty(capacity,device=like.device,dtype=torch.int64)
            self.missing_values = torch.empty(capacity,device=like.device,dtype=like.dtype)
            self.missing_count = torch.zeros((),device=like.device,dtype=torch.int64)
            self.missing_capacity = capacity
        self.missing_count.zero_()

    def invalidate(self):
        self.slots = self.transpose_order = None
        self.crow = self.columns = self.transpose_crow = self.transpose_columns = None
        self.pattern_shape = None


def _pointers(rows,size):
    counts = torch.bincount(rows,minlength=size)
    return torch.cat((counts.new_zeros(1),counts.cumsum(0)))


def _merge_positions(old,new):
    # Arrays are individually sorted and disjoint. Neither is concatenated
    # into a global sort; each element has one deterministic merge position.
    old_pos = torch.arange(len(old),device=old.device)+torch.searchsorted(new,old,right=True)
    new_pos = torch.arange(len(new),device=new.device)+torch.searchsorted(old,new)
    return old_pos,new_pos


def _find_slots(keys,workspace):
    if keys.is_cuda:
        from ._triton_transfer_assembly import find_slots
        return find_slots(keys,workspace)
    slots = []
    for key in keys.tolist():
        slot = hash_slot(key,workspace.capacity)
        for _ in range(128):
            if int(workspace.keys[slot])==key:
                slots.append(slot)
                break
            slot = (slot+1)&(workspace.capacity-1)
        else:
            raise RuntimeError('cached IB slot lookup failed after completed insertion')
    return keys.new_tensor(slots)


def finish_cached_component(workspace,cache,node_count,shape,max_entries,*,missing=None):
    """Build once or merge only newly introduced keys into both sorted indices."""
    nf = prod(shape)
    if missing is None:
        occupied = torch.where(workspace.keys>=0)[0]
        if len(occupied)>max_entries:
            raise PatternBudgetExceeded('cached IB union exceeds entry budget')
        keys,order = torch.sort(workspace.keys[occupied])
        slots = occupied[order]
        rows,columns = keys//nf,keys%nf
        transpose_order = torch.argsort(columns*node_count+rows)
        crow = _pointers(rows,node_count)
        transpose_crow = _pointers(columns,nf)
        sorted_count = len(keys)
    else:
        old_keys = workspace.keys.index_select(0,cache.slots)
        new_keys = torch.unique(cache.missing_keys[:missing],sorted=True)
        at = torch.searchsorted(old_keys,new_keys)
        exists = ((at<len(old_keys)) & (old_keys[at.clamp_max(len(old_keys)-1)]==new_keys)
                  if len(old_keys) else torch.zeros_like(at,dtype=torch.bool))
        new_keys = new_keys[~exists]
        if cache.nnz+len(new_keys)>max_entries:
            raise PatternBudgetExceeded('cached IB union exceeds entry budget')
        if not len(new_keys):
            cache.last_sorted_keys = 0
            return numeric_snapshot(workspace,cache,node_count,shape)
        old_pos,new_pos = _merge_positions(old_keys,new_keys)
        keys = old_keys.new_empty(cache.nnz+len(new_keys))
        keys[old_pos],keys[new_pos] = old_keys,new_keys
        slots = keys.new_empty(len(keys))
        slots[old_pos] = cache.slots.to(torch.int64)
        slots[new_pos] = _find_slots(new_keys,workspace)
        rows,columns = keys//nf,keys%nf
        new_rows,new_columns = new_keys//nf,new_keys%nf
        # Prefix counts require only the new entries, avoiding a full-grid
        # bincount of millions of unchanged row/column indices.
        crow = cache.crow+_pointers(new_rows,node_count)
        transpose_crow = cache.transpose_crow+_pointers(new_columns,nf)
        old_reverse_keys = old_keys.index_select(0,cache.transpose_order)
        old_reverse_keys = old_reverse_keys%nf*node_count+old_reverse_keys//nf
        new_reverse_keys,new_reverse_order = torch.sort(new_columns*node_count+new_rows)
        old_reverse_pos,new_reverse_pos = _merge_positions(old_reverse_keys,new_reverse_keys)
        transpose_order = keys.new_empty(len(keys))
        transpose_order[old_reverse_pos] = old_pos.index_select(0,cache.transpose_order)
        transpose_order[new_reverse_pos] = new_pos[new_reverse_order]
        sorted_count = len(new_keys)
    values = workspace.values[slots]
    valid = (keys<nf*node_count).all() & torch.isfinite(values).all()
    if not valid.item():
        raise FloatingPointError('invalid cached IB entries: nonfinite values or out-of-range keys')
    # 32-bit mapping indices halve retained mapping traffic on normal grids.
    mapping_dtype = torch.int32 if max(workspace.capacity,len(keys))<2147483648 else torch.int64
    # Commit only after all validation/allocation has succeeded. Old stencils
    # retain their own values and immutable indices when this cache is replaced.
    transpose_columns = rows[transpose_order]
    compact_slots,compact_order = slots.to(mapping_dtype),transpose_order.to(mapping_dtype)
    cache.slots,cache.transpose_order = compact_slots,compact_order
    cache.crow,cache.columns = crow,columns
    cache.transpose_crow,cache.transpose_columns = transpose_crow,transpose_columns
    cache.pattern_shape = (node_count,*shape)
    cache.last_sorted_keys = sorted_count
    cache.total_sorted_keys += sorted_count
    cache.rebuilds += 1
    return numeric_snapshot(workspace,cache,node_count,shape,validate=False,values=values)


def numeric_snapshot(workspace,cache,node_count,shape,*,validate=True,values=None):
    # index_select creates independently owned values; never alias resettable
    # hash scratch. Transpose values come from exactly this same numeric vector.
    if values is None:
        values = workspace.values.index_select(0,cache.slots)
    if validate and not torch.isfinite(values).all().item():
        raise FloatingPointError('invalid cached IB entries: nonfinite values')
    reverse = values.index_select(0,cache.transpose_order)
    matrix = torch.sparse_csr_tensor(cache.crow,cache.columns,values,
        (node_count,prod(shape)),check_invariants=False)
    transpose = torch.sparse_csr_tensor(cache.transpose_crow,cache.transpose_columns,reverse,
        (prod(shape),node_count),check_invariants=False)
    return matrix,transpose


def _cpu_cached_add(keys,values,workspace,cache):
    # Independent serial lookup oracle; no CUDA tensor download is permitted.
    for key,value in zip(keys.tolist(),values.tolist()):
        if value==0:
            continue
        slot = hash_slot(key,workspace.capacity)
        for _ in range(128):
            previous = int(workspace.keys[slot])
            if previous==key:
                workspace.values[slot] += value
                break
            if previous==-1:
                break
            slot = (slot+1)&(workspace.capacity-1)
        else:
            previous = -1
        if previous!=key:
            index = int(cache.missing_count)
            if index<cache.missing_capacity:
                cache.missing_keys[index] = key
                cache.missing_values[index] = value
            cache.missing_count += 1


def update_cached_plans(stencil,c,plans,shape,workspace,cache,chunk_entries,*,contraction_backend='sites'):
    from .assembled_transfer import _cpu_entries
    launches = peak = 0
    for group,offset,low,width,prefix,count in plans:
        if group.weights.is_cuda:
            if contraction_backend=='cuda':
                from .cuda_ib import contract
                contract(stencil,group,c,offset,low,width,prefix,count,shape,workspace,cache)
            elif contraction_backend=='cell':
                from ._triton_cell_transfer_assembly import cell_entries
                cell_entries(stencil,group,c,offset,low,width,prefix,count,shape,workspace,cache)
            else:
                from ._triton_transfer_assembly import cached_entries
                cached_entries(stencil,group,c,offset,low,width,prefix,count,shape,workspace,cache)
            launches += 1
        else:
            for start in range(0,count,chunk_entries//4):
                size = min(chunk_entries//4,count-start)
                keys,values = _cpu_entries(stencil,group,c,offset,low,width,prefix,start,size,shape)
                _cpu_cached_add(keys,values,workspace,cache)
                launches += 1; peak = max(peak,len(keys))
    return launches,peak


def merge_missing(workspace,cache,count,*,contraction_backend='sites'):
    if count<1:
        return
    keys,values = cache.missing_keys[:count],cache.missing_values[:count]
    if values.is_cuda:
        if contraction_backend=='cuda':
            from .cuda_ib import hash_accumulate
        else:
            from ._triton_transfer_assembly import hash_accumulate
        hash_accumulate(keys,values,workspace)
    else:
        cpu_accumulate(keys,values,workspace)


def assemble_component_cached(stencil,c,grid,node_count,*,chunk_entries,max_entries,workspace,cache,contraction_backend='sites'):
    shape,plans,raw = component_plans(stencil,c,grid)
    like = stencil.rule.groups[0].weights
    attempts = launches = peak = missing = 0
    reused = False
    reason = 'initial'
    # A single bounded stream holds only contributions missing from the old
    # pattern. It never stores the full raw cell-entry stream on CUDA.
    cache.scratch(chunk_entries,like)
    compatible = (cache.slots is not None and workspace.values is not None and
                  workspace.values.device==like.device and workspace.values.dtype==like.dtype and
                  cache.pattern_shape==(node_count,*shape))
    if compatible and cache.nnz<=max_entries:
        workspace.values.zero_(); workspace.flag.zero_(); workspace.reuses += 1
        calls,peak = update_cached_plans(stencil,c,plans,shape,workspace,cache,chunk_entries,
                                       contraction_backend=contraction_backend)
        launches += calls
        missing = int(cache.missing_count.item())
        cache.total_missing_entries += missing
        if missing<=cache.missing_capacity:
            # No reader of the key table remains when mutable insertion starts.
            try:
                if contraction_backend=='cuda':
                    merge_missing(workspace,cache,missing,contraction_backend='cuda')
                else:
                    merge_missing(workspace,cache,missing)
            except Exception:
                cache.invalidate()
                raise
            # The immutable-key pass cannot set the insertion overflow flag.
            if not missing or not workspace.flag.item():
                if missing:
                    try:
                        matrix,transpose = finish_cached_component(workspace,cache,node_count,shape,max_entries,missing=missing)
                        cache.extensions += 1
                        reason = 'support-extension'
                    except PatternBudgetExceeded:
                        reason = 'union-budget-reset'
                    except Exception:
                        # Insertion has extended the key table, but a rejected
                        # finalization has not published its new CSR mapping.
                        cache.invalidate()
                        raise
                else:
                    matrix,transpose = numeric_snapshot(workspace,cache,node_count,shape)
                    cache.reuses += 1
                    reason,reused = 'numeric-only',True
                if reason in ('support-extension','numeric-only'):
                    return matrix,transpose,_statistics(workspace,cache,raw,launches,peak,
                        missing,reused,reason,attempts)
            else:
                reason = 'probe-overflow-reset'
                cache.overflow_fallbacks += 1
        else:
            reason = 'missing-stream-overflow-reset'
            cache.overflow_fallbacks += 1
    elif compatible:
        reason = 'union-budget-reset'
    if cache.slots is not None:
        cache.resets += 1
    # A rejected rebuild must not leave an old mapping pointing into scratch
    # whose keys have subsequently been cleared or reallocated.
    cache.invalidate()
    # Any overflowed/over-budget partial update is discarded in full. Rebuild
    # the CURRENT quadrature matrix with the original bounded hash algorithm.
    limit = power_of_two(2*max_entries)
    capacity = min(limit,max(workspace.capacity,power_of_two(max(128,raw//8))))
    while True:
        workspace.reset(capacity,like)
        attempts += 1
        calls,batch_peak = accumulate_plans(stencil,c,plans,shape,workspace,chunk_entries,
                                           contraction_backend=contraction_backend)
        launches += calls; peak = max(peak,batch_peak)
        if not workspace.flag.item():
            break
        if capacity>=limit:
            raise RuntimeError('assembled IB hash probe/budget exceeded; no cached entries were truncated')
        capacity = min(limit,2*capacity)
    try:
        matrix,transpose = finish_cached_component(workspace,cache,node_count,shape,max_entries)
    except PatternBudgetExceeded as exc:
        raise RuntimeError('assembled IB entry budget exceeded; no cached entries were truncated') from exc
    return matrix,transpose,_statistics(workspace,cache,raw,launches,peak,
        missing,reused,reason,attempts)


def _statistics(workspace,cache,raw,launches,peak,missing,reused,reason,attempts):
    return dict(raw_cell_entries=raw,batches=launches,maximum_batch_entries=peak,
        nnz=cache.nnz,hash_capacity=workspace.capacity,hash_attempts=attempts,
        hash_workspace_bytes=workspace.storage_bytes,hash_workspace_allocations=workspace.allocations,
        hash_workspace_reuses=workspace.reuses,raw_entries_materialized=not workspace.values.is_cuda,
        unique_key_sorts=0 if reused else 1,transpose_key_sorts=0 if reused else 1,
        duplicate_merge_sorts=0,symbolic_reused=reused,symbolic_reason=reason,
        sorted_unique_entries=0 if reused else cache.last_sorted_keys,
        sort_scope='none' if reused else 'new-keys-only' if reason=='support-extension' else 'current-pattern',
        symbolic_rebuilds=cache.rebuilds,symbolic_reuses=cache.reuses,
        symbolic_extensions=cache.extensions,symbolic_resets=cache.resets,
        missing_entries=missing,missing_capacity=cache.missing_capacity,
        overflow_fallbacks=cache.overflow_fallbacks,symbolic_cache_bytes=cache.storage_bytes)
