"""Host-side allocator counters sampled at existing output/checkpoint cadence."""
import torch


def allocator_sample(device):
    device=torch.device(device)
    if device.type!='cuda':
        return {}
    stats=torch.cuda.memory_stats(device)
    return dict(gpu_allocated_bytes=stats.get('allocated_bytes.all.current',0),
                gpu_reserved_bytes=stats.get('reserved_bytes.all.current',0),
                gpu_peak_allocated_bytes=stats.get('allocated_bytes.all.peak',0),
                gpu_peak_reserved_bytes=stats.get('reserved_bytes.all.peak',0),
                gpu_inactive_split_bytes=stats.get('inactive_split_bytes.all.current',0),
                gpu_allocation_retries=stats.get('num_alloc_retries',0),
                gpu_oom_count=stats.get('num_ooms',0))
