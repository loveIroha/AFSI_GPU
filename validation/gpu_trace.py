"""Diagnostic ranges and Chrome trace analysis; never used by production steps."""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
from math import isfinite
import torch


class RangeRecorder:
    """Annotate nested calls without events, synchronization or tensor inspection."""
    def __init__(self, *, nvtx=False):
        self.nvtx = nvtx
        self.calls = defaultdict(int)

    @contextmanager
    def range(self, label):
        self.calls[label] += 1
        with torch.profiler.record_function('afsi.'+label):
            if self.nvtx:
                torch.cuda.nvtx.range_push('afsi.'+label)
            try:
                yield
            finally:
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()

    def wrap(self, fn, label):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with self.range(label):
                return fn(*args, **kwargs)
        return wrapped


def union_duration(intervals):
    """Union length in the input units, including overlapping stream intervals."""
    total = 0.
    end = None
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if end is None or start >= end:
            total += stop-start
        elif stop > end:
            total += stop-end
        end = stop if end is None else max(end, stop)
    return total


def summarize_trace(path):
    """Summarize measured GPU activities, not estimated bytes or occupancy.

    The capture range includes its final synchronize. Union, rather than the
    sum of durations, handles overlapping streams. Missing graph/CUPTI events
    must not be interpreted as proof of an idle GPU.
    """
    with open(path, encoding='utf-8') as stream:
        raw = json.load(stream)
    events = raw.get('traceEvents', []) if isinstance(raw, dict) else raw
    complete = []
    for e in events:
        if e.get('ph') != 'X':
            continue
        ts, dur = e.get('ts'), e.get('dur')
        if (not isinstance(ts, (int, float)) or not isinstance(dur, (int, float))
                or not isfinite(ts) or not isfinite(dur) or dur < 0):
            continue
        complete.append(e)
    captures = [(e['ts'], e['ts']+e['dur']) for e in complete if e.get('name')=='afsi.capture']
    kernel_intervals, activity_intervals = [], []
    kernels = defaultdict(lambda: dict(calls=0, total_us=0., max_us=0.))
    synchronizations = defaultdict(lambda: dict(calls=0, total_us=0.))
    launch_calls = 0
    for e in complete:
        start, stop = e['ts'], e['ts']+e['dur']
        clipped = [(max(start,a), min(stop,b)) for a,b in captures if start < b and stop > a]
        if not clipped:
            continue
        name, cat = e.get('name',''), e.get('cat','').lower()
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            activity_intervals.extend(clipped)
        if cat=='kernel':
            kernel_intervals.extend(clipped)
            values = kernels[name]
            duration = union_duration(clipped)
            values['calls'] += 1
            values['total_us'] += duration
            values['max_us'] = max(values['max_us'],duration)
        if cat in ('cuda_runtime', 'cuda_driver'):
            if 'launch' in name.lower():
                launch_calls += 1
            if 'synchronize' in name.lower():
                values = synchronizations[name]
                values['calls'] += 1
                values['total_us'] += union_duration(clipped)
    span = union_duration(captures)
    activity = union_duration(activity_intervals)
    return dict(capture_ms=span/1000, gpu_kernel_union_ms=union_duration(kernel_intervals)/1000,
        gpu_activity_union_ms=activity/1000,
        capture_without_recorded_gpu_activity_ms=max(0.,span-activity)/1000 if activity_intervals else None,
        gpu_events_available=bool(activity_intervals), gpu_kernel_launches=sum(v['calls'] for v in kernels.values()),
        host_launch_api_calls=launch_calls,
        kernels=sorted([dict(name=k, calls=v['calls'], total_ms=v['total_us']/1000,
                            max_ms=v['max_us']/1000) for k,v in kernels.items()],key=lambda v:-v['total_ms']),
        host_synchronizations=[dict(name=k,calls=v['calls'],total_ms=v['total_us']/1000)
                               for k,v in synchronizations.items()],
        interpretation='Profiled window only. Kernel totals can overlap across streams. Uncovered time is not a measurement of CPU overhead or hardware GPU utilization; missing CUDA graph/CUPTI events can undercount activity. No bandwidth, register or atomic-throughput inference is made.')
