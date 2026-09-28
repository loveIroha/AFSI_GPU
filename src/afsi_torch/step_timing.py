"""Optional per-stage timing for the coupled LV step.

CUDA events stay on the active stream and are read only at report checkpoints.
This avoids a new host/device synchronization at every stage boundary.
"""
from collections import defaultdict
from time import perf_counter

import torch


class StepTimingRecorder:
    def __init__(self, device, previous=None):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.seconds = defaultdict(float, (previous or {}).get('stage_seconds', {}))
        self.steps = (previous or {}).get('measured_steps', 0)
        self.wall_seconds = (previous or {}).get('wall_step_seconds', 0.)
        self.pending = []
        self._last = None
        self._wall_start = None
        self._pending_start = 0
        self._step_seconds = defaultdict(float)

    def begin(self):
        if self._last is not None:
            raise RuntimeError('a timed step is already active')
        self._wall_start = perf_counter()
        self._pending_start = len(self.pending)
        if self.cuda:
            self._last = torch.cuda.Event(enable_timing=True)
            self._last.record(torch.cuda.current_stream(self.device))
        else:
            self._last = self._wall_start

    def mark(self, stage):
        if self._last is None:
            raise RuntimeError('no timed step is active')
        if self.cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self.pending.append((stage, self._last, end))
        else:
            end = perf_counter()
            self._step_seconds[stage] += end-self._last
        self._last = end

    def finish(self):
        self.mark('other_coupling')
        for stage, seconds in self._step_seconds.items():
            self.seconds[stage] += seconds
        self._step_seconds.clear()
        self.wall_seconds += perf_counter()-self._wall_start
        self.steps += 1
        self._last = self._wall_start = None

    def abort(self):
        """Exclude an unaccepted step while preserving prior completed steps."""
        del self.pending[self._pending_start:]
        self._step_seconds.clear()
        self._last = self._wall_start = None

    def snapshot(self):
        if self._last is not None:
            raise RuntimeError('cannot read timing inside a step')
        if self.pending:
            torch.cuda.synchronize(self.device)
            for stage, start, end in self.pending:
                self.seconds[stage] += start.elapsed_time(end)/1000.
            self.pending.clear()
        measured = sum(self.seconds.values())
        return dict(clock='cuda_event' if self.cuda else 'cpu_perf_counter',
                    measured_steps=self.steps, wall_step_seconds=self.wall_seconds,
                    mean_wall_step_ms=1000*self.wall_seconds/self.steps if self.steps else 0.,
                    stage_seconds=dict(sorted(self.seconds.items())),
                    stage_fraction={name: value/measured for name, value in sorted(self.seconds.items())}
                    if measured else {})
