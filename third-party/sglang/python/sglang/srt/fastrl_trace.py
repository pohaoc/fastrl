"""Lightweight GPU timeline tracer for FastRL rollout analysis.

Enabled only when the ``FASTRL_TRACE_DIR`` environment variable is set. Spans are
bracketed with CUDA events (no extra host synchronization), resolved lazily once
the GPU has passed them, converted to wall-clock time via an anchor event, and
appended as JSON lines to ``$FASTRL_TRACE_DIR/sgl_<pid>.jsonl``.
"""

import json
import os
import socket
import time
from contextlib import contextmanager

import torch

_TRACE_DIR = os.environ.get("FASTRL_TRACE_DIR")


class _GpuTracer:
    def __init__(self):
        self.enabled = bool(_TRACE_DIR)
        self._initialized = False
        self._pending = []  # (name, start_event, end_event, attrs)
        self._buffer = []
        self._last_flush = time.time()

    def _lazy_init(self):
        if self._initialized:
            return
        self._initialized = True
        os.makedirs(_TRACE_DIR, exist_ok=True)
        device = torch.cuda.current_device()
        self._device_uuid = str(torch.cuda.get_device_properties(device).uuid)
        self._file = open(
            os.path.join(_TRACE_DIR, f"sgl_{socket.gethostname()}_{os.getpid()}.jsonl"), "a"
        )
        self._reanchor()

    def _reanchor(self):
        # Pair a CUDA event with a host timestamp while the device is idle.
        torch.cuda.synchronize()
        self._anchor_event = torch.cuda.Event(enable_timing=True)
        self._anchor_event.record()
        torch.cuda.synchronize()
        self._anchor_wall = time.time()

    @contextmanager
    def span(self, name, **attrs):
        """Time the GPU work enqueued inside the block. ``attrs`` may be updated in-place."""
        if not self.enabled:
            yield attrs
            return
        self._lazy_init()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield attrs
        finally:
            end.record()
            self._pending.append((name, start, end, attrs))
            self.poll()

    def poll(self):
        if not self.enabled or not self._initialized:
            return
        while self._pending and self._pending[0][2].query():
            name, start, end, attrs = self._pending.pop(0)
            t0 = self._anchor_wall + self._anchor_event.elapsed_time(start) / 1e3
            t1 = self._anchor_wall + self._anchor_event.elapsed_time(end) / 1e3
            self._buffer.append(
                {"src": "sglang", "gpu_uuid": self._device_uuid, "pid": os.getpid(),
                 "name": name, "start": t0, "end": t1, **attrs}
            )
        if self._buffer and time.time() - self._last_flush > 2.0:
            self.flush()

    def flush(self, reanchor=False):
        """Write resolved spans; optionally re-anchor the clock (call only when idle)."""
        if not self.enabled or not self._initialized:
            return
        if reanchor:
            if not self._pending and not self._buffer:
                return
            torch.cuda.synchronize()
            self.poll()
            self._reanchor()
        for rec in self._buffer:
            self._file.write(json.dumps(rec) + "\n")
        self._file.flush()
        self._buffer = []
        self._last_flush = time.time()


tracer = _GpuTracer()
