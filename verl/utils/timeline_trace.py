"""Wall-clock timeline spans for per-GPU Gantt analysis.

Enabled only when ``FASTRL_TRACE_DIR`` is set. Each process appends JSON lines to
``$FASTRL_TRACE_DIR/verl_<host>_<pid>.jsonl``. GPU processes synchronize the device at
span boundaries so that the span covers the GPU work it launched.
"""

import functools
import json
import os
import socket
import time
from contextlib import contextmanager

_TRACE_DIR = os.environ.get("FASTRL_TRACE_DIR")
_file = None
_device_uuid = None


def _get_file():
    global _file
    if _file is None:
        os.makedirs(_TRACE_DIR, exist_ok=True)
        _file = open(os.path.join(_TRACE_DIR, f"verl_{socket.gethostname()}_{os.getpid()}.jsonl"), "a")
    return _file


def _gpu_uuid():
    global _device_uuid
    if _device_uuid is None:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            _device_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return _device_uuid


@contextmanager
def trace_span(name: str, gpu: bool = False, **attrs):
    if not _TRACE_DIR:
        yield attrs
        return
    if gpu:
        import torch

        torch.cuda.synchronize()
    start = time.time()
    try:
        yield attrs
    finally:
        if gpu:
            import torch

            torch.cuda.synchronize()
        rec = {
            "src": "verl",
            "pid": os.getpid(),
            "rank": int(os.environ.get("RANK", -1)),
            "gpu_uuid": _gpu_uuid() if gpu else None,
            "name": name,
            "start": start,
            "end": time.time(),
            **attrs,
        }
        f = _get_file()
        f.write(json.dumps(rec) + "\n")
        f.flush()


def emit_span(name: str, start: float, end: float, **attrs):
    """Record an already-timed wall-clock span (no device sync); no-op unless tracing is on."""
    if not _TRACE_DIR:
        return
    rec = {"src": "verl", "pid": os.getpid(), "rank": int(os.environ.get("RANK", -1)), "gpu_uuid": None,
           "name": name, "start": start, "end": end, **attrs}
    f = _get_file()
    f.write(json.dumps(rec) + "\n")
    f.flush()


def traced_gpu_method(name: str):
    """Decorator recording a per-GPU span around a worker method."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with trace_span(name, gpu=True):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
