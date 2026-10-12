"""Release the speculative-decoding engine's extra GPU memory while RL training runs (FastRL).

verl pauses the engine with tags ["kv_cache", "weights"]. Without this module the EAGLE drafter's weights and
every CUDA graph stay resident during training: the drafter is loaded outside any memory-saver region and graphs
are captured with plain torch.cuda.graph. On a 44 GB L40S that left ~5 GB in use and the actor update ran out of
memory (SkyRL-SQL, 35-step run, job 7097953).

When it applies (memory saver on, a speculative algorithm set, FASTRL_RELEASE_SD_MEMORY != "0"):
  - drafter weights are allocated in the "weights_drafter" region with a CPU backup (verl never re-pushes them);
  - CUDA graphs are captured in the "cuda_graph" region, without backup (graph memory is scratch that each
    replay rewrites; the graphs' static input buffers live outside the pool);
  - both are paused together with "weights" and resumed together with it.
Engines without speculation are unchanged.

torch_memory_saver (0.0.9rc2) serves every region from ONE memory pool and tags memory per cudaMalloc'd segment, so
the caching allocator can place drafter tensors in free space of a segment the target's "weights" region created.
verl runs without a target CPU backup and re-pushes only the target's own weights, so such drafter tensors came
back as garbage: acceptance 1.02 instead of 3.91 (sd_accept_check 7301325; ~0 accepted drafts in RL, job 7283745).
So we also keep a CPU copy of the drafter's own parameters and buffers (taken before the first pause) and copy
it back in place after every resume. Tensors shared with the target (embedding, lm_head) are skipped: verl
pushes them.

CUDA-graph release is OPT-IN (FASTRL_RELEASE_SD_GRAPHS=1): capturing graphs inside a torch_memory_saver region
(0.0.9rc2, preload hook) broke the caching allocator in job 7170795 (torch.cuda.empty_cache -> "CUDA error:
invalid argument" right after the first MAB graph group was captured). Drafter-weight release stays on.
"""

import os
from contextlib import contextmanager, nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH

_drafter_snapshot = None  # [(gpu tensor, pinned cpu copy)] for the drafter's own parameters and buffers


def enabled(server_args) -> bool:
    return bool(
        getattr(server_args, "enable_memory_saver", False)
        and getattr(server_args, "speculative_algorithm", None)
        and os.environ.get("FASTRL_RELEASE_SD_MEMORY", "1") != "0"
    )


def graphs_enabled(server_args) -> bool:
    return enabled(server_args) and os.environ.get("FASTRL_RELEASE_SD_GRAPHS", "0") == "1"


@contextmanager
def graph_capture_region(memory_saver_adapter, server_args):
    """Use inside a torch.cuda.graph(...) block, around the captured forward (as torch_memory_saver.cuda_graph
    does), so the graph pool's allocations become pausable."""
    ctx = (
        memory_saver_adapter.region(GPU_MEMORY_TYPE_CUDA_GRAPH)
        if graphs_enabled(server_args) and memory_saver_adapter.enabled
        else nullcontext()
    )
    with ctx:
        yield


def snapshot_drafter(draft_model, target_model):
    """Before the first pause: CPU copies of the drafter's own tensors (not the ones it shares with the target)."""
    global _drafter_snapshot
    if _drafter_snapshot is not None:
        return
    target_ptrs = {t.data_ptr() for t in [*target_model.parameters(), *target_model.buffers()]}
    seen, snap = set(), []
    for t in [*draft_model.parameters(), *draft_model.buffers()]:
        ptr = t.data_ptr()
        if not t.is_cuda or ptr in target_ptrs or ptr in seen:
            continue
        seen.add(ptr)
        snap.append((t, t.detach().to("cpu", copy=True).pin_memory()))
    _drafter_snapshot = snap


def restore_drafter():
    """After resuming the weights: copy the snapshot back in place (same addresses, so CUDA graphs stay valid)."""
    if _drafter_snapshot is None:
        return
    with torch.no_grad():
        for t, cpu in _drafter_snapshot:
            t.data.copy_(cpu, non_blocking=True)
    torch.cuda.synchronize()
