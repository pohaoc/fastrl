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
"""

import os
from contextlib import contextmanager, nullcontext

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH


def enabled(server_args) -> bool:
    return bool(
        getattr(server_args, "enable_memory_saver", False)
        and getattr(server_args, "speculative_algorithm", None)
        and os.environ.get("FASTRL_RELEASE_SD_MEMORY", "1") != "0"
    )


@contextmanager
def graph_capture_region(memory_saver_adapter, server_args):
    """Use inside a torch.cuda.graph(...) block, around the captured forward (as torch_memory_saver.cuda_graph
    does), so the graph pool's allocations become pausable."""
    ctx = (
        memory_saver_adapter.region(GPU_MEMORY_TYPE_CUDA_GRAPH)
        if enabled(server_args) and memory_saver_adapter.enabled
        else nullcontext()
    )
    with ctx:
        yield
