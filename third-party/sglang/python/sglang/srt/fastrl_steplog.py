"""Per-iteration scheduler log for FastRL speculative-decoding ablations.

Enabled only when ``FASTRL_STEPLOG_DIR`` is set. TP rank 0 of each engine appends one JSON line per
``Scheduler.run_batch`` call to ``$FASTRL_STEPLOG_DIR/steps_<pid>.jsonl``: host wall time at entry and
exit, forward mode, batch size, queue length, prefill tokens, CUDA-graph use, and for EAGLE engines whether
the step speculated and how many draft tokens were accepted. Host timestamps work the same with and without
the overlap scheduler, so the time between consecutive entries is the per-iteration cost in every config.
"""

import json
import os
import time

_DIR = os.environ.get("FASTRL_STEPLOG_DIR")


class _StepLog:
    def __init__(self):
        self.enabled = bool(_DIR)
        self._file = None

    def record(self, rec):
        if self._file is None:
            os.makedirs(_DIR, exist_ok=True)
            self._file = open(os.path.join(_DIR, f"steps_{os.getpid()}.jsonl"), "a")
        self._file.write(json.dumps(rec) + "\n")
        self._file.flush()  # the engine process is killed at shutdown; keep every step


steplog = _StepLog()
now = time.time
