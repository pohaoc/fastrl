---
name: fastrl-env-setup
description: How the FastRL training env on this box was built (2026-09-28) and the non-obvious fixes it needed
metadata:
  type: project
---
Env set up 2026-09-28: `/home/cc/fastrl/.venv` (system Python 3.12, no conda on box). Activate with `source .venv/bin/activate`; activate script exports CUDA_HOME=/home/cc/cuda-12.8.
Eurus data: `/home/cc/fastrl/Eurus-2-RL-Data/{train,validation}.parquet` from HF `PRIME-RL/Eurus-2-RL-Data`.

Non-obvious fixes:
- flashinfer_python==0.4.0 (pinned by sglang 0.5.3.post2) is sdist-only and needs apache-tvm-ffi==0.1.0b15, which is gone from PyPI. 0.1.0 final is API-incompatible (JIT compile errors). Fixed by building tvm-ffi from git commit 7092774 (apache/tvm-ffi) into a wheel.
- JIT needs nvcc matching torch cu12.8; system only has /usr/local/cuda-13.4. CUDA 12.8.1 toolkit installed via runfile to /home/cc/cuda-12.8 (user-owned, no sudo).
- verl requires numpy<2; pinned scipy==1.15.3, contourpy==1.3.3. pyext not needed (import commented out).
- flash-attn wheel must be installed with --no-deps or it upgrades torch.

**Why:** rebuilding from README steps alone fails on this machine.
**How to apply:** reuse the venv; if rebuilding, apply these fixes rather than rediscovering them.
