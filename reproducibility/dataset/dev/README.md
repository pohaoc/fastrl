# Development scripts

One-off scripts used while building the artifact; the canonical tools are one level up.

| File | Purpose |
| --- | --- |
| `test_skyrl_loop.py` | Standalone check of the SkyRL-SQL agent loop against a real SGLang engine (1 GPU, no SD) |
| `chain_after_sd_on.sh` | Waits for an SD-on run to finish, then runs SD-off and the analysis (paths under `experiments/sd_breakdown`) |
| `bench_sd_data.sh` | `examples/bench_sd.sh` with `MODEL_PATH` / `DATA_PATH` overrides (superseded by `../bench_acceptance.sh`) |
