#!/bin/bash
# Does the SD engine release its drafter and CUDA graphs during training, and restore them exactly?
# (fastrl_sd_memory.py; motivated by the SD-on OOM at step 7 of job 7097953). Three variants on one node:
#   sd_old : SD on, FASTRL_RELEASE_SD_MEMORY=0 (previous behaviour)
#   sd_new : SD on, the fix
#   off    : SD off (reference residue)
#   sbatch reproducibility/oscar/sd_memory_check.sbatch       (submit from the repo root)
#SBATCH --job-name=sd-memory-check
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
OUT=$RUN_ROOT/sd_memory_check/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD)"
git status --short
gpu_preflight
MODEL=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
DRAFTER=$SCRATCH_ROOT/eagle/Qwen2.5-Coder-7B-Instruct/drafters/Qwen2.5-Coder-7B-Instruct-Eagle-sql
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true

variant() {  # name RELEASE_FLAG sd
    local name=$1
    echo; echo "=================== $name ($(date +%T))"
    FASTRL_RELEASE_SD_MEMORY=$2 setsid python reproducibility/oscar/sd_memory_check.py --out "$OUT/$name.json" \
        --model "$MODEL" --drafter "$DRAFTER" --sd "$3" > "$OUT/$name.log" 2>&1 &
    local pid=$!
    ( sleep 1200; echo "TIMEOUT $name"; kill -TERM -- -$pid ) 2>/dev/null & local dog=$!
    wait $pid; local rc=$?
    kill $dog 2>/dev/null; kill -9 -- -$pid 2>/dev/null; sleep 5
    grep -a "^RESULT" "$OUT/$name.log" || tail -30 "$OUT/$name.log"
    echo "=================== $name exit=$rc"
}
variant sd_old 0 1
variant sd_new 1 1
variant off    1 0

echo; echo "=================== SUMMARY (MiB per GPU; released = what training gets back)"
python - "$OUT" <<'EOF'
import json, sys, os
for n in ("sd_old", "sd_new", "off"):
    p = os.path.join(sys.argv[1], n + ".json")
    if not os.path.exists(p):
        print(f"{n:7} no result"); continue
    r = json.load(open(p))
    if r.get("phase") != "done":
        print(f"{n:7} FAILED in phase {r.get('phase')}"); continue
    m = r["mem_mib"]
    print(f"{n:7} active {m['active']}  released {m['released']}  resumed {m['resumed']}  "
          f"greedy identical {r['greedy_identical']}/{r['greedy_total']}  accept {r['accept_before']} -> {r['accept_after']}  "
          f"release {r['release_s']:.2f} s  resume {r['resume_s']:.2f} s")
EOF
echo "end=$(date)"
