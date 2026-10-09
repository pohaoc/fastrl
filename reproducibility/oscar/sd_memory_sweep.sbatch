#!/bin/bash
# Pick one mem_fraction_static for both SQL arms at TP 1 (SysX config): the highest at which the SD engine still
# starts (the drafter + speculation CUDA graphs must fit in what is left), with the KV pool and the memory left
# resident during training for SD on/off at each fraction. Drafter-weight release is on (fastrl_sd_memory.py).
# Eight single-GPU engines, four at a time (one per GPU), each with a 15 min cap (the TP-1 smoke run hung on OOM).
#   sbatch reproducibility/oscar/sd_memory_sweep.sbatch       (submit from the repo root)
#SBATCH --job-name=sd-memory-sweep
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:l40s:4
#SBATCH --constraint=l40s
#SBATCH --exclude=gpu3001                 # uncorrectable ECC errors (2026-09-30, 2026-10-04)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
OUT=$RUN_ROOT/sd_memory_sweep/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD)"
git status --short
gpu_preflight
MODEL=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-Coder-7B-Instruct', revision='c03e6d358207e414f1eca0bb1891e29f1db0e242'))")
DRAFTER=$SCRATCH_ROOT/eagle/Qwen2.5-Coder-7B-Instruct/drafters/Qwen2.5-Coder-7B-Instruct-Eagle-sql
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true

run_on() {  # gpu name sd mem  (own process group, so a hung or dead engine's children are killed with it)
    local gpu=$1 name=$2
    CUDA_VISIBLE_DEVICES=$gpu setsid python reproducibility/oscar/sd_memory_check.py \
        --out "$OUT/$name.json" --model "$MODEL" --drafter "$DRAFTER" --sd "$3" --tp 1 --mem "$4" \
        --chunked_prefill 8192 --smi_index "$gpu" > "$OUT/$name.log" 2>&1 &
    local pid=$!
    ( sleep 900; echo "TIMEOUT $name"; kill -TERM -- -$pid; sleep 20; kill -9 -- -$pid ) 2>/dev/null & local dog=$!
    wait $pid; local rc=$?
    kill $dog 2>/dev/null; kill -9 -- -$pid 2>/dev/null
    echo "$name exit=$rc"
}
wave() {
    echo "=================== wave: $* ($(date +%T))"
    local i=0 spec
    for spec in "$@"; do
        set -- ${spec//:/ }
        run_on "$i" "$1" "$2" "$3" &
        i=$((i + 1))
    done
    wait
}
wave sd_085:1:0.85 sd_080:1:0.80 sd_075:1:0.75 off_085:0:0.85
wave sd_070:1:0.70 off_080:0:0.80 off_075:0:0.75 off_070:0:0.70

echo; echo "=================== SUMMARY (TP 1, one GPU each; MiB; released = left resident while training)"
python - "$OUT" <<'EOF'
import json, sys, os
for n in ("sd_085", "sd_080", "sd_075", "sd_070", "off_085", "off_080", "off_075", "off_070"):
    p = os.path.join(sys.argv[1], n + ".json")
    r = json.load(open(p)) if os.path.exists(p) else {}
    if r.get("phase") != "done":
        print(f"{n:8} FAILED in phase {r.get('phase', 'startup')} (see {n}.log)"); continue
    m, e = r["mem_mib"], r.get("engine", {})
    print(f"{n:8} KV tokens {e.get('max_total_num_tokens')!s:>8}  active {m['active'][0]:6d}  released {m['released'][0]:6d}  "
          f"resumed {m['resumed'][0]:6d}  greedy identical {r['greedy_identical']}/{r['greedy_total']}  "
          f"accept {r['accept_before']} -> {r['accept_after']}")
EOF
echo "end=$(date)"
