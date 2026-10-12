#!/bin/bash
# Does speculative decoding change DeepSeek-R1-Distill-Llama-8B's DAPO outputs? (see dapo_sd_quality.py)
# Four single-GPU engines side by side, same prompts: off, sd_rl (as in RL, with the switch fix), sd_always (no
# adaptive switch), sd_rl_nofix (FASTRL_TRANSITION_FIX=0). VARIANTS overrides the list (one per GPU). Each: 8 greedy
# prompts x 4,096 tokens, then 64 prompts x 4 samples at T 1.0 up to 32,000 tokens (one GPU's share of an RL
# step). 2 h 45 min cap per engine. Per-iteration scheduler logs in $OUT/steplog_<variant>.
#   sbatch reproducibility/oscar/dapo_sd_quality.sbatch       (submit from the repo root)
#SBATCH --job-name=dapo-sd-quality
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --output=/oscar/scratch/%u/jobs/%x-%j.out
set -uo pipefail
REPO=${FASTRL_REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"
source reproducibility/oscar/env.sh
OUT=$RUN_ROOT/dapo_sd_quality/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date) commit=$(git rev-parse --short HEAD)"
git status --short
gpu_preflight
MODEL=$(python -c "from huggingface_hub import snapshot_download as s; print(s('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', revision='6a6f4aa4197940add57724a7707d069478df56b1'))")
DRAFTER=$SCRATCH_ROOT/eagle/DeepSeek-R1-Distill-Llama-8B/drafters/DeepSeek-R1-Distill-Llama-8B-Eagle-dapo
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=true

run_on() {  # gpu variant  (own process group, capped)
    local gpu=$1 name=$2
    local fix=1; [ "$name" = sd_rl_nofix ] && fix=0
    FASTRL_TRANSITION_FIX=$fix FASTRL_STEPLOG_DIR=$OUT/steplog_$name CUDA_VISIBLE_DEVICES=$gpu setsid python reproducibility/oscar/dapo_sd_quality.py \
        --out "$OUT/$name.json" --variant "$name" --model "$MODEL" --drafter "$DRAFTER" > "$OUT/$name.log" 2>&1 &
    local pid=$!
    ( sleep 9900; echo "TIMEOUT $name"; kill -TERM -- -$pid; sleep 20; kill -9 -- -$pid ) 2>/dev/null & local dog=$!
    wait $pid; local rc=$?
    kill $dog 2>/dev/null; kill -9 -- -$pid 2>/dev/null
    echo "$name exit=$rc $(date +%T)"
}
VARIANTS=(${VARIANTS:-off sd_rl sd_always sd_rl_nofix})
for i in "${!VARIANTS[@]}"; do run_on "$i" "${VARIANTS[$i]}" & done
wait

echo; echo "=================== SUMMARY"
python - "$OUT" <<'EOF'
import json, os, sys
d = sys.argv[1]
res = {}
for v in ("off", "sd_rl", "sd_always", "sd_rl_nofix", "sd_triton", "sd_always_64", "sd_always_256", "sd_eager_32"):
    p = os.path.join(d, v + ".json")
    if os.path.exists(p):
        res[v] = json.load(open(p))
print("sampled (64 prompts x 4, T 1.0, <= 32,000 tokens):")
for v, r in res.items():
    print(f"  {v:10s} phase={r.get('phase')} {r.get('summary')}")
ref = res.get("off", {}).get("greedy") or []
print("greedy (8 prompts, 4,096 tokens): index of the first token that differs from `off` (None = identical)")
for v, r in res.items():
    g = r.get("greedy") or []
    if v == "off" or not g or not ref:
        continue
    first = []
    for a, b in zip(ref, g):
        x, y = a["ids"], b["ids"]
        first.append(next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None if len(x) == len(y) else min(len(x), len(y))))
    print(f"  {v:10s} {first}")
EOF
echo "end=$(date)"
