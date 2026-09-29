#!/bin/bash
# Drafter acceptance check: EAGLE accept length on 32 sampled prompts of a dataset (batch size 1).
# Usage: DATASET=eurus|dapo|sql [MODEL_PATH=Qwen/Qwen2.5-7B] [TP=2] [DATA_ROOT=<repo root>] \
#     bash reproducibility/dataset/bench_acceptance.sh
# Same settings as examples/bench_sd.sh (8 draft steps, top-4, 48 draft tokens, TP=2), except an
# 8k context so that SQL prompts (schema included) fit. The stats CSV is written to $OUT/bench.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
DATASET=${DATASET:?set DATASET=eurus, dapo or sql}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B}
OUT=${OUT:-$HERE/outputs}
DATA_ROOT=${DATA_ROOT:-$REPO}
TP=${TP:-2}
mkdir -p "$OUT/bench"
case $DATASET in
  eurus) PARQUET=$DATA_ROOT/Eurus-2-RL-Data/train.parquet ;;
  dapo) PARQUET=$DATA_ROOT/DAPO-Math-17k/train.parquet ;;
  sql) PARQUET=$DATA_ROOT/SkyRL-SQL/train.parquet ;;
  *) echo "DATASET must be eurus, dapo or sql"; exit 1 ;;
esac
PROMPTS=$OUT/bench/${DATASET}_$(basename "$MODEL_PATH")_prompts.json
python - "$PARQUET" "$MODEL_PATH" "$PROMPTS" <<'EOF'
import json, sys
import pandas as pd
from transformers import AutoTokenizer
parquet, model, out = sys.argv[1:]
tok = AutoTokenizer.from_pretrained(model)
df = pd.read_parquet(parquet).sample(32, random_state=0)
json.dump([tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=False) for m in df.prompt], open(out, "w"))
EOF
cd "$OUT/bench"
BENCH_CONTEXT_LENGTH=8192 python "$HERE/bench_speculative_decoding.py" \
    --data_dir "$PROMPTS" \
    --spec_algorithm EAGLE \
    --model_path "$MODEL_PATH" \
    --eagle_path mit-han-lab/Qwen2.5-7B-Eagle-RL \
    --speculative_num_steps 8 \
    --speculative_eagle_topk 4 \
    --speculative_num_draft_tokens 48 \
    --tp_size $TP \
    --max_bs 1 \
    --attention_backend fa3 2>&1 | tee "$OUT/bench/${DATASET}_$(basename "$MODEL_PATH")_tp${TP}.log" | grep "Accept length"
