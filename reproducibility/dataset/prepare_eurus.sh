#!/bin/bash
# Download Eurus-2-RL-Data (PRIME-RL) into <repo>/Eurus-2-RL-Data. Already in verl format:
# 480,537 train / 2,048 validation rows, data_source in {numina_*, codecontests, apps, taco, codeforces},
# scored by verl/utils/reward_score (prime_math / prime_code).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("PRIME-RL/Eurus-2-RL-Data", repo_type="dataset", local_dir="Eurus-2-RL-Data")
import pandas as pd
for s in ("train", "validation"):
    print(s, len(pd.read_parquet(f"Eurus-2-RL-Data/{s}.parquet")), "rows")
EOF
