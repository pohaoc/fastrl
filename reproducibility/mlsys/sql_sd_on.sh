#!/bin/bash
# SkyRL-SQL, TLT (FastRL) with speculative decoding ON (adaptive EAGLE, offline-tuned drafter).
#   DATA_PATH=<SkyRL-SQL dir> SPEC_MODEL_PATH=<drafter dir> OUT=<run dir> bash reproducibility/mlsys/sql_sd_on.sh
# Run from anywhere; see README.md. Output: $OUT/sdon.log and $OUT/sdon/steplog/; both arms can share one OUT. Extra arguments are Hydra overrides.
set -euo pipefail
: "${OUT:?set OUT to the run directory}"
mkdir -p "$OUT"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SD=true NAME=${NAME:-coder7b-sql-sdon} FASTRL_STEPLOG_DIR=$OUT/sdon/steplog \
    bash "$HERE/run_sql.sh" "$@" 2>&1 | tee "$OUT/sdon.log"
exit ${PIPESTATUS[0]}
