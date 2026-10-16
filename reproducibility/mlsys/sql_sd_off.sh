#!/bin/bash
# SkyRL-SQL, TLT (FastRL) with speculative decoding OFF (plain sglang engine, overlap scheduler on).
#   DATA_PATH=<SkyRL-SQL dir> OUT=<run dir> bash reproducibility/mlsys/sql_sd_off.sh
# Run from anywhere; see README.md. Output: $OUT/sdoff.log and $OUT/sdoff/steplog/; both arms can share one OUT. Extra arguments are Hydra overrides.
set -euo pipefail
: "${OUT:?set OUT to the run directory}"
mkdir -p "$OUT"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SD=false NAME=${NAME:-coder7b-sql-sdoff} FASTRL_STEPLOG_DIR=$OUT/sdoff/steplog \
    bash "$HERE/run_sql.sh" "$@" 2>&1 | tee "$OUT/sdoff.log"
exit ${PIPESTATUS[0]}
