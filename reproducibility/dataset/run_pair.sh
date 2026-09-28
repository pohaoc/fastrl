#!/bin/bash
# Run SD-on, then the controlled SD-off baseline, on one dataset; verify each run; then analyze.
# Usage: DATASET=eurus|dapo|sql STEPS=5 bash reproducibility/dataset/run_pair.sh
# Outputs: $OUT/<dataset>/{traces,logs,results} (OUT defaults to reproducibility/dataset/outputs).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO"
DATASET=${DATASET:?set DATASET=eurus, dapo or sql}
STEPS=${STEPS:-5}
export OUT=${OUT:-$HERE/outputs}
T=$OUT/$DATASET/traces
L=$OUT/$DATASET/logs
mkdir -p "$L"
say() { echo "[pair $(date +%H:%M:%S)] $*"; }

check_run() {  # <name> <log> <trace_dir>
    local n; n=$(cat "$3"/verl_*.jsonl 2>/dev/null | grep -c '"name": "step"')
    if grep -a -qE "Traceback|[A-Za-z]Error: " "$2" 2>/dev/null; then
        say "$1 FAILED: traceback in $2"; return 1
    fi
    if [ "$n" -lt "$STEPS" ]; then
        say "$1 FAILED: only $n/$STEPS steps traced"; return 1
    fi
    say "$1 OK: $n steps traced"
}

source "$REPO/.venv/bin/activate"
for SD in on off; do
    ray stop --force >/dev/null 2>&1
    rm -rf "$T/sd_$SD"
    say "launching SD-$SD on $DATASET, $STEPS steps"
    SD=$SD DATASET=$DATASET STEPS=$STEPS FASTRL_TRACE_DIR=$T/sd_$SD \
        bash "$HERE/run_grpo_7B_4gpu.sh" > "$L/sd_$SD.log" 2>&1
    check_run "SD-$SD" "$L/sd_$SD.log" "$T/sd_$SD" || exit 1
done
ray stop --force >/dev/null 2>&1

say "running analysis"
python "$HERE/analyze.py" --runs off=$T/sd_off on=$T/sd_on --out "$OUT/$DATASET/results" > "$L/analyze.log" 2>&1 \
    && python "$HERE/sd_cost.py" --runs off=$T/sd_off on=$T/sd_on --out "$OUT/$DATASET/results" --name "$DATASET" \
        >> "$L/analyze.log" 2>&1 \
    && python "$HERE/step_bars.py" --out "$OUT/$DATASET/results/step_bars.png" --dataset "$DATASET" $T/sd_on $T/sd_off \
        >> "$L/analyze.log" 2>&1 \
    && say "analysis done: $OUT/$DATASET/results" || { say "analysis FAILED, see $L/analyze.log"; exit 1; }
