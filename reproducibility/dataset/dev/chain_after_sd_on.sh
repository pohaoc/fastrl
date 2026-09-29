#!/bin/bash
# Wait for the SD-on run to finish, verify it, then run the SD-off baseline and the analysis.
# Usage: bash experiments/sd_breakdown/chain_after_sd_on.sh <sd_on_trainer_pid> [steps]
set -uo pipefail
cd "$(dirname "$0")/../.."
PID=${1:?trainer pid}
STEPS=${2:-5}
E=experiments/sd_breakdown
say() { echo "[chain $(date +%H:%M:%S)] $*"; }

count_steps() { cat "$1"/verl_*.jsonl 2>/dev/null | grep -c '"name": "step"'; }
check_run() {  # <name> <log> <trace_dir>
    local n; n=$(count_steps "$3")
    if grep -a -qE "Traceback|[A-Za-z]Error: " "$2" 2>/dev/null; then
        say "$1 FAILED: traceback in $2"; return 1
    fi
    if [ "$n" -lt "$STEPS" ]; then
        say "$1 FAILED: only $n/$STEPS steps traced"; return 1
    fi
    say "$1 OK: $n steps traced"
}

say "waiting for SD-on trainer (pid $PID)"
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
check_run SD-on $E/logs/sd_on.log $E/traces/sd_on || exit 1

source .venv/bin/activate
ray stop --force >/dev/null 2>&1
rm -rf $E/traces/sd_off
say "launching SD-off (controlled baseline), $STEPS steps"
SD=off STEPS=$STEPS bash $E/run_grpo_7B_4gpu.sh > $E/logs/sd_off.log 2>&1
check_run SD-off $E/logs/sd_off.log $E/traces/sd_off || exit 1

say "running analysis"
python $E/analyze.py --runs off=$E/traces/sd_off on=$E/traces/sd_on --out $E/results > $E/logs/analyze.log 2>&1 \
    && say "analysis done: $E/results" || { say "analysis FAILED, see $E/logs/analyze.log"; exit 1; }
