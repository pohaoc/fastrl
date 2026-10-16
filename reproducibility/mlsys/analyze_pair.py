"""Compare the SD-on and SD-off arms of the SkyRL-SQL experiment.

Usage: python reproducibility/mlsys/analyze_pair.py <OUT> [summary.json]
<OUT> is the directory both arm scripts wrote to: <OUT>/{sdon,sdoff}.log and <OUT>/{sdon,sdoff}/steplog/.
Prints per-phase averages (verl's timers), rollout times from the engine step logs, speculation acceptance against
its break-even point, and time to a training score; optionally writes everything to summary.json.

From the engine step logs (sglang/srt/fastrl_steplog.py; one JSON line per scheduler iteration per GPU):
  Rollouts are split at gaps > 60 s in each GPU's log; rollout k on every GPU belongs to RL step k.
  rollout_s : first iteration on any GPU -> last iteration on any GPU
  step_s    : start of rollout k -> start of rollout k+1 (rollout + training); not defined for the last step
  accept    : speculative decode iterations, tokens per request per verify step = 1 + sum(accepted) / sum(batch size)
  breakeven : sum of speculative iteration times / sum of SD-off plain decode times at the same batch sizes
              (iteration time = time to the next iteration on the same GPU; SD-off reference = median per batch
              size over all its decode iterations). Speculation is faster than plain decoding when accept > breakeven.
"""

import glob
import json
import re
import statistics
import sys
from collections import defaultdict

base = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None


def load_rollouts(arm):
    per_gpu = []
    for f in sorted(glob.glob(f"{base}/{arm}/steplog/*.jsonl")):
        it = [r for r in (json.loads(line) for line in open(f)) if "mode" in r]
        if not it:
            continue
        rolls, cur = [], [it[0]]
        for prev, r in zip(it, it[1:]):
            if r["t0"] - prev["t1"] > 60:
                rolls.append(cur)
                cur = []
            cur.append(r)
        rolls.append(cur)
        per_gpu.append(rolls)
    if not per_gpu:
        sys.exit(f"no step logs under {base}/{arm}/steplog")
    return per_gpu, min(len(g) for g in per_gpu)


def intervals(roll):
    """(iteration, seconds to the next iteration) for decode iterations followed by another one < 5 s later."""
    res = []
    for r, nxt in zip(roll, roll[1:]):
        dt = nxt["t0"] - r["t0"]
        if r["mode"] == "DECODE" and 0 < dt < 5:
            res.append((r, dt))
    return res


on, n_on = load_rollouts("sdon")
off, n_off = load_rollouts("sdoff")

ref_lists = defaultdict(list)
for g in off:
    for roll in g:
        for r, dt in intervals(roll):
            ref_lists[r["bs"]].append(dt)
ref = {b: statistics.median(v) for b, v in ref_lists.items() if len(v) >= 5}


def ref_at(b):
    return ref[b] if b in ref else ref[min(ref, key=lambda k: abs(k - b))]


def per_step(per_gpu, n, sd):
    rows = []
    starts = [min(g[k][0]["t0"] for g in per_gpu) for k in range(n)]
    for k in range(n):
        end = max(g[k][-1]["t1"] for g in per_gpu)
        row = {"step": k + 1, "rollout_s": round(end - starts[k], 1),
               "step_s": round(starts[k + 1] - starts[k], 1) if k + 1 < n else None}
        if sd:
            num = den = 0
            t_sd = t_ref = 0.0
            for g in per_gpu:
                for r, dt in intervals(g[k]):
                    if r.get("sd") and r.get("acc") is not None:
                        num += r["acc"]
                        den += r["bs"]
                        t_sd += dt
                        t_ref += ref_at(r["bs"])
            row["accept"] = round(1 + num / den, 3) if den else None
            row["breakeven"] = round(t_sd / t_ref, 3) if t_ref else None
        rows.append(row)
    return rows


def logged(arm):
    """verl's per-step metrics from <OUT>/<arm>.log."""
    res = {}
    try:
        lines = open(f"{base}/{arm}.log", errors="ignore")
    except FileNotFoundError:
        return res
    for line in lines:
        if "timing_s/step" not in line:
            continue
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step:(\d+) ", line)
        if not m:
            continue
        kv = dict(re.findall(r"([\w/]+):(-?[\d.]+(?:e-?\d+)?)", line))
        res[int(m.group(1))] = {k: float(kv[k]) for k in
                                ("timing_s/gen", "timing_s/step", "critic/score/mean", "response_length/mean",
                                 "timing_s/update_actor", "timing_s/old_log_prob") if k in kv}
    return res


bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10**6)]
by_bs = []
for lo, hi in bins:
    num = den = 0
    t_sd = t_ref = 0.0
    for g in on:
        for roll in g:
            for r, dt in intervals(roll):
                if r.get("sd") and r.get("acc") is not None and lo <= r["bs"] <= hi:
                    num += r["acc"]
                    den += r["bs"]
                    t_sd += dt
                    t_ref += ref_at(r["bs"])
    if den:
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**6 else f">{lo - 1}")
        by_bs.append({"bs": label, "accept": round(1 + num / den, 3), "breakeven": round(t_sd / t_ref, 3),
                      "sd_time_s": round(t_sd, 1)})

res = {"sdon": per_step(on, n_on, True), "sdoff": per_step(off, n_off, False),
       "logged": {"sdon": logged("sdon"), "sdoff": logged("sdoff")}, "by_bs": by_bs,
       "ref_ms": {str(b): round(v * 1000, 2) for b, v in sorted(ref.items()) if b <= 64}}
if out:
    json.dump(res, open(out, "w"), indent=1)

print(f"rollouts in the step logs: SD on {n_on}, SD off {n_off}")
print("\nAverage per step (verl timers)          SD on      SD off")
for name, k in (("generation (incl. wake-up/sync)", "timing_s/gen"), ("log-probabilities", "timing_s/old_log_prob"),
                ("actor update", "timing_s/update_actor"), ("step", "timing_s/step")):
    a = [v[k] for v in res["logged"]["sdon"].values() if k in v]
    b = [v[k] for v in res["logged"]["sdoff"].values() if k in v]
    if a and b:
        print(f"  {name:36s} {sum(a) / len(a):8.1f} s {sum(b) / len(b):8.1f} s")
common = [r["step"] for r in res["sdon"] if r["step"] <= n_off]
ro = sum(res["sdon"][s - 1]["rollout_s"] for s in common)
rf = sum(res["sdoff"][s - 1]["rollout_s"] for s in common)
print(f"\nRollout time from engine logs, steps 1-{common[-1]}: SD on {ro:,.0f} s, SD off {rf:,.0f} s ({rf / ro:.2f}x)")
acc = sorted(r["accept"] for r in res["sdon"] if r["accept"])
be = sorted(r["breakeven"] for r in res["sdon"] if r["breakeven"])
if acc:
    print(f"Acceptance (tokens per request per verify step): median {acc[len(acc) // 2]:.2f}; "
          f"break-even median {be[len(be) // 2]:.2f}")
    for r in by_bs:
        print(f"  {r['bs']:>6s} running: acceptance {r['accept']:.2f}, break-even {r['breakeven']:.2f}")
print()
for arm in ("sdon", "sdoff"):
    L = res["logged"][arm]
    t, pts = 0.0, []
    for k in sorted(L):
        t += L[k]["timing_s/step"]
        pts.append((t / 3600, L[k]["critic/score/mean"], k))
    sm = [(p[0], sum(q[1] for q in pts[max(0, i - 2):i + 1]) / min(3, i + 1), p[2]) for i, p in enumerate(pts)]
    for thr in (0.5, 0.65):
        hit = next((x for x in sm if x[1] >= thr), None)
        print(f"{arm}: training score {thr} (3-step mean) " + (f"at step {hit[2]}, {hit[0]:.2f} h" if hit else "not reached"))
