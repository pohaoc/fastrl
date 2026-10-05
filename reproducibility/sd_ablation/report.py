"""Summarise an sd_ablation run: python report.py <run dir> [--ref off]

<run dir>/<arm>/result.json + <arm>/steplog/steps_*.jsonl (one line per scheduler iteration of TP rank 0).
Per-iteration cost = time to the next iteration's start. Gaps > 2 s are the engine waiting on the SQL
environment (every live trajectory between turns): the iteration is charged the median cost of its kind and
the rest is reported as idle.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import median

BUCKETS = [(1, 1), (2, 4), (5, 20), (21, 32), (33, 128), (129, 256), (257, 512), (513, 10**9)]
IDLE_GAP = 2.0


def bucket(bs):
    for lo, hi in BUCKETS:
        if lo <= bs <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}"
    return "?"


def load(arm_dir):
    r = json.load(open(os.path.join(arm_dir, "result.json")))
    recs = [json.loads(l) for f in glob.glob(os.path.join(arm_dir, "steplog", "steps_*.jsonl")) for l in open(f)]
    if r.get("phase") == "done":
        recs = [x for x in recs if r["rollout_t0"] <= x["t0"] <= r["rollout_t1"]]
    return r, sorted(recs, key=lambda x: x["t0"])


def analyse(r, recs):
    steps = [x for x in recs if "mode" in x]
    events = [x for x in recs if "event" in x]
    end = r.get("rollout_t1", steps[-1]["t1"] if steps else 0)
    for a, b in zip(steps, steps[1:] + [None]):
        a["dt"] = (b["t0"] if b else end) - a["t0"]
        a["kind"] = "prefill" if a["mode"] != "DECODE" else ("decode<=32" if a["bs"] <= 32 else "decode>32")
    med = {}
    for k in ("prefill", "decode<=32", "decode>32"):
        d = [s["dt"] for s in steps if s["kind"] == k and s["dt"] <= IDLE_GAP]
        med[k] = median(d) if d else 0.0
    idle = 0.0
    for s in steps:
        if s["dt"] > IDLE_GAP:
            idle += s["dt"] - med[s["kind"]]
            s["dt"] = med[s["kind"]]
    out = {"time": defaultdict(float), "buckets": defaultdict(lambda: defaultdict(float)), "idle": idle}
    for s in steps:
        out["time"][s["kind"]] += s["dt"]
        if s["mode"] != "DECODE":
            out["time"]["prefill_tok"] += s.get("prefill_tok", 0)
            continue
        b = out["buckets"][bucket(s["bs"])]
        sd = bool(s.get("sd"))
        acc = s.get("acc") or 0
        b["steps"] += 1
        b["time"] += s["dt"]
        b["tokens"] += s["bs"] + (acc if sd else 0)
        b["sd_steps"] += sd
        b["cg_steps"] += bool(s.get("cg"))
        if sd:
            b["sd_bs"] += s["bs"]
            b["sd_tokens"] += s["bs"] + acc
    out["transitions"] = [e for e in events if e["event"] == "sd_transition"]
    sd_bs = sum(b["sd_bs"] for b in out["buckets"].values())
    out["accept"] = sum(b["sd_tokens"] for b in out["buckets"].values()) / sd_bs if sd_bs else None
    out["sd_time"] = sum(s["dt"] for s in steps if s["mode"] == "DECODE" and s.get("sd"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default="off")
    args = ap.parse_args()
    arms = {}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "*", "result.json"))):
        name = os.path.basename(os.path.dirname(d))
        r, recs = load(os.path.dirname(d))
        arms[name] = (r, analyse(r, recs) if recs else None)
    ref = arms.get(args.ref, (None,))[0]
    ref_s = ref["rollout_s"] if ref and ref.get("phase") == "done" else None

    print(f"\n== Rollout of one RL step ({next(iter(arms.values()))[0]['args']['prompts']} prompts x "
          f"{next(iter(arms.values()))[0]['args']['n']}); times in s; ref = {args.ref}")
    print(f"{'arm':16} {'rollout':>8} {'vs ref':>7} {'tok/s':>7} {'prefill':>8} {'dec<=32':>8} {'dec>32':>8} "
          f"{'idle':>6} {'SD time':>8} {'accept':>6} {'trans':>5} {'prefill Mtok':>12} {'KV pool':>9} {'turns':>5} {'reward':>6}")
    for name, (r, a) in arms.items():
        if r.get("phase") != "done" or a is None:
            print(f"{name:16} FAILED during {r.get('phase')} (see the job log)")
            continue
        t = a["time"]
        acc = f"{a['accept']:.2f}" if a["accept"] else "-"
        print(f"{name:16} {r['rollout_s']:8.1f} {(r['rollout_s'] / ref_s if ref_s else float('nan')):7.3f} "
              f"{r['generated_tokens'] / r['rollout_s']:7.0f} {t['prefill']:8.1f} {t['decode<=32']:8.1f} "
              f"{t['decode>32']:8.1f} {a['idle']:6.1f} {a['sd_time']:8.1f} "
              f"{acc:>6} {len(a['transitions']):5d} "
              f"{t['prefill_tok'] / 1e6:12.2f} {str((r.get('engine') or {}).get('max_total_num_tokens')):>9} "
              f"{r['turns_mean']:5.2f} {r['reward_mean']:6.2f}")

    print("\n== Decode cost by running batch size (ms per iteration | tokens/s | % iterations speculative | "
          "accept length on speculative iterations | % CUDA graph)")
    labels = [bucket(lo) for lo, _ in BUCKETS]
    for name, (r, a) in arms.items():
        if a is None:
            continue
        print(f"\n{name}")
        print(f"  {'bs':>8} {'iters':>7} {'time s':>7} {'ms/it':>7} {'tok/s':>8} {'%SD':>5} {'accept':>6} {'%CG':>5}")
        for lab in labels:
            b = a["buckets"].get(lab)
            if not b or not b["steps"]:
                continue
            acc = f"{b['sd_tokens'] / b['sd_bs']:.2f}" if b["sd_bs"] else "-"
            print(f"  {lab:>8} {int(b['steps']):7d} {b['time']:7.1f} {1000 * b['time'] / b['steps']:7.1f} "
                  f"{b['tokens'] / b['time'] if b['time'] else 0:8.0f} {100 * b['sd_steps'] / b['steps']:5.0f} "
                  f"{acc:>6} {100 * b['cg_steps'] / b['steps']:5.0f}")
        if a["transitions"]:
            req = [e["requeued"] for e in a["transitions"]]
            print(f"  SD transitions: {len(req)} (prefix cache dropped each time); requeued per transition: "
                  f"min {min(req)} / median {median(req)} / max {max(req)}")


if __name__ == "__main__":
    main()
