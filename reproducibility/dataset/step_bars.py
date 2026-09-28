"""Per-step wall time, speculative vs baseline, split into rollout (colored by run) and training (gray).

Uses the trainer's own phase timers (driver process):
  rollout  = gen + reward: generation (including weight sync into the engine) and reward scoring.
             For SkyRL-SQL the env computes rewards inside generation, so this keeps datasets comparable.
  training = old_log_prob + ref + update_actor
  other    = rest of the step (advantages, metrics, bookkeeping)

Usage: python step_bars.py --out step_bars.png \
           --dataset Eurus traces/sd_on traces/sd_off --dataset DAPO-Math traces/dapo/sd_on traces/dapo/sd_off
"""

import argparse
import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Rollout is colored by run; training (about equal in both) is gray. "other" (advantages,
# bookkeeping) is ~0.5 s per step: kept in the totals, not drawn.
RUNS = [("Speculative", "#33a02c"), ("Baseline", "#1f78b4")]
TRAINING_COLOR = "#b0b0b0"


def driver_steps(trace_dir):
    recs = []
    for path in glob.glob(f"{trace_dir}/verl_*.jsonl"):
        with open(path) as f:
            recs += [json.loads(line) for line in f if line.strip()]
    steps = sorted((r for r in recs if r["name"] == "step"), key=lambda r: r["start"])
    drv = [r for r in recs if r["pid"] == steps[0]["pid"]]
    out = []
    for s in steps:
        ph = {}
        for r in drv:
            if s["start"] <= r["start"] <= s["end"] and r["name"] != "step":
                ph[r["name"]] = ph.get(r["name"], 0.0) + r["end"] - r["start"]
        total = s["end"] - s["start"]
        rollout = ph.get("gen", 0.0) + ph.get("reward", 0.0)
        training = ph.get("old_log_prob", 0.0) + ph.get("ref", 0.0) + ph.get("update_actor", 0.0)
        out.append({"total": total, "rollout": rollout, "generation": ph.get("gen", 0.0),
                    "training": training, "other": total - rollout - training})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs=3, action="append", metavar=("NAME", "SD_ON_DIR", "SD_OFF_DIR"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = len(args.dataset)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.9 * n), squeeze=False)
    summary = {}
    for ax, (name, on_dir, off_dir) in zip(axes[:, 0], args.dataset):
        on, off = driver_steps(on_dir), driver_steps(off_dir)
        k = min(len(on), len(off))
        w, gap = 0.36, 0.04
        for i in range(k):
            for j, ((label, color), rec) in enumerate(zip(RUNS, (on[i], off[i]))):
                x = i + (j - 0.5) * (w + gap)
                ax.bar(x, rec["rollout"], w, color=color, edgecolor="white", linewidth=0.8)
                ax.bar(x, rec["training"], w, bottom=rec["rollout"], color=TRAINING_COLOR, edgecolor="white",
                       linewidth=0.8)
                if j == 0:
                    top = rec["rollout"] + rec["training"]
                    ax.text(x, top + 4, f"{off[i]['total'] / on[i]['total']:.2f}x", ha="center", va="bottom",
                            fontsize=9, fontweight="bold")
        ax.set_xticks(range(k), [f"step {i + 1}" for i in range(k)])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("seconds per step")
        tot_on, tot_off = sum(r["total"] for r in on[:k]), sum(r["total"] for r in off[:k])
        ax.set_title(f"{name}: {tot_off / tot_on:.2f}x overall speedup", loc="left", fontsize=12)
        ax.set_ylim(0, max(max(r["total"] for r in on[:k]), max(r["total"] for r in off[:k])) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        summary[name] = {"speculative": on[:k], "baseline": off[:k]}
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c) for _, c in RUNS]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=TRAINING_COLOR))
    labels = [l for l, _ in RUNS] + ["Training"]
    axes[0, 0].legend(handles, labels, loc="upper right", fontsize=9, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    json.dump(summary, open(args.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
