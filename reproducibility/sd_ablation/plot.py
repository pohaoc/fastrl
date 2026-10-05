"""Figure explaining the SD-on rollout slowdown on SkyRL-SQL, from an sd_ablation run.

Usage: python plot.py <run dir>   ->  <run dir>/sd_slowdown.png and sd_slowdown.csv (the numbers behind it)

A  bridge from SD-off to SD-on rollout time (replicate means): which engine change adds what.
B  cost of one decode iteration by running batch size, for the configurations in the bridge.
C  acceptance per speculative step vs the break-even it must reach.
Rollout time = bulk (prefill + decode while > 32 requests run) + tail (decode at <= 32, where SD may speculate).
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report import BUCKETS, analyse, bucket, load  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots 1-5, validated together; ink and chrome tokens.
C_OFF, C_TAX, C_GATED, C_SD, C_RELEASED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def phases(a):
    t = a["time"]
    return t["prefill"] + t["decode>32"] + a["idle"], t["decode<=32"]


def pooled_ms(arms_a, labels):
    out = []
    for lab in labels:
        steps = sum(a["buckets"][lab]["steps"] for a in arms_a if lab in a["buckets"])
        time = sum(a["buckets"][lab]["time"] for a in arms_a if lab in a["buckets"])
        out.append(1000 * time / steps if steps >= 20 else None)   # skip buckets with < 20 iterations
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)


def main(run_dir):
    A = {}
    for name in ("off", "off_rep", "off_tax", "sd_gated", "sd_tuned", "sd_tuned_rep", "sd_released"):
        r, recs = load(os.path.join(run_dir, name))
        A[name] = analyse(r, recs)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    bulk = {k: phases(a)[0] for k, a in A.items()}
    tail = {k: phases(a)[1] for k, a in A.items()}
    off_bulk, off_tail = mean([bulk["off"], bulk["off_rep"]]), mean([tail["off"], tail["off_rep"]])
    sd_bulk, sd_tail = mean([bulk["sd_tuned"], bulk["sd_tuned_rep"]]), mean([tail["sd_tuned"], tail["sd_tuned_rep"]])
    steps = [  # label, delta, color, kind
        ("SD off", off_bulk + off_tail, C_OFF, "total"),
        ("+ engine settings verl forces with SD\n(overlap scheduler off)", bulk["off_tax"] - off_bulk, C_TAX, "delta"),
        ("+ EAGLE engine, not speculating\n(drafter runs on every prefill)", bulk["sd_gated"] - bulk["off_tax"], C_GATED,
         "delta"),
        ("+ other bulk-phase difference (noise)", sd_bulk - bulk["sd_gated"], MUTED, "delta"),
        (f"+ speculation in the tail (<= 32 requests)\n"
         f"mean of 2 runs; per run: off {tail['off']:.0f} / {tail['off_rep']:.0f} s, on {tail['sd_tuned']:.0f} / "
         f"{tail['sd_tuned_rep']:.0f} s", sd_tail - off_tail, C_SD, "delta"),
        ("SD on (tuned drafter)", sd_bulk + sd_tail, C_SD, "total"),
    ]

    fig = plt.figure(figsize=(13, 10.2), facecolor=SURFACE)
    gs_a = fig.add_gridspec(1, 1, left=0.3, right=0.95, top=0.85, bottom=0.535)
    gs_b = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.42, bottom=0.085, wspace=0.42)
    fig.text(0.07, 0.955, "Why SD-on rollouts are slower on SkyRL-SQL (Qwen2.5-Coder-7B-Instruct, 4x L40S, fa3)",
             fontsize=15, color=INK, weight="bold")
    fig.text(0.07, 0.925,
             f"The SD-capable engine adds ~{steps[1][1] + steps[2][1]:.0f} s to the high-concurrency bulk phase, where it never "
             f"speculates; speculation only runs in the short tail and cannot repay it.",
             fontsize=11, color=INK2)
    fig.text(0.07, 0.902,
             "One replayed RL rollout step per configuration (256 prompts x 5 trajectories, <= 5 turns), all on one node. "
             "Prefix-cache reset ruled out: later-turn cache hits 96.7% (SD) vs 97.5% (off).",
             fontsize=9, color=MUTED)

    # ---- A: bridge ------------------------------------------------------------------------------------------
    ax = fig.add_subplot(gs_a[0, 0])
    style(ax)
    running, y = 0.0, list(range(len(steps)))[::-1]
    for yi, (label, v, color, kind) in zip(y, steps):
        if kind == "total":
            left, width, running = 0.0, v, v
            text = f"{v:.0f} s"
        else:
            left, width = (running, v) if v >= 0 else (running + v, -v)
            running += v
            text = f"{v:+.0f} s"
        ax.barh(yi, width, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(left + width + 2, yi, text, va="center", ha="left", fontsize=10, color=INK, weight="bold")
    ax.set_yticks(y, [s[0] for s in steps], fontsize=9.5, color=INK2)
    ax.set_xlim(0, max(s[1] for s in steps if s[3] == "total") * 1.12)
    ax.set_xlabel("rollout time for one RL step (s)", color=INK2, fontsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("A   Where the extra time goes", loc="left", fontsize=12, color=INK, weight="bold")

    # ---- B: per-iteration cost by batch size ------------------------------------------------------------------
    ax = fig.add_subplot(gs_b[0, 0])
    style(ax)
    labels = [bucket(lo) for lo, _ in BUCKETS]
    ticks = ["1", "2-4", "5-20", "21-32", "33-128", "129-256", "257-512", ">512"]
    x = list(range(len(labels)))
    ax.axvspan(-0.5, 3.5, color="#f0efec", zorder=0)
    ax.text(1.5, 0.97, "SD may speculate here", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=8.5, color=MUTED)
    series = [
        ("SD off", [A["off"], A["off_rep"]], C_OFF),
        ("SD-on engine settings, no drafter", [A["off_tax"]], C_TAX),
        ("EAGLE engine, never speculating", [A["sd_gated"]], C_GATED),
        ("SD on (tuned drafter)", [A["sd_tuned"], A["sd_tuned_rep"]], C_SD),
    ]
    ms = {}
    for name, arms_a, color in series:
        vals = pooled_ms(arms_a, labels)
        ms[name] = vals
        pts = [(xi, v) for xi, v in zip(x, vals) if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=name)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8.5, labelcolor=INK2)
    v = ms["SD-on engine settings, no drafter"][6]
    ax.text(6.15, v, "3 SD-on configs:\nsame cost,\n~1.5x SD off", ha="left", va="center", fontsize=8, color=INK2)
    v = ms["SD off"][7]
    ax.text(7, v - 4, "SD off\n(only config that runs\n> 512 at once)", ha="center", va="top", fontsize=7.5, color=INK2)
    v = ms["SD on (tuned drafter)"][1]
    ax.text(1, v + 3, "speculative steps\n~3x a plain step", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, ticks, fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, 7.6)
    ax.set_ylim(0, max(v for vals in ms.values() for v in vals if v) * 1.35)
    ax.set_xlabel("requests decoding at once (running batch size)", color=INK2, fontsize=10)
    ax.set_ylabel("time per decode iteration (ms)", color=INK2, fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("B   Cost of one decode iteration", loc="left", fontsize=12, color=INK, weight="bold")

    # ---- C: acceptance vs break-even --------------------------------------------------------------------------
    ax = fig.add_subplot(gs_b[0, 1])
    style(ax)
    small = ["1-1", "2-4"]
    plain = [A["off"], A["off_rep"]]
    sd_steps = sum(a["buckets"][b]["steps"] for a in (A["sd_tuned"], A["sd_tuned_rep"], A["sd_released"]) for b in small)
    sd_time = sum(a["buckets"][b]["time"] for a in (A["sd_tuned"], A["sd_tuned_rep"], A["sd_released"]) for b in small)
    pl_steps = sum(a["buckets"][b]["steps"] for a in plain for b in small)
    pl_time = sum(a["buckets"][b]["time"] for a in plain for b in small)
    breakeven = (sd_time / sd_steps) / (pl_time / pl_steps)
    bars = [("released drafter\n(mit-han-lab Eagle-RL)", A["sd_released"]["accept"], C_RELEASED),
            ("tuned drafter, run 1", A["sd_tuned"]["accept"], C_SD),
            ("tuned drafter, run 2", A["sd_tuned_rep"]["accept"], C_SD)]
    yb = list(range(len(bars)))[::-1]
    for yi, (n, v, c) in zip(yb, bars):
        ax.barh(yi, v, height=0.55, color=c, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.08, yi, f"{v:.2f}", va="center", fontsize=10, color=INK, weight="bold")
    ax.axvline(breakeven, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(breakeven + 0.08, 2.55, f"break-even {breakeven:.1f}\n= cost of a speculative step\n/ cost of a plain step "
            f"({1000 * sd_time / sd_steps:.0f} vs {1000 * pl_time / pl_steps:.0f} ms, 1-4 requests)",
            fontsize=8, color=INK2, va="top")
    ax.set_yticks(yb, [b[0] for b in bars], fontsize=9, color=INK2)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlim(0, 9)
    ax.set_xlabel("tokens produced per speculative step per request (accept length; max 9)", color=INK2, fontsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("C   Can speculation in the tail pay it back?", loc="left", fontsize=12, color=INK, weight="bold")

    out = os.path.join(run_dir, "sd_slowdown.png")
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    with open(os.path.join(run_dir, "sd_slowdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "item", "value", "unit"])
        for label, v, _, kind in steps:
            w.writerow(["A", label.replace("\n", " "), round(v, 1), "s" if kind == "total" else "s (delta)"])
        for k in A:
            w.writerow(["A-detail", f"{k} bulk / tail", f"{bulk[k]:.1f} / {tail[k]:.1f}", "s"])
        for name, vals in ms.items():
            for lab, v in zip(ticks, vals):
                w.writerow(["B", f"{name} @ {lab}", "" if v is None else round(v, 1), "ms per iteration"])
        for n, v, _ in bars:
            w.writerow(["C", n.replace("\n", " "), round(v, 2), "accept length"])
        w.writerow(["C", "break-even", round(breakeven, 2), "accept length"])
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
