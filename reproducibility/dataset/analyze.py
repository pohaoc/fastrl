"""Analyze FastRL timeline traces: per-GPU Gantt chart and speedup attribution.

Usage (run_pair.sh calls this for you):
    python reproducibility/dataset/analyze.py \
        --runs off=<outputs>/<dataset>/traces/sd_off on=<outputs>/<dataset>/traces/sd_on \
        --out <outputs>/<dataset>/results
"""

import argparse
import bisect
import collections
import glob
import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Display order, label and color of each Gantt category.
CATEGORIES = {
    "weight_sync": ("Weight sync / engine wake", "#9e9e9e"),
    "prefill": ("Prefill (target)", "#6a3d9a"),
    "reprefill": ("Re-prefill on SD switch", "#e31a1c"),
    "decode": ("Decode (no SD)", "#1f78b4"),
    "draft": ("SD: draft", "#ff7f00"),
    "verify": ("SD: verify", "#33a02c"),
    "draft_extend": ("SD: draft extend", "#fdbf6f"),
    "old_log_prob": ("Old log-prob", "#a6cee3"),
    "ref_log_prob": ("Ref log-prob", "#b2df8a"),
    "update_actor": ("Actor update", "#fb9a99"),
}
TRAIN_SPANS = {"old_log_prob", "ref_log_prob", "update_actor"}
ROLLOUT_CATS = ["prefill", "reprefill", "decode", "draft", "verify", "draft_extend"]


def gpu_index_by_uuid():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout
    mapping = {}
    for line in out.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        mapping[uuid.removeprefix("GPU-")] = int(idx)
    return mapping


def load_run(trace_dir):
    recs = []
    for path in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        with open(path) as f:
            recs.extend(json.loads(line) for line in f if line.strip())
    return recs


class Run:
    def __init__(self, name, trace_dir, uuid_to_idx):
        self.name = name
        recs = load_run(trace_dir)
        if not recs:
            raise SystemExit(f"no traces in {trace_dir}")
        # Driver: the process that emits the "step" timer.
        steps = sorted((r for r in recs if r["src"] == "verl" and r["name"] == "step"), key=lambda r: r["start"])
        self.driver_pid = steps[0]["pid"]
        self.steps = steps
        self.driver = [r for r in recs if r["src"] == "verl" and r["pid"] == self.driver_pid]
        self.t0 = steps[0]["start"]

        # Per-GPU verl spans (worker methods synchronize the device at boundaries).
        self.gpu_verl = collections.defaultdict(list)
        for r in recs:
            if r["src"] == "verl" and r.get("gpu_uuid"):
                self.gpu_verl[uuid_to_idx[r["gpu_uuid"]]].append(r)
        # Per-GPU SGLang spans (CUDA-event timed).
        self.gpu_sgl = collections.defaultdict(list)
        for r in recs:
            if r["src"] == "sglang":
                self.gpu_sgl[uuid_to_idx[r["gpu_uuid"]]].append(r)
        for d in (self.gpu_verl, self.gpu_sgl):
            for v in d.values():
                v.sort(key=lambda r: r["start"])
        self.gpus = sorted(set(self.gpu_verl) | set(self.gpu_sgl))
        self.spec = any(r["name"] == "verify" for v in self.gpu_sgl.values() for r in v)

    def step_of(self, t):
        starts = [s["start"] for s in self.steps]
        i = bisect.bisect_right(starts, t) - 1
        if i < 0 or t > self.steps[i]["end"]:
            return None
        return i + 1

    def categorized(self, gpu):
        """Yield (category, start, end, record) for leaf spans on this GPU."""
        for r in self.gpu_verl.get(gpu, []):
            if r["name"] in TRAIN_SPANS:
                yield r["name"], r["start"], r["end"], r
        # Weight sync = generate_sequences minus the inner rollout span.
        gens = [r for r in self.gpu_verl.get(gpu, []) if r["name"] == "generate_sequences"]
        rolls = [r for r in self.gpu_verl.get(gpu, []) if r["name"] == "rollout"]
        for g in gens:
            inner = [r for r in rolls if g["start"] <= r["start"] and r["end"] <= g["end"]]
            if inner:
                yield "weight_sync", g["start"], inner[0]["start"], g
                yield "weight_sync", inner[-1]["end"], g["end"], g
        sgl = self.gpu_sgl.get(gpu, [])
        decoded = False  # within a rollout, a prefill after decoding began is the SD-switch re-prefill
        last_end = None
        for r in sgl:
            if last_end is not None and r["start"] - last_end > 5.0:
                decoded = False  # new rollout
            n = r["name"]
            if self.spec:
                if n == "forward":
                    last_end = r["end"]
                    continue
                if n in ("target_extend", "draft_extend"):
                    cat = "reprefill" if decoded else "prefill"
                    yield cat, r["start"], r["end"], r
                elif n == "decode_nosd":
                    decoded = True
                    yield "decode", r["start"], r["end"], r
                elif n == "draft":
                    decoded = True
                    yield "draft", r["start"], r["end"], r
                elif n == "verify":
                    yield "verify", r["start"], r["end"], r
                elif n == "draft_extend_after_decode":
                    yield "draft_extend", r["start"], r["end"], r
            elif n == "forward":
                cat = "decode" if r["mode"] == "DECODE" else "prefill"
                yield cat, r["start"], r["end"], r
            last_end = r["end"]

    def forwards(self, gpu):
        return [r for r in self.gpu_sgl.get(gpu, []) if r["name"] == "forward"]


def merge_intervals(items, gap=2e-3):
    """Merge consecutive same-category intervals separated by < gap seconds (for drawing)."""
    out = []
    for cat, s, e in sorted(items, key=lambda x: x[1]):
        if out and out[-1][0] == cat and s - out[-1][2] < gap:
            out[-1][2] = max(out[-1][2], e)
        else:
            out.append([cat, s, e])
    return out


def plot_gantt(runs, out_path, step_filter=None):
    n = len(runs)
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(18, 3.2 * n + 1.6 * n), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2] * n},
    )
    xmax = 0
    for k, run in enumerate(runs):
        ax, axb = axes[2 * k], axes[2 * k + 1]
        steps = [s for i, s in enumerate(run.steps, 1) if step_filter is None or i in step_filter]
        lo, hi = steps[0]["start"], steps[-1]["end"]
        t0 = lo
        for row, gpu in enumerate(run.gpus):
            items = [(c, s, e) for c, s, e, _ in run.categorized(gpu) if e > lo and s < hi]
            by_cat = collections.defaultdict(list)
            for c, s, e in merge_intervals(items):
                by_cat[c].append((s - t0, e - s))
            for c, bars in by_cat.items():
                ax.broken_barh(bars, (row - 0.4, 0.8), facecolors=CATEGORIES[c][1], linewidth=0)
        for i, s in enumerate(run.steps, 1):
            if s in steps:
                ax.axvline(s["start"] - t0, color="k", lw=0.6, ls="--")
                ax.text(s["start"] - t0, len(run.gpus) - 0.45, f" step {i}", fontsize=8, va="bottom")
        ax.set_yticks(range(len(run.gpus)), [f"GPU {g}" for g in run.gpus])
        ax.set_ylim(-0.6, len(run.gpus) + 0.1)
        ax.invert_yaxis()
        ax.set_title(f"{run.name}: total {hi - lo:.0f}s for {len(steps)} step(s)", loc="left", fontsize=11)
        ax.set_facecolor("#f4f4f4")  # background = idle / host-side time
        # Running batch size (GPU 0 engine view; all TP ranks run in lockstep).
        fw = [r for r in run.forwards(run.gpus[0]) if r["end"] > lo and r["start"] < hi]
        axb.step([r["start"] - t0 for r in fw], [r["bs"] for r in fw], where="post", lw=0.8, color="#333")
        if run.spec:
            axb.axhline(32, color="#ff7f00", lw=0.8, ls=":", label="SD batch-size threshold (32)")
            axb.legend(loc="upper right", fontsize=8)
        axb.set_ylabel("running\nbatch size", fontsize=8)
        axb.set_yscale("symlog", linthresh=8)
        axb.set_ylim(bottom=0)
        xmax = max(xmax, hi - lo)
    axes[-1].set_xlim(0, xmax * 1.01)
    axes[-1].set_xlabel("seconds since first plotted step start")
    handles = [Patch(color=c, label=l) for l, c in CATEGORIES.values()]
    handles.append(Patch(facecolor="#f4f4f4", edgecolor="#bbb", label="Idle / host-side (reward, adv, scheduling)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def step_breakdown(run):
    """Per-step seconds per category on GPU 0, plus driver phases and token counts."""
    gpu = run.gpus[0]
    rows = []
    for i, s in enumerate(run.steps, 1):
        row = {"step": i, "step_s": s["end"] - s["start"]}
        for d in run.driver:
            if d["name"] in ("gen", "reward", "old_log_prob", "ref", "adv", "update_actor") and s["start"] <= d["start"] <= s["end"]:
                row[f"drv_{d['name']}_s"] = d["end"] - d["start"]
        for c in CATEGORIES:
            row[c] = 0.0
        for c, a, b, _ in run.categorized(gpu):
            if s["start"] <= a <= s["end"]:
                row[c] += b - a
        fw = [r for r in run.forwards(gpu) if s["start"] <= r["start"] <= s["end"]]
        row["rollout_busy_s"] = sum(row[c] for c in ROLLOUT_CATS)
        if fw:
            row["rollout_span_s"] = fw[-1]["end"] - fw[0]["start"]
            row["rollout_idle_s"] = row["rollout_span_s"] - sum(r["end"] - r["start"] for r in fw)
        dec = [r for r in fw if r["mode"] == "DECODE"]
        row["decode_tokens"] = sum(r["bs"] + r.get("accepted", 0) for r in dec)
        sd = [r for r in dec if run.spec and _is_sd_step(run, gpu, r)]
        row["sd_steps"] = len(sd)
        row["sd_tokens"] = sum(r["bs"] + r.get("accepted", 0) for r in sd)
        row["sd_mean_accept_len"] = (row["sd_tokens"] / sum(r["bs"] for r in sd)) if sd else None
        tail = [r for r in dec if r["bs"] <= 32]
        row["tail_bs<=32_s"] = sum(r["end"] - r["start"] for r in tail)
        row["tail_bs<=32_tokens"] = sum(r["bs"] + r.get("accepted", 0) for r in tail)
        rows.append(row)
    return rows


_sd_cache = {}


def _is_sd_step(run, gpu, fwd):
    """True if this scheduler forward contained a verify span (i.e. ran speculatively)."""
    key = (id(run), gpu)
    if key not in _sd_cache:
        _sd_cache[key] = [r["start"] for r in run.gpu_sgl[gpu] if r["name"] == "verify"]
    vs = _sd_cache[key]
    i = bisect.bisect_left(vs, fwd["start"])
    return i < len(vs) and vs[i] <= fwd["end"]


def baseline_latency_model(base_run):
    """Median non-SD decode step latency per batch size, from the SD-off run."""
    by_bs = collections.defaultdict(list)
    for r in base_run.forwards(base_run.gpus[0]):
        if r["mode"] == "DECODE":
            by_bs[r["bs"]].append(r["end"] - r["start"])
    med = {bs: sorted(v)[len(v) // 2] for bs, v in by_bs.items()}
    keys = sorted(med)

    def lat(bs):
        if bs in med:
            return med[bs]
        i = bisect.bisect_left(keys, bs)
        lo = keys[max(i - 1, 0)]
        hi = keys[min(i, len(keys) - 1)]
        if lo == hi:
            return med[lo]
        w = (bs - lo) / (hi - lo)
        return med[lo] * (1 - w) + med[hi] * w

    return lat, med


def sd_cost_effectiveness(sd_run, lat):
    """For each SD decode step, compare its cost to plain decoding of the same tokens."""
    gpu = sd_run.gpus[0]
    sgl = sd_run.gpu_sgl[gpu]
    buckets = collections.OrderedDict((b, collections.Counter()) for b in ["1", "2-4", "5-8", "9-16", "17-32"])

    def bucket(bs):
        return "1" if bs == 1 else "2-4" if bs <= 4 else "5-8" if bs <= 8 else "9-16" if bs <= 16 else "17-32"

    # Group draft / verify / draft_extend spans into one SD step each (they occur in that order).
    i = 0
    while i < len(sgl):
        r = sgl[i]
        if r["name"] != "draft":
            i += 1
            continue
        grp = {"draft": r}
        j = i + 1
        while j < len(sgl) and sgl[j]["name"] in ("forward", "verify", "draft_extend_after_decode") and len(grp) < 3:
            if sgl[j]["name"] != "forward":
                grp[sgl[j]["name"]] = sgl[j]
            j += 1
        i = j
        if "verify" not in grp:
            continue
        bs = r["bs"]
        tokens = bs + grp["verify"]["accepted"]
        c = buckets[bucket(bs)]
        c["steps"] += 1
        c["tokens"] += tokens
        c["req_steps"] += bs
        c["draft_s"] += grp["draft"]["end"] - grp["draft"]["start"]
        c["verify_s"] += grp["verify"]["end"] - grp["verify"]["start"]
        if "draft_extend_after_decode" in grp:
            d = grp["draft_extend_after_decode"]
            c["extend_s"] += d["end"] - d["start"]
        # Plain decoding needs tokens/bs steps at this batch size to emit the same tokens.
        c["baseline_equiv_s"] += (tokens / bs) * lat(bs)
    out = []
    for b, c in buckets.items():
        if not c["steps"]:
            continue
        sd_s = c["draft_s"] + c["verify_s"] + c["extend_s"]
        out.append({
            "batch_size": b,
            "sd_steps": c["steps"],
            "mean_accept_len": c["tokens"] / c["req_steps"],
            "draft_s": c["draft_s"],
            "verify_s": c["verify_s"],
            "draft_extend_s": c["extend_s"],
            "draft_overhead_frac": (c["draft_s"] + c["extend_s"]) / sd_s,
            "sd_total_s": sd_s,
            "baseline_equiv_s": c["baseline_equiv_s"],
            "speedup_vs_plain_decode": c["baseline_equiv_s"] / sd_s,
            "time_saved_s": c["baseline_equiv_s"] - sd_s,
        })
    return out


def group_forwards(run, gpu):
    """Scheduler forward spans with the EAGLE sub-spans (draft/verify/...) that ran inside each."""
    sgl = sorted(run.gpu_sgl.get(gpu, []), key=lambda r: (r["start"], -r["end"]))
    out, cur = [], None
    for r in sgl:
        if r["name"] == "forward":
            cur = {"fw": r, "inner": {}}
            out.append(cur)
        elif cur is not None and cur["fw"]["start"] <= r["start"] and r["end"] <= cur["fw"]["end"] + 1e-6:
            cur["inner"][r["name"]] = r
    return out


def rollouts(run, gpu):
    """Split grouped forwards into one list per training step (one rollout per step)."""
    groups = group_forwards(run, gpu)
    per_step = collections.defaultdict(list)
    for g in groups:
        i = run.step_of(g["fw"]["start"])
        if i is not None:
            per_step[i].append(g)
    return per_step


def baseline_period_model(base_run):
    """Median decode-step period (start-to-start, incl. host overhead) per batch size, SD-off run."""
    by_bs = collections.defaultdict(list)
    for _, gs in rollouts(base_run, base_run.gpus[0]).items():
        fws = [g["fw"] for g in gs]
        for a, b in zip(fws, fws[1:]):
            if a["mode"] == "DECODE" and b["start"] - a["start"] < 1.0:
                by_bs[a["bs"]].append(b["start"] - a["start"])
    med = {bs: sorted(v)[len(v) // 2] for bs, v in by_bs.items()}
    keys = sorted(med)

    def period(bs):
        if bs in med:
            return med[bs]
        i = bisect.bisect_left(keys, bs)
        lo, hi = keys[max(i - 1, 0)], keys[min(i, len(keys) - 1)]
        if lo == hi:
            return med[lo]
        w = (bs - lo) / (hi - lo)
        return med[lo] * (1 - w) + med[hi] * w

    return period, med


def _tail_start_index(fws, threshold=32):
    """Index right after the last decode forward with batch size above the threshold."""
    for i in range(len(fws) - 1, -1, -1):
        if fws[i]["mode"] == "DECODE" and fws[i]["bs"] > threshold:
            return i + 1 if i + 1 < len(fws) else None
    return None


def straggler_analysis(run, period):
    """Critical-path view of each rollout: head (bs>32), SD switch, and the straggler-bound tail."""
    gpu = run.gpus[0]
    rows, sd_step_points = [], []
    for step, gs in sorted(rollouts(run, gpu).items()):
        fws = [g["fw"] for g in gs]
        start, end = fws[0]["start"], fws[-1]["end"]
        ti = _tail_start_index(fws)
        row = {"step": step, "rollout_s": end - start}
        if ti is None:
            rows.append(row)
            continue
        tail_t0 = fws[ti]["start"]
        row["head_s"] = tail_t0 - start
        row["tail_s"] = end - tail_t0
        row["tail_frac"] = row["tail_s"] / row["rollout_s"]
        tail = gs[ti:]
        sd = [g for g in tail if "verify" in g["inner"]]
        if not sd:
            # Plain decoding: the straggler is in every tail step and gains exactly one token per step.
            n = sum(1 for g in tail if g["fw"]["mode"] == "DECODE")
            row["straggler_tail_tokens"] = n
            row["tail_s_per_straggler_token_ms"] = 1e3 * row["tail_s"] / max(n, 1)
            rows.append(row)
            continue
        sd_t0 = sd[0]["inner"]["draft"]["start"] if "draft" in sd[0]["inner"] else sd[0]["fw"]["start"]
        pre = [g for g in tail if g["fw"]["start"] < sd_t0 and g["fw"]["mode"] == "DECODE" and "verify" not in g["inner"]]
        re = [g for g in tail if g["fw"]["start"] < sd_t0 and g["fw"]["mode"] != "DECODE"]
        row["tail_pre_sd_s"] = (re[0]["fw"]["start"] if re else sd_t0) - tail_t0
        row["reprefill_s"] = (sd_t0 - re[0]["fw"]["start"]) if re else 0.0
        row["sd_phase_s"] = end - sd_t0
        # The straggler: longest request still running at the final verify step.
        last = sd[-1]["inner"]["verify"]
        if "rids" not in last:
            rows.append(row)
            continue
        s_rid = last["rids"][max(range(len(last["rids"])), key=lambda i: last["out_len"][i])]
        row["straggler_len"] = max(last["out_len"])
        s_tokens = s_steps = b_tokens = b_reqs = 0
        gain = loss = n_below = 0.0
        comp = collections.Counter()
        for k, g in enumerate(sd):
            v = g["inner"]["verify"]
            if "rids" not in v or s_rid not in v["rids"]:
                continue
            i = v["rids"].index(s_rid)
            tok = v["acc"][i] + 1
            nxt = sd[k + 1]["fw"]["start"] if k + 1 < len(sd) else g["fw"]["end"]
            per = nxt - g["fw"]["start"] if nxt - g["fw"]["start"] < 1.0 else g["fw"]["end"] - g["fw"]["start"]
            d = g["inner"].get("draft")
            x = g["inner"].get("draft_extend_after_decode")
            comp["draft"] += (d["end"] - d["start"]) if d else 0
            comp["verify"] += v["end"] - v["start"]
            comp["draft_extend"] += (x["end"] - x["start"]) if x else 0
            comp["period"] += per
            plain = tok * period(v["bs"])  # plain decode needs `tok` steps at this batch size
            delta = plain - per
            if delta >= 0:
                gain += delta
            else:
                loss += -delta
                n_below += 1
            s_tokens += tok
            s_steps += 1
            b_tokens += sum(a + 1 for a in v["acc"])
            b_reqs += len(v["acc"])
            sd_step_points.append({
                "step": step, "t": g["fw"]["start"] - sd_t0, "bs": v["bs"], "straggler_tokens": tok,
                "break_even": per / period(v["bs"]), "period_s": per,
            })
        if not s_steps:
            rows.append(row)
            continue
        row["straggler_sd_steps"] = s_steps
        row["straggler_accept_len"] = s_tokens / s_steps
        row["batch_accept_len"] = b_tokens / b_reqs
        row["draft_frac"] = (comp["draft"] + comp["draft_extend"]) / comp["period"]
        row["verify_frac"] = comp["verify"] / comp["period"]
        row["host_frac"] = 1 - row["draft_frac"] - row["verify_frac"]
        row["steps_below_break_even"] = int(n_below)
        row["sd_gain_s"] = gain
        row["sd_loss_s"] = loss
        row["sd_net_vs_plain_s"] = gain - loss - row["reprefill_s"]
        n = len(pre) + s_tokens
        row["straggler_tail_tokens"] = n
        row["tail_s_per_straggler_token_ms"] = 1e3 * row["tail_s"] / max(n, 1)
        rows.append(row)
    return rows, sd_step_points


def plot_straggler(points, out_path):
    steps = sorted({p["step"] for p in points})
    fig, axes = plt.subplots(len(steps), 1, figsize=(14, 2.6 * len(steps)), squeeze=False)
    for ax, s in zip(axes[:, 0], steps):
        ps = [p for p in points if p["step"] == s]
        t = [p["t"] for p in ps]
        ax.scatter(t, [p["straggler_tokens"] for p in ps], s=4,
                   c=["#33a02c" if p["straggler_tokens"] >= p["break_even"] else "#e31a1c" for p in ps])
        ax.plot(t, [p["break_even"] for p in ps], color="k", lw=0.8, label="break-even tokens/step vs plain decode")
        w = max(1, len(ps) // 60)
        vals = [p["straggler_tokens"] for p in ps]
        roll = [sum(vals[max(0, i - w + 1): i + 1]) / len(vals[max(0, i - w + 1): i + 1]) for i in range(len(vals))]
        ax.plot(t, roll, color="#6a3d9a", lw=1.2, label=f"rolling mean ({w} steps)")
        ax2 = ax.twinx()
        ax2.step(t, [p["bs"] for p in ps], where="post", color="#1f78b4", lw=0.7, alpha=0.6)
        ax2.set_ylabel("batch size", color="#1f78b4", fontsize=8)
        ax.set_ylabel("straggler tokens\nper SD step", fontsize=8)
        ax.set_ylim(0, 10)  # at most spec_steps + 1 = 9 tokens per step; clips break-even spikes
        ax.set_title(f"step {s}: straggler acceptance during SD phase (green = SD beat plain decode, red = slower)",
                     loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("seconds since SD switched on")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_critical_path(results, out_path):
    """Stacked bars of rollout time per step: head / pre-SD tail / re-prefill / SD phase (or plain tail)."""
    names = list(results)
    steps = sorted({r["step"] for n in names for r in results[n]})
    fig, ax = plt.subplots(figsize=(1.6 * len(steps) * len(names) + 2, 4.5))
    parts = [("head_s", "Head (batch > 32)", "#1f78b4"), ("tail_pre_sd_s", "Tail before SD on", "#a6cee3"),
             ("reprefill_s", "Re-prefill on SD switch", "#e31a1c"), ("sd_phase_s", "Tail with SD", "#33a02c"),
             ("plain_tail_s", "Tail, plain decode", "#fb9a99")]
    x, labels = 0, []
    for s in steps:
        for n in names:
            row = next((r for r in results[n] if r["step"] == s), None)
            if row is None:
                continue
            if "sd_phase_s" not in row and "tail_s" in row:
                row = {**row, "plain_tail_s": row["tail_s"]}
            bottom = 0
            for key, label, color in parts:
                v = row.get(key) or 0
                ax.bar(x, v, bottom=bottom, color=color, label=label)
                bottom += v
            ax.text(x, bottom, f"{bottom:.0f}s", ha="center", va="bottom", fontsize=7)
            labels.append((x, f"step {s}\n{n}"))
            x += 1
        x += 0.6
    ax.set_xticks([p for p, _ in labels], [l for _, l in labels], fontsize=8)
    ax.set_ylabel("rollout seconds")
    handles, lbls = ax.get_legend_handles_labels()
    uniq = dict(zip(lbls, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper left")
    ax.set_title("Rollout critical path: the straggler-bound tail", loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fmt_table(rows, cols):
    def f(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=trace_dir, baseline (SD off) first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-first", action="store_true", help="exclude step 1 (warmup) from summaries")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    uuid_to_idx = gpu_index_by_uuid()
    runs = [Run(n, d, uuid_to_idx) for n, d in (x.split("=", 1) for x in args.runs)]

    plot_gantt(runs, os.path.join(args.out, "gantt_all_steps.png"))
    common = min(len(r.steps) for r in runs)
    for i in range(1, common + 1):
        plot_gantt(runs, os.path.join(args.out, f"gantt_step{i}.png"), step_filter={i})

    result = {"runs": {}}
    report = []
    for run in runs:
        rows = step_breakdown(run)
        result["runs"][run.name] = {"spec": run.spec, "gpus": run.gpus, "steps": rows}
        cols = ["step", "step_s", "drv_gen_s", "weight_sync", "prefill", "reprefill", "decode", "draft",
                "verify", "draft_extend", "rollout_idle_s", "old_log_prob", "ref_log_prob", "update_actor",
                "decode_tokens", "sd_mean_accept_len", "tail_bs<=32_s"]
        report.append(f"## {run.name} (GPU {run.gpus[0]} view, seconds)\n\n" + fmt_table(rows, cols))

    base, others = runs[0], runs[1:]
    lat, med = baseline_latency_model(base)
    result["baseline_decode_latency_by_bs"] = {str(k): v for k, v in sorted(med.items())}
    for run in others:
        if not run.spec:
            continue
        ce = sd_cost_effectiveness(run, lat)
        result["runs"][run.name]["sd_cost_effectiveness"] = ce
        report.append(
            f"## SD cost-effectiveness in {run.name} vs plain decode latency from {base.name}\n\n"
            + fmt_table(ce, ["batch_size", "sd_steps", "mean_accept_len", "draft_s", "verify_s", "draft_extend_s",
                             "draft_overhead_frac", "sd_total_s", "baseline_equiv_s", "speedup_vs_plain_decode",
                             "time_saved_s"])
        )
    # Straggler / critical-path view: what actually bounds end-to-end rollout time.
    period, pmed = baseline_period_model(base)
    result["baseline_decode_period_by_bs"] = {str(k): v for k, v in sorted(pmed.items())}
    crit = {}
    for run in runs:
        rows, points = straggler_analysis(run, period)
        crit[run.name] = rows
        result["runs"][run.name]["straggler"] = rows
        cols = ["step", "rollout_s", "head_s", "tail_s", "tail_frac", "straggler_tail_tokens",
                "tail_s_per_straggler_token_ms"]
        if run.spec:
            cols += ["tail_pre_sd_s", "reprefill_s", "sd_phase_s", "straggler_len", "straggler_accept_len",
                     "batch_accept_len", "draft_frac", "verify_frac", "host_frac", "steps_below_break_even",
                     "sd_gain_s", "sd_loss_s", "sd_net_vs_plain_s"]
            if points:
                plot_straggler(points, os.path.join(args.out, f"straggler_{run.name}.png"))
        report.append(f"## Straggler critical path: {run.name}\n\n" + fmt_table(rows, cols))
    plot_critical_path(crit, os.path.join(args.out, "critical_path.png"))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(result, f, indent=1)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n\n".join(report) + "\n")
    print("\n\n".join(report))


if __name__ == "__main__":
    main()
