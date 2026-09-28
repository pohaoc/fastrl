"""Where does speculative decoding's overhead come from? Drafter vs verification cost per step.

Each SD decode step (draft -> verify -> draft_extend) is decomposed against the ONE plain decode
step the same engine would run at the same batch size (measured in the SD-off baseline):

    SD step = plain step + drafting + draft-extend + (verify - plain step GPU time) + extra host time

- drafting        : the drafter's autoregressive draft passes (spec_steps of them) -- pure overhead
- draft-extend    : re-running the drafter on accepted tokens to refresh its KV cache -- pure overhead
- extra verify    : verifying the draft tree costs more than decoding one token per request
- extra host      : scheduling/bookkeeping between steps beyond what a plain step needs
- break-even      : SD step time / plain step period = tokens per request per step needed to tie
- net_saved_s     : plain decode of the SAME tokens at the batch-average acceptance minus SD time
                    (a throughput view); straggler_net_s is the end-to-end view (the straggler's own
                    tokens, net of re-prefill), from analyze.straggler_analysis

Usage: python sd_cost.py --runs off=<sd_off traces> on=<sd_on traces> --out <dir> [--name Eurus]
"""

import argparse
import collections
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

BUCKETS = ["1", "2-4", "5-8", "9-16", "17-32"]
PARTS = [
    ("plain", "Plain step it replaces", "#1f78b4"),
    ("draft", "Drafting", "#ff7f00"),
    ("dext", "Draft-extend", "#fdbf6f"),
    ("xverify", "Extra verification", "#33a02c"),
    ("xhost", "Extra host time", "#b15928"),
]


def bucket(bs):
    return "1" if bs == 1 else "2-4" if bs <= 4 else "5-8" if bs <= 8 else "9-16" if bs <= 16 else "17-32"


def plain_models(base):
    """Median plain decode forward (GPU) time and step period (incl. host) per batch size."""
    fwd, per = collections.defaultdict(list), collections.defaultdict(list)
    for _, gs in A.rollouts(base, base.gpus[0]).items():
        fws = [g["fw"] for g in gs]
        for a, b in zip(fws, fws[1:]):
            if a["mode"] == "DECODE" and b["start"] - a["start"] < 1.0:
                fwd[a["bs"]].append(a["end"] - a["start"])
                per[a["bs"]].append(b["start"] - a["start"])

    def model(d):
        med = {k: statistics.median(v) for k, v in d.items()}
        keys = sorted(med)

        def f(bs):
            if bs in med:
                return med[bs]
            lo = max([k for k in keys if k < bs], default=keys[0])
            hi = min([k for k in keys if k > bs], default=keys[-1])
            if lo == hi:
                return med[lo]
            w = (bs - lo) / (hi - lo)
            return med[lo] * (1 - w) + med[hi] * w

        return f

    return model(fwd), model(per)


def sd_steps(run, plain_fwd, plain_per):
    """One record per SD decode step with its cost decomposition (seconds)."""
    out = []
    for step, gs in sorted(A.rollouts(run, run.gpus[0]).items()):
        for k, g in enumerate(gs):
            inner = g["inner"]
            if "verify" not in inner or "draft" not in inner:
                continue
            fw, v, d = g["fw"], inner["verify"], inner["draft"]
            x = inner.get("draft_extend_after_decode")
            nxt = gs[k + 1]["fw"]["start"] if k + 1 < len(gs) else None
            period = (nxt - fw["start"]) if nxt is not None and nxt - fw["start"] < 1.0 else fw["end"] - fw["start"]
            bs = v["bs"]
            t_draft = d["end"] - d["start"]
            t_verify = v["end"] - v["start"]
            t_dext = (x["end"] - x["start"]) if x else 0.0
            host_sd = period - (t_draft + t_verify + t_dext)
            pf, pp = plain_fwd(bs), plain_per(bs)
            out.append({
                "step": step, "bs": bs, "strategy": v.get("strategy", "?"),
                "tokens": v["accepted"] + bs, "period": period,
                "draft": t_draft, "verify": t_verify, "dext": t_dext, "host": host_sd,
                "plain_fwd": pf, "plain_period": pp,
                # decomposition against one plain step (sums to `period`)
                "plain": pp, "xverify": t_verify - pf, "xhost": host_sd - (pp - pf),
            })
    return out


def summarize(recs, key):
    groups = collections.OrderedDict()
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    rows = []
    for k, rs in groups.items():
        n = len(rs)
        tot = {p: sum(r[p] for r in rs) for p in ("period", "draft", "dext", "verify", "host", "plain", "xverify", "xhost")}
        tokens = sum(r["tokens"] for r in rs)
        req_steps = sum(r["bs"] for r in rs)
        plain_equiv = sum(r["tokens"] / r["bs"] * r["plain_period"] for r in rs)
        rows.append({
            "group": k, "sd_steps": n,
            "draft_ms": 1e3 * tot["draft"] / n, "dext_ms": 1e3 * tot["dext"] / n,
            "verify_ms": 1e3 * tot["verify"] / n, "host_ms": 1e3 * tot["host"] / n,
            "sd_step_ms": 1e3 * tot["period"] / n, "plain_step_ms": 1e3 * tot["plain"] / n,
            "xverify_ms": 1e3 * tot["xverify"] / n, "xhost_ms": 1e3 * tot["xhost"] / n,
            "overhead_ms": 1e3 * (tot["period"] - tot["plain"]) / n,
            "drafter_share_of_overhead": (tot["draft"] + tot["dext"]) / max(1e-9, tot["period"] - tot["plain"]),
            "break_even_tokens": tot["period"] / tot["plain"],
            "tokens_per_req_step": tokens / req_steps,
            "sd_s": tot["period"], "draft_s": tot["draft"], "dext_s": tot["dext"], "verify_s": tot["verify"],
            "xverify_s": tot["xverify"], "xhost_s": tot["xhost"], "plain_part_s": tot["plain"],
            "plain_equiv_s": plain_equiv, "net_saved_s": plain_equiv - tot["period"],
            "strategies": dict(collections.Counter(r["strategy"] for r in rs)),
        })
    return rows


def fmt(rows, cols):
    def f(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                     + ["| " + " | ".join(f(r.get(c)) for c in cols) + " |" for r in rows])


def plot(name, by_step, by_bs, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    # Left: per training step, SD-phase seconds split by component, vs plain decode of the same tokens.
    xs = range(len(by_step))
    bottom = [0.0] * len(by_step)
    for key, label, color in PARTS:
        vals = [r[f"{key}_s" if key != "plain" else "plain_part_s"] for r in by_step]
        a1.bar(xs, vals, bottom=bottom, color=color, label=label, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    a1.scatter(xs, [r["plain_equiv_s"] for r in by_step], marker="_", s=900, color="k", zorder=3,
               label="Plain decode of the same tokens (batch average)")
    for i, r in enumerate(by_step):
        e2e = r.get("straggler_net_s")
        txt = f"throughput {r['net_saved_s']:+.0f}s" + (f"\nstraggler {e2e:+.0f}s" if e2e is not None else "")
        a1.text(i, max(bottom[i], r["plain_equiv_s"]) * 1.02, txt, ha="center", fontsize=8)
    a1.set_xticks(list(xs), [f"step {r['group']}" for r in by_step])
    a1.set_ylabel("seconds in SD decode steps (GPU 0)")
    a1.set_title(f"{name}: SD time per training step vs plain decode of the same tokens", loc="left", fontsize=10)
    a1.set_ylim(0, max(max(bottom), max(r["plain_equiv_s"] for r in by_step)) * 1.22)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8, frameon=False)
    # Right: per batch size, one SD step's time split by component, vs tokens it produced.
    xs2 = range(len(by_bs))
    bottom = [0.0] * len(by_bs)
    for key, label, color in PARTS:
        vals = [r[f"{key}_ms" if key != "plain" else "plain_step_ms"] for r in by_bs]
        a2.bar(xs2, vals, bottom=bottom, color=color, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, r in enumerate(by_bs):
        a2.text(i, bottom[i] * 1.02, f"break-even {r['break_even_tokens']:.1f}\nactual {r['tokens_per_req_step']:.1f} tok",
                ha="center", fontsize=8)
    a2.set_xticks(list(xs2), [f"bs {r['group']}\n{max(r['strategies'], key=r['strategies'].get)}" for r in by_bs])
    a2.set_ylabel("ms per SD step (mean)")
    a2.set_ylim(0, max(bottom) * 1.25)
    a2.set_title(f"{name}: one SD step vs the plain step it replaces, by batch size", loc="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True, help="off=<dir> on=<dir>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dirs = dict(x.split("=", 1) for x in args.runs)
    u = A.gpu_index_by_uuid()
    base, sd = A.Run("off", dirs["off"], u), A.Run("on", dirs["on"], u)
    pf, pp = plain_models(base)
    recs = sd_steps(sd, pf, pp)
    by_step = summarize(recs, lambda r: r["step"])
    # End-to-end view: SD vs plain decode for the straggler's own tokens (analyze.straggler_analysis).
    period, _ = A.baseline_period_model(base)
    strag = {r["step"]: r for r in A.straggler_analysis(sd, period)[0]}
    for r in by_step:
        r["straggler_net_s"] = strag.get(r["group"], {}).get("sd_net_vs_plain_s")
        r["straggler_tokens_per_step"] = strag.get(r["group"], {}).get("straggler_accept_len")
    by_bs = summarize(sorted(recs, key=lambda r: BUCKETS.index(bucket(r["bs"]))), lambda r: bucket(r["bs"]))
    name = args.name or os.path.basename(os.path.dirname(dirs["on"].rstrip("/")))
    plot(name, by_step, by_bs, os.path.join(args.out, "sd_cost.png"))
    json.dump({"by_step": by_step, "by_batch_size": by_bs}, open(os.path.join(args.out, "sd_cost.json"), "w"), indent=1)
    cols = ["group", "sd_steps", "tokens_per_req_step", "break_even_tokens", "sd_step_ms", "plain_step_ms", "draft_ms",
            "dext_ms", "verify_ms", "xverify_ms", "xhost_ms", "drafter_share_of_overhead"]
    cols_s = ["group", "sd_steps", "sd_s", "draft_s", "dext_s", "verify_s", "xverify_s", "xhost_s", "plain_part_s",
              "plain_equiv_s", "net_saved_s", "straggler_net_s", "tokens_per_req_step", "straggler_tokens_per_step",
              "break_even_tokens"]
    md = (f"## {name}: SD cost per training step (seconds, GPU 0)\n\n" + fmt(by_step, cols_s)
          + f"\n\n## {name}: one SD step by batch size (ms)\n\n" + fmt(by_bs, cols) + "\n")
    open(os.path.join(args.out, "sd_cost.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
