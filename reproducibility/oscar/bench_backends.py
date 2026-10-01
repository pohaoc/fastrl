"""Attention-backend benchmark for FastRL's SGLang EAGLE engine on one node (e.g. B200, where fa3 does not run).

One configuration per process: --backend {triton,fa4,flashinfer,...} --mode {sd,plain}. "plain" is the controlled
SD-off baseline (same EAGLE engine, FASTRL_FORCE_PLAIN_DECODE=1). Generates --n prompts one at a time (batch size
1, the straggler regime) with ignore_eos to a fixed length, traces with FASTRL_TRACE_DIR, and reports the GPU time
per output token by context depth, plus accept length for sd. Summarise several runs with --summarize <dir>.
"""

import argparse
import glob
import json
import os
import time

BUCKETS = [(0, 1024), (1024, 4096), (4096, 8192), (8192, 16384), (16384, 32768)]


def run(args):
    os.makedirs(args.trace_dir, exist_ok=True)
    os.environ["FASTRL_TRACE_DIR"] = args.trace_dir
    if args.mode == "plain":
        os.environ["FASTRL_FORCE_PLAIN_DECODE"] = "1"
    import pandas as pd
    import sglang as sgl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_parquet(args.parquet).sample(args.n, random_state=0)
    prompts = [tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=False) for m in df.prompt]
    t0 = time.time()
    engine = sgl.Engine(
        model_path=args.model, tp_size=args.tp, attention_backend=args.backend, dtype="bfloat16",
        mem_fraction_static=0.6, context_length=args.max_new_tokens + 2048, max_running_requests=1,
        cuda_graph_max_bs=1, speculative_algorithm="EAGLE", speculative_draft_model_path=args.drafter,
        speculative_num_steps=8, speculative_eagle_topk=4, speculative_num_draft_tokens=48, log_level="warning",
    )
    init_s = time.time() - t0
    sp = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "ignore_eos": True}
    res = []
    for p in prompts:
        t = time.time()
        o = engine.generate(p, sp)
        res.append({"wall_s": time.time() - t, "tokens": o["meta_info"]["completion_tokens"],
                    "verify_ct": o["meta_info"].get("spec_verify_ct")})
    engine.shutdown()
    json.dump({"backend": args.backend, "mode": args.mode, "tp": args.tp, "init_s": init_s, "requests": res},
              open(os.path.join(args.trace_dir, "result.json"), "w"), indent=2)
    print(json.dumps({"backend": args.backend, "mode": args.mode, "init_s": round(init_s), "requests": res}))


def summarize(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        rpath = os.path.join(d, "result.json")
        if not os.path.exists(rpath):
            rows.append((os.path.basename(d), "FAILED (no result.json; see the job log)"))
            continue
        r = json.load(open(rpath))
        spans = [json.loads(l) for f in glob.glob(os.path.join(d, "sgl_*.jsonl")) for l in open(f)]
        g0 = sorted({s["gpu_uuid"] for s in spans if s.get("gpu_uuid")})[0]
        spans = sorted((s for s in spans if s.get("gpu_uuid") == g0), key=lambda s: s["start"])
        tok = sum(q["tokens"] for q in r["requests"])
        wall = sum(q["wall_s"] for q in r["requests"])
        # GPU time and tokens per context-depth bucket. Position = tokens generated so far in the request.
        per = {b: [0.0, 0] for b in BUCKETS}
        pos, acc_tok, acc_steps = 0, 0, 0
        for s in spans:
            name, dur = s["name"], s["end"] - s["start"]
            if name == "target_extend":
                pos = 0
            elif name == "decode_nosd":
                b = next((b for b in BUCKETS if b[0] <= pos < b[1]), BUCKETS[-1])
                per[b][0] += dur
                per[b][1] += 1
                pos += 1
            elif name in ("draft", "draft_extend_after_decode", "draft_extend", "verify"):
                b = next((b for b in BUCKETS if b[0] <= pos < b[1]), BUCKETS[-1])
                per[b][0] += dur
                if name == "verify":
                    n = 1 + int(s.get("accepted", 0))
                    per[b][1] += n
                    pos += n
                    acc_tok += n
                    acc_steps += 1
        cells = []
        for b in BUCKETS:
            t, n = per[b]
            cells.append(f"{1000 * t / n:6.2f}" if n else "     -")
        acc = f"{acc_tok / acc_steps:.2f}" if acc_steps else "-"
        rows.append((os.path.basename(d),
                     f"init {r['init_s']:5.0f} s | {tok / wall:6.1f} tok/s wall | accept {acc:>5} | "
                     f"GPU ms/token by depth {'/'.join(f'{b[0] // 1024}-{b[1] // 1024}K' for b in BUCKETS)}: "
                     + " ".join(cells)))
    for name, line in rows:
        print(f"{name:24s} {line}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--mode", choices=["sd", "plain"], default="sd")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--drafter", default="mit-han-lab/Qwen2.5-7B-Eagle-RL")
    ap.add_argument("--parquet", default=os.path.join(os.environ.get("DATA_ROOT", "."), "Eurus-2-RL-Data", "train.parquet"))
    ap.add_argument("--trace_dir", default="bench_backends")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args)


if __name__ == "__main__":  # SGLang spawns subprocesses
    main()
