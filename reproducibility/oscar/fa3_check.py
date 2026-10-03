"""Does FastRL's EAGLE engine work with attention_backend=fa3 on this GPU (e.g. L40S, sm89)?

One engine configuration per process, built with exactly the arguments verl passes in-loop
(verl/workers/rollout/sglang_rollout/sglang_rollout.py: enable_memory_saver=True, no context_length,
mm_attention_backend=fa3, and the speculative_args block with the fastrl_trainer.yaml defaults).
Draft CUDA graphs for every MAB strategy are captured at engine init, so init alone covers the
capture that crashed in-loop on L40S.

Two phases, so the run exercises every MAB batch-size range and checks correctness, not just crashes:
  batch : --n SkyRL-SQL prompts submitted at once (SQL sampling: T=0.6, top_p=0.95, stop </sql>
          </solution>), so the running batch drains from n through 21-32, 5-20, 2-4, 1.
  greedy: --n_greedy prompts one at a time at T=0. Their token ids are saved so variants can be
          compared: greedy SD is lossless, so fa3-SD vs fa3-plain (and vs triton) should agree on long
          prefixes, and a broken draft/verify kernel shows up as short prefixes and accept length ~1.
Summarise several runs with --summarize <dir>.
"""

import argparse
import glob
import json
import os
import time


def run(args):
    import pandas as pd
    import sglang as sgl
    import torch
    from transformers import AutoTokenizer

    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_parquet(args.parquet).sample(args.n + args.n_greedy, random_state=0)
    prompts = [tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=False) for m in df.prompt]

    spec = {}
    if args.sd:
        spec = dict(
            speculative_algorithm="EAGLE", speculative_draft_model_path=args.drafter,
            speculative_num_steps=8, speculative_eagle_topk=4, speculative_num_draft_tokens=48,
            speculative_eagle_mab_algorithm="BEG", apdative_speculative_batch_size_threshold=32,
            speculative_eagle_mab_configs=["8_4_32", "8_4_16", "8_4_8"], speculative_mab_bs_threshold=[1, 2, 5, 21],
            max_running_requests=512, cuda_graph_max_bs=32, disable_overlap_schedule=True,
        )
    extra = {"context_length": args.context_length} if args.context_length else {}
    result = {"args": vars(args), "gpu": torch.cuda.get_device_name(0), "phase": "init"}
    rpath = os.path.join(args.out, "result.json")
    json.dump(result, open(rpath, "w"), indent=2)  # left as phase=init if the engine dies

    t0 = time.time()
    engine = sgl.Engine(
        model_path=args.model, dtype="bfloat16", mem_fraction_static=args.mem, enable_memory_saver=True,
        tp_size=args.tp, load_format="auto", mm_attention_backend="fa3", attention_backend=args.backend,
        log_level="info", **spec, **extra,
    )
    result["init_s"] = time.time() - t0

    result["phase"] = "batch"
    json.dump(result, open(rpath, "w"), indent=2)
    sp = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": args.max_new_tokens,
          "stop": ["</sql>", "</solution>"], "no_stop_trim": True}
    t = time.time()
    outs = engine.generate(prompts[: args.n], sp)
    result["batch"] = {
        "wall_s": time.time() - t,
        "tokens": sum(o["meta_info"]["completion_tokens"] for o in outs),
        "verify_ct": sum(o["meta_info"].get("spec_verify_ct", 0) for o in outs),
    }

    result["phase"] = "greedy"
    json.dump(result, open(rpath, "w"), indent=2)
    greedy = []
    sp = {"temperature": 0.0, "max_new_tokens": args.max_new_tokens}
    for p in prompts[args.n:]:
        t = time.time()
        o = engine.generate(p, sp)
        greedy.append({"wall_s": time.time() - t, "tokens": o["meta_info"]["completion_tokens"],
                       "verify_ct": o["meta_info"].get("spec_verify_ct", 0), "ids": tok.encode(o["text"])})
    result["greedy"] = greedy
    engine.shutdown()
    result["phase"] = "done"
    json.dump(result, open(rpath, "w"), indent=2)
    print("RESULT", json.dumps({k: result[k] for k in ("gpu", "init_s", "batch")}))


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def summarize(root, ref_name):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        p = os.path.join(d, "result.json")
        runs[os.path.basename(d)] = json.load(open(p)) if os.path.exists(p) else None
    ref = runs.get(ref_name)
    for name, r in runs.items():
        if r is None or r["phase"] != "done":
            print(f"{name:28s} FAILED during {r['phase'] if r else 'startup'} (see the job log)")
            continue
        b, g = r["batch"], r["greedy"]
        acc_b = f"{b['tokens'] / b['verify_ct']:.2f}" if b["verify_ct"] else "-"
        gt, gv = sum(x["tokens"] for x in g), sum(x["verify_ct"] for x in g)
        acc_g = f"{gt / gv:.2f}" if gv else "-"
        agree = "-"
        if ref and ref.get("phase") == "done" and name != ref_name:
            fr = [_common_prefix(x["ids"], y["ids"]) / max(1, min(len(x["ids"]), len(y["ids"])))
                  for x, y in zip(g, ref["greedy"])]
            agree = f"{100 * sum(fr) / len(fr):.0f}%"
        print(f"{name:28s} init {r['init_s']:4.0f} s | batch {b['tokens'] / b['wall_s']:7.1f} tok/s accept {acc_b:>5}"
              f" | greedy bs=1 {gt / sum(x['wall_s'] for x in g):6.1f} tok/s accept {acc_g:>5}"
              f" | greedy prefix agreement vs {ref_name}: {agree}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", default=None)
    ap.add_argument("--ref", default="coder-fa3-plain-tp2", help="run that --summarize compares greedy outputs against")
    ap.add_argument("--backend", default="fa3")
    ap.add_argument("--sd", type=int, default=1)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--mem", type=float, default=0.4, help="upstream examples/grpo_7B.sh uses 0.4")
    ap.add_argument("--context_length", type=int, default=None, help="verl passes none; set only to test that")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n_greedy", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--drafter", default="mit-han-lab/Qwen2.5-7B-Eagle-RL")
    ap.add_argument("--parquet", default=os.path.join(os.environ.get("DATA_ROOT", "."), "SkyRL-SQL", "train.parquet"))
    ap.add_argument("--out", default="fa3_check")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize, args.ref)
    else:
        run(args)


if __name__ == "__main__":  # SGLang spawns subprocesses
    main()
