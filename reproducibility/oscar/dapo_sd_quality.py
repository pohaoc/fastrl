"""Does speculative decoding change what DeepSeek-R1-Distill-Llama-8B generates on DAPO?

In RL (job 7324390, step 1, same prompts, same untrained model) the SD arm's responses were 12,420 tokens long on
average with 11.3% hitting the 32,000-token limit and mean score -0.38, against 8,157 tokens, 0% and -0.05 without
speculation. Speculative decoding should not change the output distribution. One single-GPU engine per variant,
built like verl's (TP 1, fa3, chunked prefill 8192, KV pinned at 327,680 tokens, FastRL's adaptive EAGLE args):
  off        : no speculation (overlap scheduler, as in the SD-off arm)
  sd_rl      : the RL SD arm: adaptive (speculate at <= 32 running requests), BEG over the MAB configs
  sd_always  : speculate on every step (no adaptive switch, no re-prefill); max 32 running so CUDA graphs apply
  sd_triton  : sd_rl with the triton attention backend instead of fa3
  sd_rl_nofix: sd_rl without the switch fix (run with FASTRL_TRANSITION_FIX=0; scheduler.event_loop_normal)
  sd_always_64 / sd_always_256: speculate on every step with up to 64 / 256 running requests (CUDA graphs cover
               <= 32, larger batches run eagerly)
  sd_eager_32: speculate on every step, <= 32 running, CUDA graphs off (the eager path at small batch sizes)
  sd_rl_nograph: sd_rl with CUDA graphs off (no extra "normal decode" graphs/attention backend in adaptive mode)
  sd_never   : adaptive mode that never switches (threshold 0): plain decode inside the EAGLE engine throughout
  sd_rl_t1000: adaptive, threshold 1000: switches after the first 10 decode steps, requeueing every running request
  sd_rl_noprepfix: sd_rl with upstream's decode preparation (run with FASTRL_PREP_FIX=0; scheduler.update_running_batch)
Job 7351369 (greedy pass first): sd_rl and sd_rl_nograph garbage (score -1.0), sd_rl_t1000 and sd_always clean: once
adaptive SD is on, upstream prepared every batch above the threshold as a plain decode step while the worker
speculated on it.
Job 7334995 (gpu-debug, 32 prompts x 4, <= 16K tokens): off, sd_always_64, sd_always_256 (SD at > 64 running, 21
retractions) and sd_eager_32 were all clean, so neither batch size, retraction nor the eager path breaks SD; only
the adaptive mode does.
Job 7333364 showed the switch fires once, at the first greedy request, and SD then stays on for good: the sampled
pass speculated at up to 256 running requests, as the RL SD arm did after its switch (bs > 32 in 31K of 262K SD
decode steps, job 7324390). sd_always (<= 32) and SQL (never above 32) were fine.
Job 7329307: off 7,020 tokens / 0% at the limit / score 0.12; sd_always 6,851 / 0% / 0.10; sd_rl 20,166 / 29% / -1.0
(before the switch fix); sd_triton too slow to finish.
Two passes:
  1. greedy: --greedy_n prompts one at a time, --greedy_tokens tokens; the summary compares token ids with `off`
  2. sampled: --n_prompts x --n at T 1.0 / top-p 1.0, up to 32,000 tokens, all at once (like one GPU's RL share):
     length, share hitting the limit, SkyRL "aime" score, acceptance, repetition in the last 1,000 tokens
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dapo"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", required=True, choices=["off", "sd_rl", "sd_always", "sd_triton", "sd_rl_nofix", "sd_always_64", "sd_always_256", "sd_eager_32",
                                                         "sd_rl_nograph", "sd_never", "sd_rl_t1000", "sd_rl_noprepfix"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--parquet", default="/oscar/data/deeptir/mborjigi/datasets/dapo_math17k/train.parquet")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_new", type=int, default=32000)
    ap.add_argument("--greedy_n", type=int, default=8)
    ap.add_argument("--greedy_tokens", type=int, default=4096)
    ap.add_argument("--kv", type=int, default=327680)
    ap.add_argument("--mem", type=float, default=0.80)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import pandas as pd
    import sglang as sgl
    from aime_reward import compute_score
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    df = pd.read_parquet(args.parquet).sample(args.greedy_n + args.n_prompts, random_state=0)
    ids = [tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=True) for m in df.prompt]
    truth = [rm["ground_truth"] for rm in df.reward_model]
    greedy_ids, sample_ids, sample_truth = ids[: args.greedy_n], ids[args.greedy_n:], truth[args.greedy_n:]

    kw = dict(model_path=args.model, dtype="bfloat16", mem_fraction_static=args.mem, max_total_tokens=args.kv,
              chunked_prefill_size=8192, max_running_requests=256, attention_backend="fa3",
              mm_attention_backend="fa3", random_seed=0, log_level="warning")
    if args.variant != "off":
        kw.update(speculative_algorithm="EAGLE", speculative_draft_model_path=args.drafter,
                  speculative_num_steps=8, speculative_eagle_topk=4, speculative_num_draft_tokens=48,
                  speculative_eagle_mab_algorithm="BEG", apdative_speculative_batch_size_threshold=32,
                  speculative_eagle_mab_configs=["8_4_32", "8_4_16", "8_4_8"],
                  speculative_mab_bs_threshold=[1, 2, 5, 21], cuda_graph_max_bs=32, disable_overlap_schedule=True)
    if args.variant == "sd_always":
        kw.update(apdative_speculative_batch_size_threshold=None, max_running_requests=32)
    if args.variant in ("sd_always_64", "sd_always_256"):
        kw.update(apdative_speculative_batch_size_threshold=None, max_running_requests=int(args.variant.split("_")[-1]))
    if args.variant == "sd_rl_nograph":
        kw.update(disable_cuda_graph=True)
    if args.variant == "sd_never":
        kw.update(apdative_speculative_batch_size_threshold=0)
    if args.variant == "sd_rl_t1000":
        kw.update(apdative_speculative_batch_size_threshold=1000)
    if args.variant == "sd_eager_32":
        kw.update(apdative_speculative_batch_size_threshold=None, max_running_requests=32, disable_cuda_graph=True)
    if args.variant == "sd_triton":
        kw.update(attention_backend="triton")

    res = {"variant": args.variant, "engine_kwargs": {k: v for k, v in kw.items() if k != "model_path"}, "phase": "init"}
    save = lambda: json.dump(res, open(args.out, "w"))  # noqa: E731
    save()
    engine = sgl.Engine(**kw)

    res["phase"] = "greedy"; save()
    t = time.time()
    res["greedy"] = []
    for p in greedy_ids:
        o = engine.generate(input_ids=p, sampling_params={"temperature": 0.0, "max_new_tokens": args.greedy_tokens})
        n_gen = o["meta_info"]["completion_tokens"]
        res["greedy"].append({"ids": list(o["output_ids"])[-n_gen:], "verify_ct": o["meta_info"].get("spec_verify_ct", 0)})
    res["greedy_s"] = time.time() - t

    res["phase"] = "sampled"; save()
    prompts = [p for p in sample_ids for _ in range(args.n)]
    truths = [g for g in sample_truth for _ in range(args.n)]
    t = time.time()
    outs = engine.generate(input_ids=prompts, sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": -1,
                                                               "max_new_tokens": args.max_new})
    res["sampled_s"] = time.time() - t
    engine.shutdown()

    recs = []
    for o, g in zip(outs, truths):
        m = o["meta_info"]
        n_gen = m["completion_tokens"]
        tail = list(o["output_ids"])[-n_gen:][-1000:]
        grams = [tuple(tail[i:i + 4]) for i in range(len(tail) - 3)]
        fin = m["finish_reason"]
        recs.append({"tokens": n_gen, "verify_ct": m.get("spec_verify_ct", 0),
                     "finish": fin.get("type") if isinstance(fin, dict) else str(fin),
                     "score": compute_score("math_dapo", o["text"], g)["score"],
                     "tail_distinct_4gram": round(len(set(grams)) / max(1, len(grams)), 3)})
    res["sampled"] = recs
    n = len(recs)
    tokens, verify = sum(r["tokens"] for r in recs), sum(r["verify_ct"] for r in recs)
    res["summary"] = {
        "n": n, "mean_tokens": round(tokens / n), "hit_limit": round(sum(r["finish"] == "length" for r in recs) / n, 3),
        "mean_score": round(sum(r["score"] for r in recs) / n, 3),
        "accept": round(tokens / verify, 2) if verify else None,
        "tail_distinct_4gram": round(sum(r["tail_distinct_4gram"] for r in recs) / n, 3),
        "greedy_s": round(res["greedy_s"]), "sampled_s": round(res["sampled_s"])}
    res["phase"] = "done"; save()
    print("RESULT", json.dumps(res["summary"]))


if __name__ == "__main__":  # SGLang spawns subprocesses
    main()
