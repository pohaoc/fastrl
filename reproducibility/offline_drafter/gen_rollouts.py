"""Sample the target model's rollouts on DAPO-Math or SkyRL-SQL prompts, as training data for an offline EAGLE drafter.

The drafter must learn the policy's own distribution, so responses come from the target model (--model, the RL
policy before training) at the RL sampling settings: DAPO --temperature / --top_p (defaults 0.9 / 1.0 = examples/grpo_7B.sh;
per-target values in paths.sh); SQL
temperature 0.6 / top-p 0.95 in the SkyRL-gym text2sql agent loop used by RL (skyrl_env_rollout.py; <=5 turns,
3000 tokens per turn, 8192-token context limit, as in reproducibility/oscar/grpo_sql.sbatch).

Output: <out>/data/train.parquet, pre-tokenized for eagle-train's eagle_datagen.py:
    input_ids  prompt + response token ids exactly as the engine produced them
    loss_mask  1 on generated tokens, 0 on prompt and (SQL) environment-observation tokens
plus metadata (index, prompt_len, finish_reason / reward, turns, response text).
DAPO also writes <out>/heldout.parquet: the prompts NOT used for training, for acceptance benchmarks.

Usage (fastrl env, 1 GPU, from the repo root):
    python reproducibility/offline_drafter/gen_rollouts.py --dataset dapo --model <target> --out <dir> [--num_prompts 9200]
    python reproducibility/offline_drafter/gen_rollouts.py --dataset sql --model <target> --out <dir> [--n 4]
"""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


def engine_for(args, context_length):
    import sglang as sgl

    return sgl.Engine(
        model_path=args.model,
        tp_size=1,
        mem_fraction_static=args.mem_fraction,
        context_length=context_length,
        # Cap concurrency so every running request fits in the KV pool even at the full response length: SGLang
        # then never retracts (evicts and requeues) requests, the path whose KV accounting failed in job 7060958.
        max_running_requests=args.max_running,
        attention_backend=args.attention_backend,  # flashinfer's JIT is broken in this env
        random_seed=args.seed,
        log_level="warning",
    )


def dapo_split(args):
    df = pd.read_parquet(args.dapo_parquet or os.path.join(args.data_root, "DAPO-Math-17k", "train.parquet"))
    if args.max_prompt_tokens:   # same filter as the RL runs' max prompt length
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        n = [len(tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=True)) for m in df.prompt]
        df = df[[x <= args.max_prompt_tokens for x in n]].reset_index(drop=True)
    perm = np.random.default_rng(args.seed).permutation(len(df))
    train_idx, heldout_idx = perm[: args.num_prompts], perm[args.num_prompts :]
    chunks = [train_idx[k : k + args.chunk] for k in range(0, len(train_idx), args.chunk)]
    return df, chunks, heldout_idx


def part_path(args, k):
    return os.path.join(args.out, "parts", f"chunk_{k:04d}.parquet")


def gen_dapo_worker(args, tok):
    """One process per GPU: generate this shard's chunks of prompts, writing each chunk as soon as it is done.
    Chunks already on disk are skipped, so a rerun resumes after a crash."""
    df, chunks, _ = dapo_split(args)
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)
    mine = [k for k in range(len(chunks)) if k % args.num_shards == args.shard and not os.path.exists(part_path(args, k))]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} chunks to generate", flush=True)
    if not mine:
        return
    tmpl = lambda i: tok.apply_chat_template(list(df.iloc[i]["prompt"]), add_generation_prompt=True, tokenize=True)  # noqa: E731
    max_prompt = max(len(tmpl(i)) for c in chunks for i in c)
    engine = engine_for(args, context_length=max_prompt + args.max_new_tokens + 16)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    for k in mine:
        idx = chunks[k]
        prompts = [tmpl(i) for i in idx]
        t0 = time.time()
        outs = engine.generate(input_ids=prompts, sampling_params=sampling)
        rows = []
        for i, p, o in zip(idx, prompts, outs):
            # SGLang (tokenizer enabled) prefixes output_ids with up to 5 context tokens; keep the generated ones
            out_ids = list(o["output_ids"])
            n_gen = int(o["meta_info"]["completion_tokens"])
            out_ids = out_ids[len(out_ids) - n_gen :] if n_gen > 0 else []
            fin = o["meta_info"]["finish_reason"]
            rows.append({
                "index": int(i), "input_ids": list(p) + out_ids, "loss_mask": [0] * len(p) + [1] * len(out_ids),
                "prompt_len": len(p), "response_len": len(out_ids),
                "finish_reason": fin.get("type") if isinstance(fin, dict) else str(fin), "response": o["text"],
            })
        tmp = part_path(args, k) + ".tmp"
        pd.DataFrame(rows).to_parquet(tmp)
        os.replace(tmp, part_path(args, k))
        n_tok = sum(r["response_len"] for r in rows)
        print(f"shard {args.shard}: chunk {k} done, {len(rows)} rows, {n_tok} tokens in {time.time() - t0:.0f} s "
              f"({n_tok / (time.time() - t0):.0f} tok/s)", flush=True)
    engine.shutdown()


def merge_dapo(args):
    """Join the chunk files into data/train.parquet (and write the held-out prompts); fails if any chunk is missing."""
    df, chunks, heldout_idx = dapo_split(args)
    missing = [k for k in range(len(chunks)) if not os.path.exists(part_path(args, k))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(chunks)} chunks missing: {missing}")
    df.iloc[heldout_idx].to_parquet(os.path.join(args.out, "heldout.parquet"))
    return pd.concat([pd.read_parquet(part_path(args, k)) for k in range(len(chunks))]).to_dict("records")


def gen_sql(args, tok):
    from omegaconf import OmegaConf

    from verl.workers.rollout.sglang_rollout.skyrl_env_rollout import run_skyrl_trajectory

    root = os.path.join(args.data_root, "SkyRL-SQL")
    df = pd.read_parquet(os.path.join(root, "train.parquet"))
    env_cfg = OmegaConf.create({"db_path": os.path.join(root, "db", "data")})
    ex = ThreadPoolExecutor(64)
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}

    async def one(engine, r):
        p = tok.apply_chat_template(list(r.prompt), add_generation_prompt=True, tokenize=True)
        t = await run_skyrl_trajectory(
            engine, tok, p, list(r.prompt), dict(r.extra_info["tools_kwargs"]), env_cfg,
            {"temperature": 0.6, "top_p": 0.95, "top_k": -1},
            max_turns=5, max_generate_length=3000, max_input_length=8192,
            stop=["</sql>", "</solution>"], eos_ids=eos_ids, executor=ex,
        )
        return r, p, t

    async def run_all():
        # created inside the running loop (SGLang binds its event loop at construction)
        engine = engine_for(args, context_length=16384)
        t0 = time.time()
        res = await asyncio.gather(*[one(engine, r) for r in df.itertuples() for _ in range(args.n)])
        print(f"generated {len(res)} trajectories in {time.time() - t0:.0f} s")
        engine.shutdown()
        return res

    results = asyncio.run(run_all())
    rows = []
    for r, p, t in results:
        rows.append({
            "index": int(r.extra_info["index"]), "input_ids": list(p) + list(t.response_ids),
            "loss_mask": [0] * len(p) + list(t.loss_mask), "prompt_len": len(p), "response_len": len(t.response_ids),
            "finish_reason": t.stop_reason, "reward": float(t.reward), "turns": int(t.num_turns),
            "response": tok.decode(t.response_ids),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["dapo", "sql"], required=True)
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "."))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="target model (policy) or its local dir")
    ap.add_argument("--num_prompts", type=int, default=9200, help="DAPO: training prompts (the rest are held out)")
    ap.add_argument("--n", type=int, default=4, help="SQL: trajectories per prompt")
    ap.add_argument("--max_new_tokens", type=int, default=3584, help="DAPO: response cap (datagen truncates rows to its max length)")
    ap.add_argument("--temperature", type=float, default=0.9, help="DAPO: the RL rollout temperature for this target")
    ap.add_argument("--top_p", type=float, default=1.0, help="DAPO: the RL rollout top-p for this target")
    ap.add_argument("--dapo_parquet", default=None, help="DAPO: prompt parquet (default: $DATA_ROOT/DAPO-Math-17k)")
    ap.add_argument("--max_prompt_tokens", type=int, default=None, help="DAPO: drop longer prompts (chat template on)")
    ap.add_argument("--shard", type=int, default=0, help="DAPO: this worker's shard (one worker process per GPU)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=100, help="DAPO: prompts per chunk file")
    ap.add_argument("--merge", action="store_true", help="DAPO: join the chunk files instead of generating")
    ap.add_argument("--max_running", type=int, default=None, help="max concurrent requests per engine")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--attention_backend", default="fa3")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.dataset == "dapo" and not args.merge:
        gen_dapo_worker(args, tok)
        return
    rows = merge_dapo(args) if args.dataset == "dapo" else gen_sql(args, tok)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(args.out, "data", "train.parquet"))

    stats = {
        "rows": len(out), "total_tokens": int(out.input_ids.map(len).sum()),
        "generated_tokens": int(out.loss_mask.map(sum).sum()),
        "response_len_mean": float(out.response_len.mean()), "response_len_p99": float(out.response_len.quantile(0.99)),
        "finish_reason": out.finish_reason.value_counts().to_dict(),
        "seq_len_max": int(out.input_ids.map(len).max()),
    }
    if "reward" in out:
        stats.update(reward_mean=float(out.reward.mean()), turns_mean=float(out.turns.mean()))
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":  # SGLang spawns subprocesses
    main()
