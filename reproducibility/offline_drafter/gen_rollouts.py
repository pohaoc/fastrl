"""Sample Qwen2.5-7B rollouts on DAPO-Math or SkyRL-SQL prompts, as training data for an offline EAGLE drafter.

The drafter must learn the policy's own distribution, so responses come from the target model at the RL
sampling settings (run_grpo_7B_4gpu.sh): DAPO temperature 0.9 / top-p 1.0; SQL temperature 0.6 / top-p 0.95
in the SkyRL-gym text2sql agent loop (<=5 turns, 3000 tokens per turn, 8192-token context limit).

Output: <out>/data/train.parquet, pre-tokenized for eagle-train's eagle_datagen.py:
    input_ids  prompt + response token ids exactly as the engine produced them
    loss_mask  1 on generated tokens, 0 on prompt and (SQL) environment-observation tokens
plus metadata (index, prompt_len, finish_reason / reward, turns, response text).
DAPO also writes <out>/heldout.parquet: the prompts NOT used for training, for acceptance benchmarks.

Usage (fastrl env, 1 GPU, from the repo root):
    python reproducibility/offline_drafter/gen_rollouts.py --dataset dapo --out <dir> [--num_prompts 9200]
    python reproducibility/offline_drafter/gen_rollouts.py --dataset sql --out <dir> [--n 4]
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
        attention_backend=args.attention_backend,  # flashinfer's JIT is broken in this env
        random_seed=args.seed,
        log_level="warning",
    )


def gen_dapo(args, tok):
    df = pd.read_parquet(os.path.join(args.data_root, "DAPO-Math-17k", "train.parquet"))
    perm = np.random.default_rng(args.seed).permutation(len(df))
    train_idx, heldout_idx = perm[: args.num_prompts], perm[args.num_prompts :]
    df.iloc[heldout_idx].to_parquet(os.path.join(args.out, "heldout.parquet"))
    prompts = [tok.apply_chat_template(list(df.iloc[i]["prompt"]), add_generation_prompt=True, tokenize=True)
               for i in train_idx]
    engine = engine_for(args, context_length=8192)
    t0 = time.time()
    outs = engine.generate(
        input_ids=prompts,
        sampling_params={"temperature": 0.9, "top_p": 1.0, "max_new_tokens": args.max_new_tokens},
    )
    print(f"generated {len(outs)} responses in {time.time() - t0:.0f} s")
    engine.shutdown()
    rows = []
    for i, p, o in zip(train_idx, prompts, outs):
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
    return rows


def gen_sql(args, tok):
    from omegaconf import OmegaConf

    from verl.workers.rollout.sglang_rollout.skyrl_env_rollout import run_skyrl_trajectory

    root = os.path.join(args.data_root, "SkyRL-SQL")
    df = pd.read_parquet(os.path.join(root, "train.parquet"))
    env_cfg = OmegaConf.create({"db_path": os.path.join(root, "db", "data")})
    ex = ThreadPoolExecutor(64)
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

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
        # created inside the running loop, as in reproducibility/dataset/dev/test_skyrl_loop.py
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
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--num_prompts", type=int, default=9200, help="DAPO: training prompts (the rest are held out)")
    ap.add_argument("--n", type=int, default=4, help="SQL: trajectories per prompt")
    ap.add_argument("--max_new_tokens", type=int, default=3584, help="DAPO: the drafter trains on <=4K-token sequences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem_fraction", type=float, default=0.85)
    ap.add_argument("--attention_backend", default="fa3")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "data"), exist_ok=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = (gen_dapo if args.dataset == "dapo" else gen_sql)(args, tok)
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
