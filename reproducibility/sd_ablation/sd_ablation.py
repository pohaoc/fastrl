"""Rollout-only ablation of FastRL speculative decoding on SkyRL-SQL (one engine configuration per process).

Replays ONE RL rollout step outside the trainer: --prompts SkyRL-SQL prompts x --n trajectories through the same
agent loop as RL (skyrl_env_rollout.run_skyrl_trajectory; 5 turns, 3000 tokens/turn, 8192-token context, T 0.6,
top-p 0.95), on an engine built with exactly verl's arguments (sglang_rollout.py: enable_memory_saver=True,
mem_fraction_static, TP, fa3, no context_length, and for SD the speculative_args block with the
fastrl_trainer.yaml defaults). Arms differ only in the switches below, so differences in rollout time decompose
the SD-on vs SD-off gap:
  --sd 0                      plain engine (verl SD off)
  --sd 0 --mrr/--cg_max_bs/--no_overlap   plain engine + the engine-wide settings verl adds whenever SD is on
  --sd 1 --force_plain 1      EAGLE engine (drafter loaded, graphs, draft-extend on prefill) that never speculates
  --sd 1                      verl SD on (adaptive: speculates while the running batch <= 32)
The engine's scheduler writes a per-iteration step log (FASTRL_STEPLOG_DIR, sglang/srt/fastrl_steplog.py):
time, batch size, CUDA graph, speculated or not, accepted draft tokens, SD transitions. report.py turns the step
logs into time by batch size and acceptance by batch size.
"""

import argparse
import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor


class TurnRecorder:
    """Stands in for the engine in one trajectory's agent loop (which only calls async_generate) and records
    SGLang's per-turn meta_info: prompt tokens, how many of them came from the prefix cache, and the
    speculation counters. Times use the same host clock as the scheduler step log."""

    def __init__(self, engine, traj, sink):
        self.engine, self.traj, self.sink, self.turn = engine, traj, sink, 0

    async def async_generate(self, **kw):
        t0 = time.time()
        out = await self.engine.async_generate(**kw)
        m = out["meta_info"]
        fin = m.get("finish_reason")
        self.sink.append({
            "traj": self.traj, "turn": self.turn, "t0": t0, "t1": time.time(),
            "prompt_tokens": m.get("prompt_tokens"), "cached_tokens": m.get("cached_tokens"),
            "completion_tokens": m.get("completion_tokens"), "spec_verify_ct": m.get("spec_verify_ct"),
            "spec_accept_length": m.get("spec_accept_length"),
            "finish": fin.get("type") if isinstance(fin, dict) else fin,
        })
        self.turn += 1
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", default=None)
    ap.add_argument("--sd", type=int, default=0)
    ap.add_argument("--force_plain", type=int, default=0, help="SD engine that never speculates on decode")
    ap.add_argument("--mrr", type=int, default=None, help="max_running_requests (verl SD-on: 512)")
    ap.add_argument("--cg_max_bs", type=int, default=None, help="cuda_graph_max_bs (verl SD-on: 32)")
    ap.add_argument("--no_overlap", type=int, default=0, help="disable_overlap_schedule (verl SD-on: True)")
    ap.add_argument("--bs_threshold", type=int, default=32)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--mem", type=float, default=0.4)
    ap.add_argument("--attn", default="fa3")
    ap.add_argument("--prompts", type=int, default=256)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "."))
    ap.add_argument("--env_workers", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.environ["FASTRL_STEPLOG_DIR"] = os.path.join(args.out, "steplog")   # inherited by the scheduler processes
    if args.force_plain:
        os.environ["FASTRL_FORCE_PLAIN_DECODE"] = "1"

    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from verl.workers.rollout.sglang_rollout.skyrl_env_rollout import run_skyrl_trajectory

    tok = AutoTokenizer.from_pretrained(args.model)
    root = os.path.join(args.data_root, "SkyRL-SQL")
    df = pd.read_parquet(os.path.join(root, "train.parquet"))
    prompt_ids = [tok.apply_chat_template(list(m), add_generation_prompt=True, tokenize=True) for m in df.prompt]
    keep = [i for i, p in enumerate(prompt_ids) if len(p) <= 4096]          # verl filter_overlong_prompts
    chosen = np.random.default_rng(args.seed).choice(keep, size=args.prompts, replace=False)
    env_cfg = OmegaConf.create({"db_path": os.path.join(root, "db", "data")})
    eos_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    ex = ThreadPoolExecutor(args.env_workers)
    # verl _init_sampling_params + rollout config (temperature 0.6, top_p 0.95, top_k -1)
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": -1, "presence_penalty": 0.0,
                "frequency_penalty": 0.0, "repetition_penalty": 1.0}

    engine_kwargs = dict(
        model_path=args.model, dtype="bfloat16", mem_fraction_static=args.mem, enable_memory_saver=True,
        tp_size=args.tp, load_format="auto", mm_attention_backend="fa3", attention_backend=args.attn,
        random_seed=args.seed, log_level="warning",
    )
    if args.sd:
        engine_kwargs.update(   # verl sglang_rollout.py speculative_args, fastrl_trainer.yaml defaults
            speculative_algorithm="EAGLE", speculative_draft_model_path=args.drafter,
            speculative_num_steps=8, speculative_eagle_topk=4, speculative_num_draft_tokens=48,
            speculative_eagle_mab_algorithm="BEG", apdative_speculative_batch_size_threshold=args.bs_threshold,
            speculative_eagle_mab_configs=["8_4_32", "8_4_16", "8_4_8"], speculative_mab_bs_threshold=[1, 2, 5, 21],
            max_running_requests=512, cuda_graph_max_bs=32, disable_overlap_schedule=True,
        )
    if args.mrr is not None:
        engine_kwargs["max_running_requests"] = args.mrr
    if args.cg_max_bs is not None:
        engine_kwargs["cuda_graph_max_bs"] = args.cg_max_bs
    if args.no_overlap:
        engine_kwargs["disable_overlap_schedule"] = True
    result = {"args": vars(args), "engine_kwargs": {k: v for k, v in engine_kwargs.items() if k != "model_path"},
              "prompt_rows": [int(i) for i in chosen], "phase": "init"}
    rpath = os.path.join(args.out, "result.json")
    json.dump(result, open(rpath, "w"), indent=2)

    async def run():
        import sglang as sgl

        t = time.time()
        engine = sgl.Engine(**engine_kwargs)   # created inside the running loop (SGLang binds it)
        result["init_s"] = time.time() - t
        try:   # Engine.get_server_info() would call run_until_complete inside this running loop
            import dataclasses

            info = {**dataclasses.asdict(engine.tokenizer_manager.server_args), **engine.scheduler_info}
            result["engine"] = {k: info.get(k) for k in ("max_total_num_tokens", "max_running_requests",
                                                         "cuda_graph_max_bs", "disable_overlap_schedule",
                                                         "attention_backend", "max_req_input_len")}
        except Exception as e:   # informational only
            result["engine"] = {"error": repr(e)}
        result["phase"] = "rollout"
        json.dump(result, open(rpath, "w"), indent=2)

        async def one(i, traj):
            r = df.iloc[i]
            return await run_skyrl_trajectory(
                TurnRecorder(engine, traj, turns), tok, prompt_ids[i], list(r.prompt), dict(r.extra_info["tools_kwargs"]), env_cfg, sampling,
                max_turns=5, max_generate_length=3000, max_input_length=8192, stop=["</sql>", "</solution>"],
                eos_ids=eos_ids, executor=ex,
            )

        result["rollout_t0"] = time.time()
        jobs = [int(i) for i in chosen for _ in range(args.n)]
        trajs = await asyncio.gather(*[one(i, k) for k, i in enumerate(jobs)])
        result["rollout_t1"] = time.time()
        await engine.tokenizer_manager.flush_cache()   # as verl does after every rollout (resets adaptive SD state)
        engine.shutdown()
        return trajs

    turns = []
    trajs = asyncio.run(run())
    with open(os.path.join(args.out, "turns.jsonl"), "w") as f:
        f.write("".join(json.dumps(t) + "\n" for t in turns))
    gen = [sum(t.loss_mask) for t in trajs]
    result.update(
        phase="done",
        rollout_s=result["rollout_t1"] - result["rollout_t0"],
        trajectories=len(trajs),
        generated_tokens=int(sum(gen)),
        response_tokens=int(sum(len(t.response_ids) for t in trajs)),
        turns_mean=float(np.mean([t.num_turns for t in trajs])),
        reward_mean=float(np.mean([t.reward for t in trajs])),
        correct=int(sum(t.reward >= 1.0 for t in trajs)),
        stop_reasons={k: int(v) for k, v in pd.Series([t.stop_reason for t in trajs]).value_counts().items()},
    )
    json.dump(result, open(rpath, "w"), indent=2)
    print("RESULT", json.dumps({k: result[k] for k in ("rollout_s", "init_s", "generated_tokens", "turns_mean",
                                                      "reward_mean", "engine")}))


if __name__ == "__main__":  # SGLang spawns subprocesses
    main()
