"""Standalone check of the SkyRL-SQL agent loop port against a real SGLang engine (1 GPU, no SD)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import sglang as sgl
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.workers.rollout.sglang_rollout.skyrl_env_rollout import run_skyrl_trajectory

MODEL = "Qwen/Qwen2.5-7B"


async def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    engine = sgl.Engine(model_path=MODEL, mem_fraction_static=0.7, context_length=16384)
    df = pd.read_parquet("SkyRL-SQL/train.parquet").head(8)
    env_cfg = OmegaConf.create({"db_path": "SkyRL-SQL/db/data"})
    ex = ThreadPoolExecutor(16)
    trajs = await asyncio.gather(
        *[
            run_skyrl_trajectory(
                engine, tok,
                tok.apply_chat_template(list(r.prompt), add_generation_prompt=True, tokenize=True),
                list(r.prompt), dict(r.extra_info["tools_kwargs"]), env_cfg,
                {"temperature": 0.6, "top_p": 0.95, "top_k": -1},
                max_turns=5, max_generate_length=3000, max_input_length=8192,
                stop=["</sql>", "</solution>"], eos_ids={tok.eos_token_id, 151645}, executor=ex,
            )
            for r in df.itertuples()
        ]
    )
    for t in trajs:
        assert len(t.response_ids) == len(t.loss_mask)
        print(f"turns={t.num_turns} reward={t.reward} stop={t.stop_reason} len={len(t.response_ids)} "
              f"masked_obs_tokens={t.loss_mask.count(0)}")
    t = max(trajs, key=lambda t: t.num_turns)
    print("---- longest trajectory, tokens with loss_mask=0 shown in [[ ]] ----")
    out, cur = [], None
    for tid, m in zip(t.response_ids, t.loss_mask):
        if m != cur:
            out.append("[[" if m == 0 else "]]") if cur is not None or m == 0 else None
            cur = m
        out.append(tok.decode([tid]))
    print("".join(x for x in out if x)[:4000])
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
