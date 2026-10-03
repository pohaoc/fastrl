"""SkyRL-gym text environments (e.g. SkyRL-SQL ``text2sql``) as a FastRL rollout.

This reproduces the agent loop of SkyRL's ``SkyRLGymGenerator`` as used by the granular-cais-rl
harness with ``use_conversation_multi_turn=False``: the whole trajectory is ONE assistant message.
Each turn generates until a stop tag (kept in the output), the environment's observation is appended
as plain tokens with loss mask 0, and generation resumes in the same message, token-in-token-out,
until the environment reports done, ``max_turns`` is reached, or the context exceeds
``max_input_length``. The environment (``skyrl_gym.make``) and its reward are used unchanged.

Enable with ``actor_rollout_ref.rollout.skyrl_env.enable=true``; per-row environment extras
(``env_class``, ``db_id``, ``data``, ``reward_spec``) come from ``extra_info.tools_kwargs``.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SkyRLTrajectory:
    response_ids: list[int]
    loss_mask: list[int]
    reward: float
    num_turns: int
    stop_reason: str
    env_metrics: dict = field(default_factory=dict)


async def run_skyrl_trajectory(
    engine,
    tokenizer,
    prompt_ids: list[int],
    messages: list[dict],
    env_extras: dict[str, Any],
    env_config: Any,
    sampling_params: dict[str, Any],
    *,
    max_turns: int,
    max_generate_length: int,
    max_input_length: int,
    stop: list[str],
    eos_ids: set[int],
    executor: ThreadPoolExecutor,
) -> SkyRLTrajectory:
    import skyrl_gym

    loop = asyncio.get_running_loop()
    extras = dict(env_extras)
    extras["max_turns"] = max_turns
    env_class = extras.pop("env_class")
    env = await loop.run_in_executor(executor, lambda: skyrl_gym.make(env_class, env_config=env_config, extras=extras))
    await loop.run_in_executor(executor, env.init, list(messages))

    input_ids = list(prompt_ids)
    prompt_len = len(input_ids)
    loss_mask: list[int] = []
    response_end_idx = prompt_len - 1
    reward = 0.0
    turns = 0
    stop_reason = "stop"
    done = False
    turn_params = dict(sampling_params)
    turn_params.update(n=1, max_new_tokens=max_generate_length, stop=list(stop), no_stop_trim=True)

    while not done:
        if len(input_ids) > max_input_length:
            stop_reason = "length"
            break
        out = await engine.async_generate(input_ids=input_ids, sampling_params=turn_params)
        # With the tokenizer enabled (sync rollout mode), SGLang's output_ids start with up to
        # INIT_INCREMENTAL_DETOKENIZATION_OFFSET (5) context tokens kept for incremental detokenization;
        # only the last completion_tokens are generated. Without this, every turn re-appended the end of the
        # context (prompt tail, observation tail) to input_ids, with loss_mask 1.
        output_ids = list(out["output_ids"])
        n_gen = int(out["meta_info"]["completion_tokens"])
        output_ids = output_ids[len(output_ids) - n_gen :] if n_gen > 0 else []
        finish = out["meta_info"]["finish_reason"]
        stop_reason = finish.get("type", "stop") if isinstance(finish, dict) else str(finish)
        # vLLM's include_stop_str_in_output (used by the harness) cuts the TEXT right after the stop
        # string; SGLang keeps the whole final token, which may carry trailing characters.
        text = out["text"]
        cuts = [text.find(s) + len(s) for s in stop if s in text]
        if cuts:
            text = text[: min(cuts)]
        turns += 1

        step = await loop.run_in_executor(executor, env.step, text)
        done = step["done"]
        reward += float(step["reward"])
        obs_ids: list[int] = []
        for obs in step["observations"]:
            obs_ids += tokenizer.encode(obs["content"], add_special_tokens=False)

        # Continue the same assistant message: drop a trailing EOS before appending the observation.
        if output_ids and output_ids[-1] in eos_ids:
            output_ids = output_ids[:-1]
        input_ids += output_ids + obs_ids
        loss_mask += [1] * len(output_ids) + [0] * len(obs_ids)
        response_end_idx = len(input_ids) - 1 - len(obs_ids)

    env_metrics = env.get_metrics() if hasattr(env, "get_metrics") else {}
    await loop.run_in_executor(executor, env.close)

    # The response ends at the last model token: a trailing observation is not part of it.
    n = response_end_idx - prompt_len + 1
    response_ids = input_ids[prompt_len : prompt_len + n]
    loss_mask = loss_mask[:n]
    if stop_reason != "length" and response_ids and response_ids[-1] not in eos_ids:
        response_ids.append(tokenizer.eos_token_id)
        loss_mask.append(1)
    return SkyRLTrajectory(response_ids, loss_mask, reward, turns, stop_reason, env_metrics)


def build_env_config(skyrl_cfg) -> Any:
    """Per-env-class config object, matching SkyRL's ``getattr(skyrl_gym_cfg, env_class)``."""
    from omegaconf import OmegaConf

    env_cfgs = OmegaConf.to_container(skyrl_cfg.get("env_configs", {}), resolve=True) or {}
    return {name: OmegaConf.create(cfg) for name, cfg in env_cfgs.items()}
