"""verl custom reward for DAPO-Math-17k that reproduces the SysX runs' SkyRL-gym "aime" environment exactly.

The SysX DAPO runs (~/scratch/runs/dapo-b128-*, env_class = "aime") score each response with
skyrl_gym.envs.aime (skyrl-gym 0.3.0, the version installed here): the last "Answer:" line, normalized, compared
with the ground truth, +1 / -1. verl's built-in math_dapo reward is the same DAPO function except that its answer
pattern does not stop at "<"; calling SkyRL-gym's scorer removes that difference.

Use: custom_reward_function.path=reproducibility/dapo/aime_reward.py custom_reward_function.name=compute_score
"""

from skyrl_gym.envs.aime import utils as aime_utils


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Same call as skyrl_gym.envs.aime.env.AIMEEnv.step
    info = aime_utils.compute_score(solution_str, ground_truth)
    return {"score": float(info["score"]), "acc": float(info["acc"])}
