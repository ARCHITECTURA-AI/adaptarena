"""The debugging story, regenerated: the real failed PPO run and its fix, from logged diagnostics.

First symptom (recorded during development): on a tiny solvable scenario with a certified optimal return,
PPO sat at random-level survival, with value loss of 30-90, explained variance swinging between -8 and +2, and
occasional approximate-KL spikes above 6. Cause: with a reward of 1 per step and gamma 0.99 the returns are ~100,
so the unscaled value-loss gradient dominated the single joint gradient-norm clip (0.5) and shrank the policy
gradient to almost nothing. Fix: scale the reward by 0.01 inside the learner (`PPOConfig.reward_scale`), which
leaves the environment and every reported survival number untouched.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..core import ScenarioSpec, build_scenario, plan
from ..core.env import EnvConfig
from .experiments import tier_dir, write_json
from .models import FFPolicy
from .ppo import PPOConfig, SamplerSource, train

SPEC = ScenarioSpec(layout_id=2, wall_pct=0, arena=5, n_food=1, n_water=1, food_delay_min=8, food_delay_max=8,
                    water_delay_min=8, water_delay_max=8, horizon=200)


def run(tier_name: str = "full", updates: int = 150) -> dict:
    import torch
    torch.set_num_threads(1)
    out = {"scenario": SPEC.to_json(), "planner_survival": plan(build_scenario(SPEC)).survival, "runs": {}}
    base = os.path.join(tier_dir(tier_name), "debug")
    for name, scale in (("failed_reward_scale_1.0", 1.0), ("fixed_reward_scale_0.01", 0.01)):
        d = os.path.join(base, name)
        torch.manual_seed(0)
        cfg = PPOConfig(total_steps=8 * 256 * updates, eval_every=10, save_fracs=(), seed=0, reward_scale=scale)
        train(FFPolicy(), cfg, SamplerSource(lambda rng: SPEC, 0), EnvConfig(), [SPEC], d)
        rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
        out["runs"][name] = {k: [float(r[k]) for r in rows] for k in
                             ("update", "ep_survival", "value_loss", "approx_kl", "explained_variance", "entropy", "grad_norm")}
    write_json(os.path.join(base, "debug.json"), out)
    return out
