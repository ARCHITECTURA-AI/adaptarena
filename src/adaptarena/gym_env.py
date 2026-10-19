"""Gymnasium wrapper around the reference environment."""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .core import params as P
from .core.distributions import sample_train
from .core.env import EnvConfig, SurvivalEnv
from .core.scenario import ScenarioSpec


class AdaptArenaSurvivalEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, scenario: Optional[ScenarioSpec] = None, config: Optional[EnvConfig] = None,
                 render_mode: Optional[str] = None):
        self._fixed = scenario
        self.core = SurvivalEnv(config)
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (P.N_CHANNELS, P.VIEW, P.VIEW), np.float32),
            "stats": spaces.Box(0.0, 1.0, (P.STATS_DIM,), np.float32),
        })
        self.action_space = spaces.Discrete(P.N_ACTIONS)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        spec = (options or {}).get("scenario", self._fixed)
        if spec is None:
            spec = sample_train(self.np_random)
        obs = self.core.reset(spec)
        return obs, {"scenario_id": self.core.scenario_id(), "state_hash": self.core.state_hash()}

    def step(self, action):
        obs, r, term, trunc, info = self.core.step(int(action))
        info["state_hash"] = self.core.state_hash()
        return obs, float(r), bool(term), bool(trunc), info

    def render(self):
        if self.render_mode != "ansi":
            return None
        sc, e = self.core.scenario, self.core
        rows = []
        for y in range(P.GRID):
            row = ""
            for x in range(P.GRID):
                ch = "#" if sc.terrain[y, x] else "."
                for p in sc.food_pos:
                    if (y, x) == tuple(p):
                        ch = "F"
                for p in sc.water_positions(e.t):
                    if (y, x) == tuple(p):
                        ch = "W"
                if (y, x) == (e.r, e.c):
                    ch = "@"
                row += ch
            rows.append(row)
        return "\n".join(rows)


def register_envs() -> None:
    if P.ENV_ID not in gym.registry:
        gym.register(id=P.ENV_ID, entry_point="adaptarena.gym_env:AdaptArenaSurvivalEnv")


register_envs()
