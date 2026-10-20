"""Prioritized Level Replay buffer, shared by arms C and D. Only the score differs.

Rank-based prioritisation with a staleness term (Jiang et al. 2021). The buffer stores
scenario specs; sampling a replay returns the stored spec and its slot index.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.scenario import ScenarioSpec


class PLRBuffer:
    def __init__(self, capacity: int = 4096, replay_prob: float = 0.5, temperature: float = 0.3,
                 staleness: float = 0.3, min_fill: int = 256, seed: int = 0):
        self.capacity, self.replay_prob, self.temperature = capacity, replay_prob, temperature
        self.staleness, self.min_fill = staleness, min_fill
        self.specs: list = [None] * capacity
        self.scores = np.full(capacity, -np.inf)
        self.last_seen = np.zeros(capacity)
        self.extra: list = [None] * capacity  # e.g. cached planner survival
        self.size = 0
        self.episode_count = 0
        self.rng = np.random.default_rng([0x9112, seed])
        self.replays = 0
        self.fresh = 0
        self.history_scores = []  # (episode_count, mean score) for the companion panel

    def want_replay(self) -> bool:
        return self.size >= self.min_fill and self.rng.random() < self.replay_prob

    def _weights(self) -> np.ndarray:
        s = self.scores[:self.size]
        ranks = np.empty(self.size)
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, self.size + 1)
        p = (1.0 / ranks) ** (1.0 / self.temperature)
        p /= p.sum()
        stale = self.episode_count - self.last_seen[:self.size]
        total = stale.sum()
        ps = stale / total if total > 0 else np.full(self.size, 1.0 / self.size)
        w = (1 - self.staleness) * p + self.staleness * ps
        return w / w.sum()

    def sample(self):
        idx = int(self.rng.choice(self.size, p=self._weights()))
        self.last_seen[idx] = self.episode_count
        self.replays += 1
        return idx, self.specs[idx], self.extra[idx]

    def note_episode(self) -> None:
        self.episode_count += 1

    def update(self, idx: int, score: float) -> None:
        self.scores[idx] = score

    def insert(self, spec: ScenarioSpec, score: float, extra=None) -> Optional[int]:
        self.fresh += 1
        if self.size < self.capacity:
            i = self.size
            self.size += 1
        else:
            i = int(np.argmin(self.scores))
            if score <= self.scores[i]:
                return None
        self.specs[i], self.scores[i], self.extra[i] = spec, score, extra
        self.last_seen[i] = self.episode_count
        return i

    def snapshot(self) -> dict:
        """Spec parameter means of the current high-priority half, for the companion panel."""
        if self.size == 0:
            return {}
        order = np.argsort(-self.scores[:self.size])[: max(1, self.size // 4)]
        sp = [self.specs[i] for i in order]
        allsp = self.specs[:self.size]
        def m(f, S):
            return float(np.mean([f(s) for s in S]))
        feats = {"n_food": lambda s: s.n_food, "n_water": lambda s: s.n_water, "wall_pct": lambda s: s.wall_pct,
                 "move_cost_pct": lambda s: s.move_cost_pct, "cold_pct": lambda s: s.cold_pct,
                 "delay": lambda s: (s.food_delay_min + s.water_delay_min) / 2}
        return {"size": self.size, "episodes": self.episode_count,
                "top_quartile": {k: m(f, sp) for k, f in feats.items()},
                "all": {k: m(f, allsp) for k, f in feats.items()},
                "mean_score": float(np.mean(self.scores[:self.size]))}
