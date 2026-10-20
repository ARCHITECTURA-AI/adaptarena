"""Prioritized Level Replay buffer, shared by arms C and D. Only the score differs.

Rank-based prioritisation with a staleness term (Jiang et al. 2021). The buffer stores
scenario specs; sampling a replay returns the stored spec and its slot index.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import distributions as D
from ..core.scenario import ScenarioSpec

# ---- mutation (v2.1 arms E, E-prime). Bounds are the arm B training distribution, read from core/distributions.py.
# Seeds and layouts are re-rolled inside the training split (layouts 0..TRAIN_LAYOUTS-1, seeds from the training
# base), so a child can never land on a validation or test layout.
EDITS = ("sites", "wall", "delay", "cost", "cold", "seed", "layout")
EDIT_P = np.array([2, 2, 2, 2, 2, 2, 0.5])  # relative weights; layout re-rolls are rare
EDIT_P = EDIT_P / EDIT_P.sum()


def _clip(v: int, lo_hi) -> int:
    return int(min(max(v, lo_hi[0]), lo_hi[1]))


def mutate_spec(spec: ScenarioSpec, rng: np.random.Generator) -> ScenarioSpec:
    """One child with 1 to 2 edits (pure and seeded: same spec and rng state, same child).

    sites: n_food or n_water +-1. wall: wall_pct +-3. delay: food or water respawn delay window shifted by +-5
    (min and max together, so the span stays in its training range). cost: move_cost_pct +-10. cold: cold_pct +-3.
    seed: re-roll the timing or the weather seed. layout: re-roll the layout (low probability).
    Every result is clipped to the training distribution. An edit that clipping turns into a no-op is retried.
    """
    for _ in range(8):
        child = spec
        for kind in rng.choice(len(EDITS), size=int(rng.integers(1, 3)), replace=False, p=EDIT_P):
            kind = EDITS[kind]
            sign = int(rng.choice((-1, 1)))
            if kind == "sites":
                f = "n_food" if rng.random() < 0.5 else "n_water"
                lim = D.TRAIN_FOOD if f == "n_food" else D.TRAIN_WATER
                child = child.with_(**{f: _clip(getattr(child, f) + sign, lim)})
            elif kind == "wall":
                child = child.with_(wall_pct=_clip(child.wall_pct + 3 * sign, D.TRAIN_WALL))
            elif kind == "delay":
                w = "food" if rng.random() < 0.5 else "water"
                lo, hi = getattr(child, f"{w}_delay_min"), getattr(child, f"{w}_delay_max")
                new_lo = _clip(lo + 5 * sign, D.TRAIN_DELAY_MIN)
                child = child.with_(**{f"{w}_delay_min": new_lo, f"{w}_delay_max": new_lo + (hi - lo)})
            elif kind == "cost":
                child = child.with_(move_cost_pct=_clip(child.move_cost_pct + 10 * sign, D.TRAIN_COST))
            elif kind == "cold":
                child = child.with_(cold_pct=_clip(child.cold_pct + 3 * sign, D.TRAIN_COLD))
            elif kind == "seed":
                f = "timing_seed" if rng.random() < 0.5 else "weather_seed"
                child = child.with_(**{f: int(rng.integers(0, D.SEED_SPAN + 1))})
            else:
                child = child.with_(layout_id=int(rng.integers(0, D.TRAIN_LAYOUTS)))
        if child.sid != spec.sid:
            return child
    return child


def in_training_distribution(spec: ScenarioSpec) -> bool:
    """True when every field the training sampler draws lies in its training range (used to test mutation)."""
    span = lambda a, b: b - a
    return (D.TRAIN_FOOD[0] <= spec.n_food <= D.TRAIN_FOOD[1] and D.TRAIN_WATER[0] <= spec.n_water <= D.TRAIN_WATER[1]
            and D.TRAIN_WALL[0] <= spec.wall_pct <= D.TRAIN_WALL[1]
            and D.TRAIN_DELAY_MIN[0] <= spec.food_delay_min <= D.TRAIN_DELAY_MIN[1]
            and D.TRAIN_DELAY_MIN[0] <= spec.water_delay_min <= D.TRAIN_DELAY_MIN[1]
            and D.TRAIN_DELAY_SPAN[0] <= span(spec.food_delay_min, spec.food_delay_max) <= D.TRAIN_DELAY_SPAN[1]
            and D.TRAIN_DELAY_SPAN[0] <= span(spec.water_delay_min, spec.water_delay_max) <= D.TRAIN_DELAY_SPAN[1]
            and D.TRAIN_COST[0] <= spec.move_cost_pct <= D.TRAIN_COST[1]
            and D.TRAIN_COLD[0] <= spec.cold_pct <= D.TRAIN_COLD[1]
            and 0 <= spec.layout_id < D.TRAIN_LAYOUTS and 0 <= spec.timing_seed <= D.SEED_SPAN
            and 0 <= spec.weather_seed <= D.SEED_SPAN and spec.interventions == () and spec.cold_len == 40)


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

    def min_score(self) -> float:
        return float(self.scores[:self.size].min()) if self.size else -np.inf

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
