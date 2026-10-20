"""Scenario sources for the four headline arms."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..core.distributions import sample_specialist, sample_train
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario
from .plr import PLRBuffer
from .ppo import SamplerSource, ScenarioSource


class ReplaySource(ScenarioSource):
    """Arms C (score = positive value loss) and D (score = planner regret).

    Both arms share this class, so the buffer code is identical and only `score_kind`
    differs. Every transition is used for the gradient update, whatever its origin.
    """

    def __init__(self, score_kind: str, seed: int, capacity: int = 4096, replay_prob: float = 0.5,
                 planner_nodes: int = 4000):
        assert score_kind in ("regret", "value_loss")
        self.kind = score_kind
        self.needs_pvl = score_kind == "value_loss"
        self.buf = PLRBuffer(capacity=capacity, replay_prob=replay_prob, seed=seed)
        self.rng = np.random.default_rng([0x50A2, seed])
        self.planner_nodes = planner_nodes
        self.planner_calls = 0
        self.planner_seconds = 0.0
        self.snapshots = []

    def sample(self, env_i):
        if self.buf.want_replay():
            idx, spec, extra = self.buf.sample()
            return spec, ("replay", idx, spec, extra)
        spec = sample_train(self.rng)
        return spec, ("fresh", None, spec, None)

    def _planner_survival(self, spec: ScenarioSpec) -> int:
        t0 = time.time()
        res = plan(build_scenario(spec), max_nodes=self.planner_nodes)
        self.planner_seconds += time.time() - t0
        self.planner_calls += 1
        return res.survival

    def episode_end(self, meta, survival: int, pvl: Optional[float]) -> None:
        origin, idx, spec, extra = meta
        self.buf.note_episode()
        if self.kind == "regret":
            if origin == "fresh":
                extra = self._planner_survival(spec)
            score = float(max(0, extra - survival))
        else:
            score = float(pvl)
        if origin == "replay":
            if self.buf.specs[idx] is spec:  # slot may have been overwritten meanwhile
                self.buf.update(idx, score)
        else:
            self.buf.insert(spec, score, extra if self.kind == "regret" else None)
        if self.buf.episode_count % 500 == 0:
            self.snapshots.append(self.buf.snapshot())

    def stats(self) -> dict:
        return {"planner_calls": self.planner_calls, "planner_seconds": self.planner_seconds,
                "replays": self.buf.replays, "fresh": self.buf.fresh, "snapshots": self.snapshots}


def make_source(arm: str, seed: int) -> ScenarioSource:
    if arm == "A":
        return SamplerSource(sample_specialist, seed)
    if arm == "B":
        return SamplerSource(sample_train, seed)
    if arm == "C":
        return ReplaySource("value_loss", seed)
    if arm == "D":
        return ReplaySource("regret", seed)
    raise ValueError(arm)
