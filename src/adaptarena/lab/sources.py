"""Scenario sources for the four headline arms."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..core.distributions import sample_specialist, sample_train
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario
from collections import deque

from .plr import PLRBuffer, mutate_spec
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
        self.seen = set()          # buffer diagnostics only; they never feed back into sampling
        self.episodes = self.repeats = 0
        self.score_pairs = []      # (previous score, new score) of replayed scenarios: how noisy the score is

    def _note(self, spec: ScenarioSpec) -> None:
        self.episodes += 1
        if spec.sid in self.seen:
            self.repeats += 1
        self.seen.add(spec.sid)

    def _diagnostics(self) -> dict:
        d = {"episodes": self.episodes, "distinct_scenarios": len(self.seen),
             "repeat_fraction": self.repeats / max(self.episodes, 1)}
        if len(self.score_pairs) > 10:
            from scipy.stats import spearmanr
            a = np.array(self.score_pairs)
            d["score_rank_correlation"] = float(spearmanr(a[:, 0], a[:, 1])[0]) if a.std(0).min() > 0 else float("nan")
        return d

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
        self._note(spec)
        if self.kind == "regret":
            if origin == "fresh":
                extra = self._planner_survival(spec)
            score = float(max(0, extra - survival))
        else:
            score = float(pvl)
        if origin == "replay":
            if self.buf.specs[idx] is spec:  # slot may have been overwritten meanwhile
                self.score_pairs.append((float(self.buf.scores[idx]), score))
                self.buf.update(idx, score)
        else:
            self.buf.insert(spec, score, extra if self.kind == "regret" else None)
        if self.buf.episode_count % 500 == 0:
            self.snapshots.append(self.buf.snapshot())

    def stats(self) -> dict:
        return {"planner_calls": self.planner_calls, "planner_seconds": self.planner_seconds,
                "replays": self.buf.replays, "fresh": self.buf.fresh, "snapshots": self.snapshots, **self._diagnostics()}


class MutationSource(ReplaySource):
    """v2.1 arms E, E-prime and F. Same buffer and replay rule as arms C and D, plus two optional additions.

    mutate=True (E, E-prime): after a replayed scenario has been trained on, make one child (1 to 2 edits inside the
    training distribution), certify it with the planner, queue it for the next rollout, score it there and insert it
    only if its score beats the lowest score in the buffer. Uncertified children are counted as rejections.
    rollouts=4 (F): every new or replayed scenario is played four times and scored by the mean over the four, so a
    less noisy score is bought with rollouts that count against the step budget. Their transitions are all trained on.
    E and E-prime share every line; only `score_kind` differs.
    """

    def __init__(self, score_kind: str, seed: int, mutate: bool = True, rollouts: int = 1, **kw):
        super().__init__(score_kind, seed, **kw)
        self.mutate, self.rollouts = mutate, rollouts
        self.mrng = np.random.default_rng([0x4D55, seed])
        self.pending = deque()
        self.children = self.accepted = self.rejected_uncertified = self.rejected_by_buffer = 0

    # -- scenario choice
    def sample(self, env_i):
        if self.pending:
            return self.pending.popleft()
        if self.buf.want_replay():
            idx, spec, extra = self.buf.sample()
            origin = "replay"
        else:
            spec, idx, extra, origin = sample_train(self.rng), None, None, "fresh"
        if self.rollouts > 1:
            job = {"origin": origin, "idx": idx, "spec": spec, "extra": extra, "surv": []}
            for _ in range(self.rollouts - 1):
                self.pending.append((spec, ("repeat", job, spec, None)))
            return spec, ("repeat", job, spec, None)
        return spec, (origin, idx, spec, extra)

    def _certify(self, spec: ScenarioSpec):
        from ..core.planner import verify_plan
        t0 = time.time()
        res = plan(build_scenario(spec), max_nodes=self.planner_nodes)
        self.planner_seconds += time.time() - t0
        self.planner_calls += 1
        ok = bool(res.certified) and verify_plan(build_scenario(spec), res.actions) >= spec.horizon
        return ok, res.survival

    def _score(self, planner_surv, survival, pvl) -> float:
        return float(max(0, planner_surv - survival)) if self.kind == "regret" else float(pvl)

    def _snap(self) -> None:
        if self.buf.episode_count % 500 == 0:
            self.snapshots.append(self.buf.snapshot())

    # -- bookkeeping
    def episode_end(self, meta, survival: int, pvl: Optional[float]) -> None:
        origin = meta[0]
        if origin == "repeat":
            return self._repeat_end(meta[1], survival, pvl)
        _, idx, spec, extra = meta
        self.buf.note_episode()
        self._snap()
        self._note(spec)
        if origin == "child":
            score = self._score(extra, survival, pvl)
            if score > self.buf.min_score() or self.buf.size < self.buf.capacity:
                if self.buf.insert(spec, score, extra if self.kind == "regret" else None) is not None:
                    self.accepted += 1
                    return
            self.rejected_by_buffer += 1
            return
        if self.kind == "regret" and origin == "fresh":
            extra = self._planner_survival(spec)
        score = self._score(extra, survival, pvl)
        if origin == "replay":
            if self.buf.specs[idx] is spec:
                self.score_pairs.append((float(self.buf.scores[idx]), score))
                self.buf.update(idx, score)
            self._spawn_child(spec)
        else:
            self.buf.insert(spec, score, extra if self.kind == "regret" else None)

    def _repeat_end(self, job: dict, survival: int, pvl: Optional[float]) -> None:
        self.buf.note_episode()
        self._snap()
        spec = job["spec"]
        self._note(spec)
        job["surv"].append(survival)
        job.setdefault("pvl", []).append(pvl)
        if len(job["surv"]) < self.rollouts:
            return
        extra = job["extra"]
        if self.kind == "regret" and job["origin"] == "fresh":
            extra = self._planner_survival(spec)
        if self.kind == "regret":
            score = float(max(0.0, extra - float(np.mean(job["surv"]))))
        else:
            score = float(np.mean([p for p in job["pvl"] if p is not None]))
        if job["origin"] == "replay":
            if self.buf.specs[job["idx"]] is spec:
                self.score_pairs.append((float(self.buf.scores[job["idx"]]), score))
                self.buf.update(job["idx"], score)
        else:
            self.buf.insert(spec, score, extra if self.kind == "regret" else None)

    def _spawn_child(self, spec: ScenarioSpec) -> None:
        if not self.mutate:
            return
        child = mutate_spec(spec, self.mrng)
        self.children += 1
        ok, surv = self._certify(child)
        if not ok:
            self.rejected_uncertified += 1
            return
        self.pending.append((child, ("child", None, child, surv)))

    def stats(self) -> dict:
        d = super().stats()
        d.update({"children": self.children, "children_accepted": self.accepted,
                  "children_rejected_uncertified": self.rejected_uncertified,
                  "children_rejected_by_buffer": self.rejected_by_buffer,
                  "uncertified_rate": self.rejected_uncertified / max(self.children, 1),
                  "acceptance_rate": self.accepted / max(self.children - self.rejected_uncertified, 1)})
        return d


def make_source(arm: str, seed: int) -> ScenarioSource:
    if arm == "A":
        return SamplerSource(sample_specialist, seed)
    if arm == "B":
        return SamplerSource(sample_train, seed)
    if arm == "C":
        return ReplaySource("value_loss", seed)
    if arm == "D":
        return ReplaySource("regret", seed)
    if arm == "E":
        return MutationSource("regret", seed, mutate=True)
    if arm == "Eprime":
        return MutationSource("value_loss", seed, mutate=True)
    if arm == "F":
        return MutationSource("regret", seed, mutate=False, rollouts=4)
    raise ValueError(arm)
