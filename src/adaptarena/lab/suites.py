"""Certified scenario suites with ladder reference scores, and the locked test suite gate."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.distributions import CONDITIONS, IN_DISTRIBUTION, MEMORY_CONDITIONS, build_suite
from ..core.env import EnvConfig
from ..core.scenario import ScenarioSpec
from . import plan as planmod
from .evaluate import evaluate, ladder_normalise, planner_survival, random_mean, survival_array

LADDER_SCRIPTS = ("reactive_script", "memory_script")


def build_suite_file(split: str, n_per_cond: int, path: str, conditions: Sequence[str] = CONDITIONS,
                     seed: int = 0, log=print) -> dict:
    out = {"split": split, "n_per_cond": n_per_cond, "seed": seed, "conditions": {}}
    for cond in conditions:
        specs, meta = build_suite(split, cond, n_per_cond, seed=seed)
        planner = planner_survival(specs)
        rand = random_mean(specs, 10)
        refs = {}
        for name in LADDER_SCRIPTS:
            refs[name] = survival_array(evaluate(name, specs, list(range(len(specs))))).tolist()
        out["conditions"][cond] = {
            "specs": [s.to_json() for s in specs], "planner": planner.tolist(), "random_mean": rand.tolist(),
            "scripts": refs, "in_distribution": IN_DISTRIBUTION.get(cond, False), "generation": meta}
        log(f"  {split}/{cond}: n={len(specs)} tried={meta['tried']} rejected={meta['rejected_uncertified']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out


class Suite:
    def __init__(self, data: dict):
        self.data = data
        self.conditions = list(data["conditions"])
        self.specs: Dict[str, List[ScenarioSpec]] = {
            c: [ScenarioSpec.from_json(s) for s in d["specs"]] for c, d in data["conditions"].items()}
        self.planner = {c: np.array(d["planner"]) for c, d in data["conditions"].items()}
        self.rand = {c: np.array(d["random_mean"]) for c, d in data["conditions"].items()}
        self.scripts = {c: {k: np.array(v) for k, v in d["scripts"].items()} for c, d in data["conditions"].items()}

    @staticmethod
    def load(path: str) -> "Suite":
        with open(path) as f:
            return Suite(json.load(f))

    def all_specs(self):
        return [s for c in self.conditions for s in self.specs[c]]

    def evaluate(self, policy, env_config: Optional[EnvConfig] = None, deterministic: bool = True,
                 record: bool = False) -> dict:
        """Run `policy` on every condition. Returns per-condition raw survival, causes and normalised scores."""
        out = {}
        for c in self.conditions:
            res = evaluate(policy, self.specs[c], list(range(len(self.specs[c]))), env_config,
                           deterministic=deterministic, record_actions=record)
            surv = survival_array(res)
            score, valid = ladder_normalise(surv, self.rand[c], self.planner[c])
            out[c] = {"survival": surv, "score": score, "valid": valid, "results": res,
                      "causes": [r.cause for r in res]}
        return out

    def script_scores(self, name: str) -> dict:
        out = {}
        for c in self.conditions:
            surv = self.scripts[c][name]
            score, valid = ladder_normalise(surv, self.rand[c], self.planner[c])
            out[c] = {"survival": surv, "score": score, "valid": valid}
        return out

    def random_scores(self) -> dict:
        out = {}
        for c in self.conditions:
            out[c] = {"survival": self.rand[c], "score": np.zeros_like(self.rand[c]), "valid": np.ones(len(self.rand[c]), bool)}
        return out


def per_seed_score(cond_result: dict) -> float:
    """Mean ladder-normalised score of one trained seed in one condition (invalid scenarios excluded)."""
    v = cond_result["valid"]
    return float(np.mean(cond_result["score"][v])) if v.any() else float("nan")


def open_locked_test_suite(tier_dir: str, purpose: str, n_per_cond: Optional[int] = None,
                           generate: bool = False) -> Suite:
    """The only sanctioned way to read (or first generate) the locked test suite."""
    info = planmod.verify_plan()
    planmod.log_access(os.path.join(tier_dir, "test_access_log.jsonl"), purpose, info)
    path = os.path.join(tier_dir, "suite_test.json")
    if not os.path.exists(path):
        if not generate or n_per_cond is None:
            raise planmod.PlanError("locked test suite has not been generated for this tier")
        build_suite_file("test", n_per_cond, path)
    return Suite.load(path)
