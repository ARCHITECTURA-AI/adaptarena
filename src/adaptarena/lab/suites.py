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
                 record: bool = False, env_factory=None) -> dict:
        """Run `policy` on every condition. Returns per-condition raw survival, causes and normalised scores."""
        out = {}
        for c in self.conditions:
            res = evaluate(policy, self.specs[c], list(range(len(self.specs[c]))), env_config,
                           deterministic=deterministic, record_actions=record, env_factory=env_factory)
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
                           generate: bool = False, name: str = "suite_test.json",
                           conditions: Sequence[str] = CONDITIONS, plan_path: Optional[str] = None,
                           hash_path: Optional[str] = None, seed: int = 0,
                           exclude_suites: Sequence[str] = ()) -> Suite:
    """The only sanctioned way to read (or first generate) a locked test suite.

    `plan_path` selects which locked plan guards the suite (v2: analysis_plan.yaml, v2.1: analysis_plan_v2_1.yaml).
    The harness refuses to open the suite unless that plan still matches its committed hash, and logs every access
    with the purpose, plan file, plan hash and commit.
    """
    plan_path = plan_path or planmod.PLAN
    hash_path = hash_path or planmod.hash_path_for(plan_path)
    info = planmod.verify_plan(plan_path, hash_path)
    planmod.check_frozen(plan_path)
    info["plan_file"] = os.path.relpath(plan_path, planmod.REPO)
    info["suite"] = name
    planmod.log_access(os.path.join(tier_dir, "test_access_log.jsonl"), purpose, info)
    path = os.path.join(tier_dir, name)
    if not os.path.exists(path):
        if not generate or n_per_cond is None:
            raise planmod.PlanError("locked test suite has not been generated for this tier")
        build_suite_file_excluding("test", n_per_cond, path, conditions=conditions, seed=seed,
                                   exclude=[os.path.join(tier_dir, e) for e in exclude_suites])
    return Suite.load(path)


def build_suite_file_excluding(split: str, n_per_cond: int, path: str, conditions: Sequence[str] = CONDITIONS,
                               seed: int = 0, exclude: Sequence[str] = (), log=print) -> dict:
    """Like `build_suite_file`, with a new generation seed and no scenario or layout shared with `exclude` suites.

    Reimplemented here (not in core/distributions.py) because the core's hashes are frozen. Certification is the same:
    only scenarios the planner certifies are kept, and the rejection counts are stored.
    """
    import numpy as np
    from ..core.distributions import SEED_SPAN, sample_condition
    from ..core.planner import plan, verify_plan
    from ..core.scenario import build_scenario
    banned_sid, banned_layout = set(), set()
    for ex in exclude:
        if os.path.exists(ex):
            for c in json.load(open(ex))["conditions"].values():
                for sp in c["specs"]:
                    spec = ScenarioSpec.from_json(sp)
                    banned_sid.add(spec.sid)
                    banned_layout.add(spec.layout_id)
    out = {"split": split, "n_per_cond": n_per_cond, "seed": seed, "excluded_suites": [os.path.basename(e) for e in exclude],
           "conditions": {}}
    sid = {"train": 0, "val": 1, "test": 2}[split]
    for cond in conditions:
        cid = (CONDITIONS + MEMORY_CONDITIONS).index(cond)
        rng = np.random.default_rng([0x5017E, sid, cid, seed])
        specs, tried, rejected, collided = [], 0, 0, 0
        while len(specs) < n_per_cond:
            spec = sample_condition(cond, rng, split)
            tried += 1
            fixed_layout = cond == "familiar"  # the specialist's own layout is part of the condition; exclude by id only
            if spec.sid in banned_sid or (not fixed_layout and spec.layout_id in banned_layout):
                collided += 1
                continue
            sc = build_scenario(spec)
            res = plan(sc, max_nodes=4000)
            if res.certified and verify_plan(sc, res.actions) >= spec.horizon:
                specs.append(spec)
                banned_sid.add(spec.sid)
                if not fixed_layout:
                    banned_layout.add(spec.layout_id)
            else:
                rejected += 1
        planner = planner_survival(specs)
        rand = random_mean(specs, 10)
        refs = {name: survival_array(evaluate(name, specs, list(range(len(specs))))).tolist() for name in LADDER_SCRIPTS}
        out["conditions"][cond] = {
            "specs": [s.to_json() for s in specs], "planner": planner.tolist(), "random_mean": rand.tolist(), "scripts": refs,
            "in_distribution": IN_DISTRIBUTION.get(cond, False),
            "generation": {"condition": cond, "split": split, "n": n_per_cond, "tried": tried, "rejected_uncertified": rejected,
                           "excluded_collisions": collided}}
        log(f"  {split}/{cond}: n={len(specs)} tried={tried} rejected={rejected} collided={collided}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    return out
