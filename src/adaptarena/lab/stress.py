"""Stress curves: sweep intensity over ten levels for every rung of the ladder.

Three stressors, each with ten levels (0 = training-typical, 9 = hardest):
  water   fewer water sites and slower respawn
  cost    movement-cost multiplier 100% .. 325%
  food    fewer food sites and slower respawn

Rules fixed before the run (they are repeated in analysis_plan.yaml):
  * breaking point = lowest level where mean survival < 0.5 x the policy's familiar-condition survival
  * a level where > 90% of a policy's episodes reach the horizon is flagged saturated, never a win
  * scenarios where the planner finds no certificate are unsolvable and are never counted against a policy
Survival is drawn as Kaplan-Meier curves; episodes alive at the horizon are right-censored.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ..core import params as P
from ..core.distributions import sample_train
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario
from . import stats as S
from .evaluate import evaluate, survival_array
from .experiments import get_val_suite, load_tier, policy_zoo, tier_dir, write_json

LEVELS = 10
STRESSORS = ("water", "cost", "food")


def level_spec(base: ScenarioSpec, stressor: str, level: int) -> ScenarioSpec:
    if stressor == "water":
        return base.with_(n_water=max(1, 5 - (level + 1) // 2), water_delay_min=50 + 6 * level,
                          water_delay_max=65 + 6 * level, n_food=4)
    if stressor == "food":
        return base.with_(n_food=max(1, 5 - (level + 1) // 2), food_delay_min=50 + 6 * level,
                          food_delay_max=65 + 6 * level, n_water=4)
    if stressor == "cost":
        return base.with_(move_cost_pct=100 + 25 * level)
    raise ValueError(stressor)


def stress_specs(n: int, seed: int = 0) -> Dict[str, List[List[ScenarioSpec]]]:
    """specs[stressor][level] -> list of n scenarios. Bases use stress-only layout ids (300000+)."""
    rng = np.random.default_rng([0x57E5, seed])
    bases = []
    for _ in range(n):
        b = sample_train(rng)
        bases.append(b.with_(layout_id=300_000 + int(rng.integers(0, 99_999)), cold_pct=0, move_cost_pct=100,
                             wall_pct=17, timing_seed=3 * 10**9 + int(rng.integers(0, 10**8)),
                             weather_seed=3 * 10**9 + int(rng.integers(0, 10**8))))
    return {s: [[level_spec(b, s, L) for b in bases] for L in range(LEVELS)] for s in STRESSORS}


def run(tier_name: str) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    n = tier["stress_scenarios"]
    specs = stress_specs(n)
    zoo = policy_zoo(tdir, tier)
    familiar_specs = get_val_suite(tdir, tier, 0).specs["familiar"]
    out: dict = {"levels": LEVELS, "n_scenarios": n, "stressors": {}, "rules": {
        "breaking_point": "lowest level with mean survival < 0.5 x familiar-condition survival",
        "saturation": "> 90% of episodes reach the horizon", "unsolvable": "planner finds no certificate"}}
    fam = {}
    for label, entry in zoo.items():
        runs = [survival_array(r).mean() for r in
                [evaluate(p, familiar_specs) for p in entry["policies"]]]
        fam[label] = float(np.mean(runs))
    out["familiar_survival"] = fam
    for st in STRESSORS:
        # solvability is per scenario: certify each (level, scenario) once
        solv = np.array([[plan(build_scenario(s)).certified for s in specs[st][L]] for L in range(LEVELS)])
        res = {"solvable_fraction": solv.mean(1).tolist(), "rungs": {}}
        for label, entry in zoo.items():
            per_level, saturated, km = [], [], []
            for L in range(LEVELS):
                keep = [s for s, ok in zip(specs[st][L], solv[L]) if ok]
                if not keep:
                    per_level.append(None); saturated.append(None); km.append(None)
                    continue
                allsurv = []
                for r in run_zoo(entry, keep):
                    allsurv.append(survival_array(r))
                allsurv = np.concatenate(allsurv)
                per_level.append(float(allsurv.mean()))
                saturated.append(bool((allsurv >= P.HORIZON).mean() > 0.9))
                t, Sv = S.kaplan_meier(allsurv, allsurv < P.HORIZON, P.HORIZON)
                km.append({"t": t[::5].tolist(), "S": Sv[::5].tolist(),
                           "censored": S.censored_fraction(allsurv, P.HORIZON)})
            valid_levels = [L for L in range(LEVELS) if per_level[L] is not None]
            thresh = 0.5 * fam[label]
            bp = None
            for L in valid_levels:
                if per_level[L] < thresh:
                    bp = L
                    break
            res["rungs"][label] = {"rung": entry["rung"], "mean_survival": per_level, "saturated": saturated,
                                   "km": km, "breaking_point": bp, "threshold": thresh}
        # planner ceiling: the planner's survival is the horizon on solvable scenarios by construction
        res["rungs"]["planner"] = {"rung": 9, "mean_survival": [float(P.HORIZON) if solv[L].any() else None for L in range(LEVELS)],
                                   "saturated": [bool(solv[L].any()) for L in range(LEVELS)], "km": None,
                                   "breaking_point": None, "threshold": None}
        out["stressors"][st] = res
    write_json(os.path.join(tdir, "stress", "stress.json"), out)
    return out


def run_zoo(entry, specs):
    from .experiments import run_zoo_entry
    return run_zoo_entry(entry, specs)
