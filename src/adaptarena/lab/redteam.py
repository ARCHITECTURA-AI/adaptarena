"""Red team search: random search then hill climbing over the bounded scenario space.

Each candidate is certified solvable by the reference planner, so every failure found is a *verified*
failure (a solvable scenario the policy died in), scored by ladder-normalised survival (lower = worse).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np

from ..core import params as P
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario
from .evaluate import evaluate, random_mean, survival_array
from .experiments import get_val_suite, load_model, load_tier, tier_dir, write_json

BASE = 400_000  # red-team layouts are disjoint from train, val, test and stress


def _i(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def sample_space(rng: np.random.Generator) -> ScenarioSpec:
    fmin, wmin = _i(rng, 20, 110), _i(rng, 20, 110)
    return ScenarioSpec(layout_id=BASE + _i(rng, 0, 10**6), timing_seed=4 * 10**9 + _i(rng, 0, 10**8),
                        weather_seed=4 * 10**9 + _i(rng, 0, 10**8), n_food=_i(rng, 1, 5), n_water=_i(rng, 1, 5),
                        wall_pct=_i(rng, 5, 30), food_delay_min=fmin, food_delay_max=fmin + _i(rng, 10, 30),
                        water_delay_min=wmin, water_delay_max=wmin + _i(rng, 10, 30),
                        move_cost_pct=_i(rng, 100, 300), cold_pct=_i(rng, 0, 60), cold_len=40)


def mutate(spec: ScenarioSpec, rng: np.random.Generator) -> ScenarioSpec:
    kind = _i(rng, 0, 7)
    if kind == 0:
        return spec.with_(n_food=int(np.clip(spec.n_food + rng.choice([-1, 1]), 1, 5)))
    if kind == 1:
        return spec.with_(n_water=int(np.clip(spec.n_water + rng.choice([-1, 1]), 1, 5)))
    if kind == 2:
        return spec.with_(wall_pct=int(np.clip(spec.wall_pct + rng.choice([-4, 4]), 5, 30)))
    if kind == 3:
        d = int(rng.choice([-15, 15]))
        return spec.with_(food_delay_min=max(20, spec.food_delay_min + d), food_delay_max=max(30, spec.food_delay_max + d))
    if kind == 4:
        d = int(rng.choice([-15, 15]))
        return spec.with_(water_delay_min=max(20, spec.water_delay_min + d), water_delay_max=max(30, spec.water_delay_max + d))
    if kind == 5:
        return spec.with_(move_cost_pct=int(np.clip(spec.move_cost_pct + rng.choice([-30, 30]), 100, 300)))
    if kind == 6:
        return spec.with_(cold_pct=int(np.clip(spec.cold_pct + rng.choice([-10, 10]), 0, 60)))
    return spec.with_(timing_seed=4 * 10**9 + _i(rng, 0, 10**8))


def _certified(spec: ScenarioSpec) -> bool:
    return plan(build_scenario(spec)).certified


def score_specs(policy, specs: List[ScenarioSpec]) -> np.ndarray:
    rand = random_mean(specs, 6)
    surv = survival_array(evaluate(policy, specs))
    denom = np.maximum(P.HORIZON - rand, 1.0)
    return (surv - rand) / denom, surv


def search(policy, budget: int, seed: int = 0, climbers: int = 3, climb_rounds: int = 6, neighbours: int = 12) -> dict:
    rng = np.random.default_rng([0x4ED7, seed])
    specs: List[ScenarioSpec] = []
    while len(specs) < budget:
        s = sample_space(rng)
        if _certified(s):
            specs.append(s)
    sc, surv = score_specs(policy, specs)
    order = np.argsort(sc)
    found = [(float(sc[i]), int(surv[i]), specs[i]) for i in order]
    best = [specs[i] for i in order[:climbers]]
    best_sc = [float(sc[i]) for i in order[:climbers]]
    n_eval = budget
    for ci in range(climbers):
        cur, cur_sc = best[ci], best_sc[ci]
        for _ in range(climb_rounds):
            cands = []
            while len(cands) < neighbours:
                m = mutate(cur, rng)
                if _certified(m):
                    cands.append(m)
            c_sc, c_surv = score_specs(policy, cands)
            n_eval += len(cands)
            j = int(np.argmin(c_sc))
            found += [(float(c_sc[k]), int(c_surv[k]), cands[k]) for k in range(len(cands))]
            if c_sc[j] < cur_sc:
                cur, cur_sc = cands[j], float(c_sc[j])
    found.sort(key=lambda x: x[0])
    uniq, seen = [], set()
    for s_, v, sp in found:
        if sp.sid not in seen:
            uniq.append({"score": s_, "survival": v, "spec": sp.to_json()})
            seen.add(sp.sid)
    return {"evaluated": n_eval, "worst": uniq[:25], "random_search_mean_score": float(np.mean(sc)),
            "random_search_worst_score": float(sc.min())}


def run(tier_name: str) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    budget = tier["redteam_budget"]
    out = {"budget": budget, "policies": {}}
    pols: Dict[str, object] = {"reactive_script": "reactive_script", "memory_script": "memory_script"}
    for arm in ("B", "D"):
        d = os.path.join(tdir, "headline", f"arm_{arm}", "seed_0")
        if os.path.exists(os.path.join(d, "model_best.pt")):
            pols[f"ppo_ff_arm{arm}_seed0"] = load_model(d, "ff")
    for name, pol in pols.items():
        print(f"red team: {name}", flush=True)
        out["policies"][name] = search(pol, budget)
    write_json(os.path.join(tdir, "redteam", "redteam.json"), out)
    return out
