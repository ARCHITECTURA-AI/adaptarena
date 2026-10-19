"""Training distribution, evaluation conditions and certified suite generation.

Seed ranges keep training, validation and the locked test suite disjoint:
  training   layouts 0..9999          timing/weather seeds < 10**9
  test       layouts 100000..         timing/weather seeds 2*10**9 ..
  validation layouts 200000..         timing/weather seeds 1*10**9 ..
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from .planner import plan, verify_plan
from .scenario import ScenarioSpec, build_scenario

TRAIN_LAYOUTS = 10_000
SPECIALIST_LAYOUT = 0
SPLIT_BASE = {
    "train": dict(layout=0, seed=0),
    "val": dict(layout=200_000, seed=10**9),
    "test": dict(layout=100_000, seed=2 * 10**9),
}
SEED_SPAN = 10**9 - 1

CONDITIONS = ("familiar", "unseen_layout", "scarce_food", "scarce_water", "higher_move_cost",
              "cold_snap", "combined_shift")
# Inside the (B, C, D) training distribution? Familiar and unseen-layout draw from it.
IN_DISTRIBUTION = {"familiar": True, "unseen_layout": True, "scarce_food": False, "scarce_water": False,
                   "higher_move_cost": False, "cold_snap": False, "combined_shift": False}
# Memory-track extras (evaluated separately, not part of the headline suite)
MEMORY_CONDITIONS = ("resource_relocation",)


# Difficulty constants. Calibrated on validation seeds with the scripted baselines and the
# planner only, then frozen before any learned policy was run (see docs/calibration.md).
TRAIN_FOOD = (3, 5)
TRAIN_WATER = (3, 5)
TRAIN_WALL = (12, 22)
TRAIN_DELAY_MIN = (50, 80)
TRAIN_DELAY_SPAN = (10, 20)
TRAIN_COST = (100, 130)
TRAIN_COLD = (0, 15)
SCARCE = (2, 2)
HIGH_COST = (180, 260)
COLD_SNAP = (40, 55)
COMBINED = dict(n=2, cost=(160, 200), cold=(35, 45))
FAMILIAR = dict(n_food=4, n_water=4, wall_pct=17, delay_min=60, delay_span=15, cost=100)


def _i(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def sample_train(rng: np.random.Generator) -> ScenarioSpec:
    fmin = _i(rng, *TRAIN_DELAY_MIN)
    wmin = _i(rng, *TRAIN_DELAY_MIN)
    return ScenarioSpec(
        layout_id=_i(rng, 0, TRAIN_LAYOUTS - 1), timing_seed=_i(rng, 0, SEED_SPAN),
        weather_seed=_i(rng, 0, SEED_SPAN), n_food=_i(rng, *TRAIN_FOOD), n_water=_i(rng, *TRAIN_WATER),
        wall_pct=_i(rng, *TRAIN_WALL), food_delay_min=fmin, food_delay_max=fmin + _i(rng, *TRAIN_DELAY_SPAN),
        water_delay_min=wmin, water_delay_max=wmin + _i(rng, *TRAIN_DELAY_SPAN),
        move_cost_pct=_i(rng, *TRAIN_COST), cold_pct=_i(rng, *TRAIN_COLD), cold_len=40)


def sample_specialist(rng: np.random.Generator) -> ScenarioSpec:
    """One fixed layout, temperate weather, default parameters; only timing seeds vary."""
    f = FAMILIAR
    return ScenarioSpec(layout_id=SPECIALIST_LAYOUT, timing_seed=_i(rng, 0, SEED_SPAN),
                        weather_seed=_i(rng, 0, SEED_SPAN), n_food=f["n_food"], n_water=f["n_water"],
                        wall_pct=f["wall_pct"], food_delay_min=f["delay_min"],
                        food_delay_max=f["delay_min"] + f["delay_span"], water_delay_min=f["delay_min"],
                        water_delay_max=f["delay_min"] + f["delay_span"], move_cost_pct=f["cost"],
                        cold_pct=0)


def sample_condition(cond: str, rng: np.random.Generator, split: str) -> ScenarioSpec:
    base = SPLIT_BASE[split]
    seeds = dict(timing_seed=base["seed"] + _i(rng, 0, SEED_SPAN // 2),
                 weather_seed=base["seed"] + _i(rng, 0, SEED_SPAN // 2))
    if cond == "familiar":
        s = sample_specialist(rng)
        return s.with_(**seeds)
    s = sample_train(rng).with_(layout_id=base["layout"] + _i(rng, 0, 99_999), **seeds)
    if cond == "unseen_layout":
        return s
    if cond == "scarce_food":
        return s.with_(n_food=_i(rng, *SCARCE))
    if cond == "scarce_water":
        return s.with_(n_water=_i(rng, *SCARCE))
    if cond == "higher_move_cost":
        return s.with_(move_cost_pct=_i(rng, *HIGH_COST))
    if cond == "cold_snap":
        return s.with_(cold_pct=_i(rng, *COLD_SNAP))
    if cond == "combined_shift":
        return s.with_(n_food=COMBINED["n"], n_water=COMBINED["n"], move_cost_pct=_i(rng, *COMBINED["cost"]),
                       cold_pct=_i(rng, *COMBINED["cold"]))
    if cond == "resource_relocation":
        return s.with_(interventions=(("reloc", _i(rng, 120, 260), 0),))
    raise ValueError(cond)


def build_suite(split: str, cond: str, n: int, seed: int = 0, max_nodes: int = 4000) -> Tuple[List[ScenarioSpec], dict]:
    """n scenarios for a condition, each certified solvable by the reference planner.

    Rejection is by the planner only, and the counts are returned so a report can state them.
    """
    cid = (CONDITIONS + MEMORY_CONDITIONS).index(cond)
    sid = {"train": 0, "val": 1, "test": 2}[split]
    rng = np.random.default_rng([0x5017E, sid, cid, seed])
    specs, tried, rejected = [], 0, 0
    while len(specs) < n:
        spec = sample_condition(cond, rng, split)
        tried += 1
        res = plan(build_scenario(spec), max_nodes=max_nodes)
        if res.certified and verify_plan(build_scenario(spec), res.actions) >= spec.horizon:
            specs.append(spec)
        else:
            rejected += 1
    return specs, {"condition": cond, "split": split, "n": n, "tried": tried, "rejected_uncertified": rejected}
