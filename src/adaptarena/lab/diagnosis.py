"""Diagnosis of why a curriculum did (not) help, computed from the headline runs and fresh rollouts.

Descriptive and exploratory: nothing here is part of the pre-declared comparisons. Quantities:
  * replay fraction and distinct fresh scenarios per arm (replay halves scenario diversity at equal budget)
  * what the replay buffers concentrated on (top-priority quartile vs buffer mean, per scenario parameter)
  * noise ceiling: rank correlation of the same scenario's survival across two independent sampled rollouts
    (the training-time regret signal comes from sampled actions)
  * predictability: cross-validated R^2 of survival on the scenario parameters
  * planner solvability of the training distribution
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr

from ..core.distributions import sample_train
from ..core.planner import plan
from ..core.scenario import build_scenario
from .evaluate import evaluate, survival_array
from .experiments import ARMS, load_model, load_tier, tier_dir, write_json

FEATS = ["n_water", "n_food", "wall_pct", "move_cost_pct", "cold_pct", "delay"]


def run(tier_name: str, n_scen: int = 400) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    out: dict = {"arms": {}}
    for arm in ("B", "C", "D"):
        rows = {"replay_fraction": [], "fresh": [], "replays": [], "ratios": [], "train_survival": []}
        for s in range(tier["seeds"]):
            d = os.path.join(tdir, "headline", f"arm_{arm}", f"seed_{s}")
            log = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
            vals = [float(r["ep_survival"]) for r in log[-40:] if r["ep_survival"] not in ("nan", "")]
            rows["train_survival"].append(float(np.nanmean(vals)) if vals else float("nan"))
            e = json.load(open(os.path.join(d, "train_result.json")))["extra"]
            if e.get("replays") is not None:
                rows["replays"].append(e["replays"])
                rows["fresh"].append(e["fresh"])
                snaps = [x for x in e.get("snapshots", []) if x]
                if snaps:
                    sn = snaps[-1]
                    rows["ratios"].append([sn["top_quartile"][f] / max(sn["all"][f], 1e-9) for f in FEATS])
        a = {"train_time_episode_survival": float(np.nanmean(rows["train_survival"]))}
        if rows["replays"]:
            a.update(replays=float(np.mean(rows["replays"])), fresh=float(np.mean(rows["fresh"])),
                     replay_fraction=float(np.mean(rows["replays"]) / (np.mean(rows["replays"]) + np.mean(rows["fresh"]))),
                     top_quartile_over_mean=dict(zip(FEATS, np.mean(rows["ratios"], 0).tolist())) if rows["ratios"] else None)
        out["arms"][arm] = a
    rng = np.random.default_rng(123)
    specs = [sample_train(rng) for _ in range(n_scen)]
    out["training_distribution_certified_fraction"] = float(np.mean([plan(build_scenario(s)).certified for s in specs]))
    corr, surv_all = [], []
    for s in range(min(3, tier["seeds"])):
        m = load_model(os.path.join(tdir, "headline", "arm_B", f"seed_{s}"), "ff")
        a = survival_array(evaluate(m, specs, list(range(1000, 1000 + n_scen)), deterministic=False))
        b = survival_array(evaluate(m, specs, list(range(2000, 2000 + n_scen)), deterministic=False))
        corr.append(float(spearmanr(a, b).statistic))
        surv_all.append((a + b) / 2)
    out["noise_ceiling_spearman_sampled_rollouts"] = corr
    X = np.array([[sp.n_water, sp.n_food, sp.wall_pct, sp.move_cost_pct, sp.cold_pct, (sp.food_delay_min + sp.water_delay_min) / 2]
                  for sp in specs], dtype=float)
    y = np.mean(surv_all, 0)
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    out["cv_r2_survival_from_parameters"] = float(np.mean(cross_val_score(GradientBoostingRegressor(random_state=0), X, y, cv=5, scoring="r2")))
    write_json(os.path.join(tdir, "headline", "diagnosis.json"), out)
    return out
