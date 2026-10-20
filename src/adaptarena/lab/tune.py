"""One small hyper-parameter sweep per learned method: 3 learning rates x 3 entropy coefficients x 3 seeds.

Scored on validation only (best validation survival of each run), same grid and seed count for every
method, and no further tuning afterwards. The winning pair is written to `tuned.json` and used by the
headline, ablation and memory runs of that method.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .experiments import get_val_suite, load_tier, run_jobs, tier_dir, train_job, write_json

LRS = (1e-4, 3e-4, 1e-3)
ENTS = (0.003, 0.01, 0.03)


def jobs(tier_name: str, model: str) -> List[dict]:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    vpath = os.path.join(tdir, "suite_val.json")
    out = []
    for lr in LRS:
        for ent in ENTS:
            for s in range(tier["tune_seeds"]):
                out.append(dict(out_dir=os.path.join(tdir, "tune", model, f"lr{lr}_ent{ent}", f"seed_{s}"),
                                arm="B", model=model, seed=100 + s, steps=tier["tune_steps"],
                                ppo={"lr": lr, "ent_coef": ent}, val_suite=vpath))
    return out


def run(tier_name: str, models=("ff", "lstm"), workers: int = 4) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    get_val_suite(tdir, tier, 0)
    result: Dict[str, dict] = {}
    for model in models:
        run_jobs(train_job, jobs(tier_name, model), workers)
        grid = {}
        for lr in LRS:
            for ent in ENTS:
                vals = []
                for s in range(tier["tune_seeds"]):
                    d = os.path.join(tdir, "tune", model, f"lr{lr}_ent{ent}", f"seed_{s}")
                    vals.append(json.load(open(os.path.join(d, "done.json")))["best_val"])
                grid[f"lr={lr},ent={ent}"] = float(np.mean(vals))
        best = max(grid, key=grid.get)  # ties: first in grid order
        lr, ent = [float(x.split("=")[1]) for x in best.split(",")]
        result[model] = {"grid": grid, "best": {"lr": lr, "ent_coef": ent}, "best_val": grid[best]}
    write_json(os.path.join(tdir, "tune", "tuned.json"), result)
    return result


def tuned_ppo(tier_name: str, model: str) -> dict:
    """The tuned (lr, entropy) for a method; SSM reuses the LSTM sweep (recorded in the manifest)."""
    path = os.path.join(tier_dir(tier_name), "tune", "tuned.json")
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    from . import plan as planmod
    amended = os.path.exists(planmod.AMEND) and "lstm-tuning-fallback" in open(planmod.AMEND).read()
    if amended and model in ("lstm", "ssm") and "ff" in d:  # dated amendment: the LSTM sweep was uninformative
        return dict(d["ff"]["best"])
    key = model if model in d else ("lstm" if model == "ssm" and "lstm" in d else None)
    return dict(d[key]["best"]) if key else {}
