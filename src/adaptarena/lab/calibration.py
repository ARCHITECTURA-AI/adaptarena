"""Difficulty calibration: baselines and planner on validation seeds only, no learned policy.

The numbers (and the constants in `core/params.py` and `core/distributions.py` that produced them)
are frozen before any learned policy is run; `docs/calibration.md` quotes this output.
"""
from __future__ import annotations

import os

import numpy as np

from ..core.distributions import CONDITIONS
from .experiments import get_val_suite, load_tier, tier_dir, write_json
from .evaluate import ladder_normalise


def run(tier_name: str) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    suite = get_val_suite(tdir, tier, 0)
    rows = {}
    print(f"{'condition':18s} {'n':>3s} {'cert':>5s} {'random':>7s} {'reactive':>9s} {'memory':>7s} {'planner':>8s}")
    for c in CONDITIONS:
        d = suite.data["conditions"][c]
        g = d["generation"]
        rows[c] = {"n": len(d["specs"]), "rejected_uncertified": g["rejected_uncertified"], "tried": g["tried"],
                   "random": float(np.mean(suite.rand[c])), "reactive": float(np.mean(suite.scripts[c]["reactive_script"])),
                   "memory": float(np.mean(suite.scripts[c]["memory_script"])), "planner": float(np.mean(suite.planner[c]))}
        r = rows[c]
        print(f"{c:18s} {r['n']:3d} {g['tried'] - g['rejected_uncertified']:2d}/{g['tried']:<2d} {r['random']:7.1f} "
              f"{r['reactive']:9.1f} {r['memory']:7.1f} {r['planner']:8.1f}")
    write_json(os.path.join(tdir, "calibration.json"), rows)
    return rows
