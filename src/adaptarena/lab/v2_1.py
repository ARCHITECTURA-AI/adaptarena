"""AdaptArena v2.1 stages. Results go to results/<tier>/v2_1/<stage>/ with a tier label on every output.

Tier labels (analysis_plan_v2_1.yaml): confirmatory (pre-declared comparison on the fresh suite), exploratory
(validation suites only), diagnostic (no claim). The fresh suite is opened only through the access harness.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from ..core.distributions import CONDITIONS
from . import plan as planmod
from . import stats as S
from .experiments import ARMS, load_model, load_tier, tier_dir, write_json
from .suites import Suite, open_locked_test_suite, per_seed_score

SUITE_NAME = "suite_test_v2_1.json"
SUITE_SEED = 20261020
V2_SUITES = ["suite_test.json", "suite_test_memory.json"]


def v21_dir(tier_name: str, *parts: str) -> str:
    d = os.path.join(tier_dir(tier_name), "v2_1", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def open_suite(tier_name: str, purpose: str, generate: bool = False) -> Suite:
    tier = load_tier(tier_name)
    return open_locked_test_suite(tier_dir(tier_name), purpose, tier["scenarios_per_condition"], generate=generate,
                                  name=SUITE_NAME, plan_path=planmod.PLAN_V2_1, hash_path=planmod.HASH_V2_1,
                                  seed=SUITE_SEED, exclude_suites=V2_SUITES)


def generate_suite(tier_name: str) -> dict:
    """Phase 2.3. Must run after the plan commit is pushed; the harness refuses otherwise."""
    s = open_suite(tier_name, "Phase 2.3: generate suite_test_v2_1.json", generate=True)
    gen = {c: s.data["conditions"][c]["generation"] for c in CONDITIONS}
    write_json(os.path.join(v21_dir(tier_name, "suite"), "generation.json"), {"seed": SUITE_SEED, "excluded": V2_SUITES, "conditions": gen})
    return gen


def rescore_existing(tier_name: str) -> dict:
    """Phase 2.4. Score the saved checkpoints of arms A to D (10 seeds each) and the LSTM on the fresh suite."""
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    suite = open_suite(tier_name, "Phase 2.4: re-score existing arms and LSTM")
    rows, scores = [], {}
    missing = []
    groups = [(f"arm_{a}", os.path.join(tdir, "headline", f"arm_{a}"), "ff", tier["seeds"]) for a in ARMS]
    groups.append(("lstm_default", os.path.join(tdir, "memory", "lstm", "default"), "lstm", tier["memory_seeds"]))
    for label, base, kind, n in groups:
        scores[label] = {c: [] for c in CONDITIONS}
        for s in range(n):
            d = os.path.join(base, f"seed_{s}")
            if not os.path.exists(os.path.join(d, "model_best.pt")):
                missing.append(d)
                continue
            ev = suite.evaluate(load_model(d, kind))
            for c in CONDITIONS:
                scores[label][c].append(per_seed_score(ev[c]))
                for k, r in enumerate(ev[c]["results"]):
                    rows.append({"policy": label, "seed": s, "condition": c, "idx": k, "sid": r.sid, "survival": r.survival,
                                 "cause": r.cause, "mode": "deterministic", "final_hash": r.final_hash})
    out = {"tier_label": "confirmatory-yardstick (re-scoring only; no new claim)", "missing_checkpoints": missing, "arms": {}}
    for label, per in scores.items():
        sc = {c: np.array(v) for c, v in per.items()}
        out["arms"][label] = {"aggregate_iqm": S.iqm_with_ci(sc), "per_condition": {c: S.iqm_with_ci({c: sc[c]}) for c in CONDITIONS},
                              "per_seed": S.per_seed_table(sc)}
    from .experiments import write_csv
    write_csv(os.path.join(v21_dir(tier_name, "rescore"), "episodes.csv"), rows)
    write_json(os.path.join(v21_dir(tier_name, "rescore"), "rescore.json"), out)
    return out
