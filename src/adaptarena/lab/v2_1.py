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


def _speed_job(job: dict) -> dict:
    import time
    import torch
    torch.set_num_threads(1)
    from .models import make_model
    from .ppo import PPOConfig, train
    from .sources import make_source
    from .suites import Suite
    from .tune import tuned_ppo
    val_specs = Suite.load(job["val_suite"]).all_specs()
    kw = dict(total_steps=job["steps"], seed=job["seed"], profile=True)
    kw.update(tuned_ppo(job["tier"], "ff"))
    kw.update(job.get("flags", {}))
    torch.manual_seed(job["seed"])
    t0 = time.time()
    res = train(make_model("ff"), PPOConfig(**kw), make_source("B", job["seed"]), job["env"], val_specs, job["out_dir"])
    return {"sps": res.steps / (time.time() - t0), "wall": time.time() - t0, "profile": res.extra.get("profile_seconds"),
            "profile_total": res.extra.get("profile_total_seconds"), "best_val": res.best_val}


def profile_training(tier_name: str, steps: int = 100_000, parallel: int = 4) -> dict:
    """Phase 3.1. Diagnostic: where one arm-B job spends its time, alone and with `parallel` jobs on the machine."""
    import concurrent.futures as cf
    import tempfile
    from ..core.env import EnvConfig
    tdir = tier_dir(tier_name)
    base = dict(tier=tier_name, steps=steps, val_suite=os.path.join(tdir, "suite_val.json"), env=EnvConfig())

    def mk(i):
        return dict(base, seed=i, out_dir=tempfile.mkdtemp(prefix="speed_"))
    single = _speed_job(mk(0))
    with cf.ProcessPoolExecutor(parallel) as ex:
        multi = list(ex.map(_speed_job, [mk(i) for i in range(parallel)]))
    out = {"tier_label": "diagnostic", "steps": steps, "single": single, "parallel": parallel,
           "parallel_jobs": multi, "parallel_sps_total": sum(m["sps"] for m in multi)}
    write_json(os.path.join(v21_dir(tier_name, "speed"), "profile.json"), out)
    return out


STAGES = ("audit", "suite", "rescore", "profile", "imitation", "mapfed", "lstm-retune", "lstm-final", "predictions",
          "mutation", "sweep", "ssm", "int8", "gamma")


def _int8_groups(tier_name: str) -> dict:
    tdir = tier_dir(tier_name)
    t = load_tier(tier_name)
    groups = {"ppo_ff": {"kind": "ff", "dirs": [os.path.join(tdir, "headline", "arm_B", f"seed_{s}") for s in range(t["seeds"])]}}
    final = os.path.join(v21_dir(tier_name, "diagnose"), "lstm_final")
    lstm_dirs = sorted(os.path.join(final, d) for d in os.listdir(final)) if os.path.isdir(final) else []
    if not lstm_dirs:
        lstm_dirs = [os.path.join(tdir, "memory", "lstm", "default", f"seed_{s}") for s in range(t["memory_seeds"])]
    groups["ppo_lstm"] = {"kind": "lstm", "dirs": [d for d in lstm_dirs if os.path.exists(os.path.join(d, "model_best.pt"))]}
    ssm = os.path.join(v21_dir(tier_name, "ssm_int8"), "ssm", "final")
    ssm_dirs = sorted(os.path.join(ssm, d) for d in os.listdir(ssm)) if os.path.isdir(ssm) else []
    groups["ppo_ssm"] = {"kind": "ssm", "dirs": [d for d in ssm_dirs if os.path.exists(os.path.join(d, "model_best.pt"))]}
    return {k: v for k, v in groups.items() if v["dirs"]}


def run_stage(stage: str, tier_name: str, workers: int = 4):
    """One v2.1 stage. Every stage is resumable and writes under results/<tier>/v2_1/<stage>/ with a tier label."""
    if stage == "audit":
        from . import audit
        return audit.run(tier_name)
    if stage == "suite":
        return generate_suite(tier_name)
    if stage == "rescore":
        return rescore_existing(tier_name)
    if stage == "profile":
        return profile_training(tier_name)
    if stage == "imitation":
        from . import imitation
        return imitation.run(tier_name, workers)
    if stage == "mapfed":
        from . import phase4
        return phase4.mapfed(tier_name, workers)
    if stage == "lstm-retune":
        from . import phase4
        im = os.path.join(v21_dir(tier_name, "diagnose"), "imitation.json")
        return phase4.lstm_retune(tier_name, workers, json.load(open(im)) if os.path.exists(im) else None)
    if stage == "lstm-final":
        from . import phase4
        return phase4.lstm_final(tier_name, workers)
    if stage == "predictions":
        from . import phase4
        return phase4.predictions(tier_name)
    if stage == "mutation":
        from . import phase5
        return phase5.run(tier_name, workers, which=("mutation",))
    if stage == "sweep":
        from . import phase5
        return phase5.run(tier_name, workers, which=("sweep",))
    if stage == "ssm":
        from . import phase5b
        return phase5b.ssm_run(tier_name, workers)
    if stage == "int8":
        from . import phase5b
        return phase5b.int8_run(tier_name, _int8_groups(tier_name))
    if stage == "gamma":
        from . import phase5b
        return phase5b.gamma_run(tier_name, workers)
    raise ValueError(stage)
