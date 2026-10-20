"""Phase 4.2 and 4.3 of v2.1: the hand-fed map and the LSTM retune. Exploratory; validation suites only.

Selection uses `suite_val.json`; the winner is re-scored on `suite_val_s1.json` before the 10-seed run. The fresh
test suite is not opened here.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..core.distributions import CONDITIONS
from . import stats as S
from .experiments import load_model, load_tier, run_jobs, tier_dir, train_job, write_json
from .memory import OBS_VARIANTS
from .suites import Suite, per_seed_score
from .tune import tuned_ppo
from .v2_1 import v21_dir

LSTM_LRS = (3e-4, 1e-3, 2e-3, 3e-3)
LSTM_ENTS = (0.003, 0.01)
FF_EXTRA_LRS = (2e-3, 3e-3)
WINDOWS = (64, 128, 256)


def _job(tier_name: str, out_dir: str, model: str, seed: int, steps: int, ppo: dict, obs: str = "v2", arm: str = "B") -> dict:
    return dict(out_dir=out_dir, arm=arm, model=model, seed=seed, steps=steps, ppo=ppo, obs=obs,
                val_suite=os.path.join(tier_dir(tier_name), "suite_val.json"))


def _d(tier_name: str, *parts) -> str:
    return os.path.join(v21_dir(tier_name, "diagnose"), *parts)


def mapfed_jobs(tier_name: str) -> List[dict]:
    t = load_tier(tier_name)
    base = tuned_ppo(tier_name, "ff")
    return [_job(tier_name, _d(tier_name, "mapfed", f"seed_{s}"), "ff", s, t["steps"], base, obs="v9-map")
            for s in range(t["map_seeds"])]


def stage1_jobs(tier_name: str) -> Dict[str, List[dict]]:
    t = load_tier(tier_name)
    out = {}
    for lr in LSTM_LRS:
        for ent in LSTM_ENTS:
            key = f"lstm_lr{lr:g}_ent{ent:g}"
            out[key] = [_job(tier_name, _d(tier_name, "lstm_tune", "stage1", key, f"seed_{s}"), "lstm", s,
                             t["lstm_tune_steps"], dict(lr=lr, ent_coef=ent)) for s in range(t["lstm_tune_seeds"])]
    for lr in FF_EXTRA_LRS:  # equal effort for the feedforward policy
        key = f"ff_lr{lr:g}_ent0.003"
        out[key] = [_job(tier_name, _d(tier_name, "lstm_tune", "stage1", key, f"seed_{s}"), "ff", s,
                         t["lstm_tune_steps"], dict(lr=lr, ent_coef=0.003)) for s in range(t["lstm_tune_seeds"])]
    return out


def stage2_jobs(tier_name: str, best: dict, windows=WINDOWS) -> Dict[str, List[dict]]:
    t = load_tier(tier_name)
    out = {}
    for w in windows:
        key = f"lstm_window{w}"
        out[key] = [_job(tier_name, _d(tier_name, "lstm_tune", "stage2", key, f"seed_{s}"), "lstm", s,
                         t["lstm_tune_steps"], dict(best, seq_len=w)) for s in range(t["lstm_tune_seeds"])]
    return out


def final_jobs(tier_name: str, cfg: dict, n_seeds: int = 10, label: str = "lstm_final") -> List[dict]:
    t = load_tier(tier_name)
    return [_job(tier_name, _d(tier_name, label, f"seed_{s}"), "lstm", s, t["steps"], cfg) for s in range(n_seeds)]


def score_group(tier_name: str, jobs: List[dict], suite_file: str = "suite_val.json") -> Dict[str, np.ndarray]:
    """Ladder-normalised per-seed score of each run's selected checkpoint, per condition: {cond: (seeds,)}."""
    suite = Suite.load(os.path.join(tier_dir(tier_name), suite_file))
    scores = {c: [] for c in CONDITIONS}
    for j in jobs:
        m = load_model(j["out_dir"], j["model"])
        factory = OBS_VARIANTS[j.get("obs", "v2")][0]
        ev = suite.evaluate(m, env_factory=factory)
        for c in CONDITIONS:
            scores[c].append(per_seed_score(ev[c]))
    return {c: np.array(v) for c, v in scores.items()}


def mean_score(scores: Dict[str, np.ndarray]) -> float:
    return float(np.mean([scores[c].mean() for c in CONDITIONS]))


def run_group(tier_name: str, groups: Dict[str, List[dict]], workers: int, suite_file: str = "suite_val.json") -> dict:
    run_jobs(train_job, [j for js in groups.values() for j in js], workers)
    out = {}
    for key, js in groups.items():
        sc = score_group(tier_name, js, suite_file)
        out[key] = {"mean": mean_score(sc), "iqm": S.iqm_with_ci(sc), "per_seed": S.per_seed_table(sc),
                    "ppo": js[0]["ppo"], "steps": js[0]["steps"], "seeds": len(js)}
    return out


def mapfed(tier_name: str, workers: int = 4) -> dict:
    """4.2: feedforward PPO on obs-v9-map, against plain feedforward and the script (all on suite_val.json)."""
    t = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    mine = run_group(tier_name, {"ff_map": mapfed_jobs(tier_name)}, workers)["ff_map"]
    suite = Suite.load(os.path.join(tdir, "suite_val.json"))
    plain_jobs = [dict(out_dir=os.path.join(tdir, "headline", "arm_B", f"seed_{s}"), model="ff", obs="v2")
                  for s in range(t["map_seeds"])]
    lstm_jobs = [dict(out_dir=os.path.join(tdir, "memory", "lstm", "default", f"seed_{s}"), model="lstm", obs="v2")
                 for s in range(min(t["map_seeds"], t["memory_seeds"]))]
    plain, lstm = score_group(tier_name, plain_jobs), score_group(tier_name, lstm_jobs)
    mapped = score_group(tier_name, mapfed_jobs(tier_name))
    sc = suite.script_scores("memory_script")
    script_per = {c: float(np.mean(sc[c]["score"][sc[c]["valid"]])) for c in CONDITIONS}
    script_mean = float(np.mean(list(script_per.values())))
    out = {"tier_label": "exploratory", "suite": "suite_val.json", "steps": t["steps"], "seeds": t["map_seeds"],
           "script_mean": script_mean, "script_per_condition": script_per,
           "ff_map": {"mean": mean_score(mapped), "iqm": S.iqm_with_ci(mapped), "per_seed": S.per_seed_table(mapped)},
           "ff_plain": {"mean": mean_score(plain), "iqm": S.iqm_with_ci(plain), "per_seed": S.per_seed_table(plain)},
           "lstm_default": {"mean": mean_score(lstm), "iqm": S.iqm_with_ci(lstm)},
           "map_minus_plain": S.iqm_difference_with_ci(mapped, plain),
           "map_fraction_of_script": mean_score(mapped) / script_mean if script_mean else float("nan"),
           "note": "The map channels are food sites, water sites and known floor. No shelters exist in this environment."}
    write_json(os.path.join(v21_dir(tier_name, "diagnose"), "mapfed.json"), out)
    return out


def lstm_retune(tier_name: str, workers: int = 4, imitation: Optional[dict] = None) -> dict:
    """4.3 stage 1, equal-effort feedforward, stage 2. Returns the table and the winning config; the 10-seed run is separate."""
    out = {"tier_label": "exploratory", "suite": "suite_val.json"}
    s1 = run_group(tier_name, stage1_jobs(tier_name), workers)
    out["stage1"] = s1
    lstm_keys = [k for k in s1 if k.startswith("lstm_")]
    best_key = max(lstm_keys, key=lambda k: s1[k]["mean"])
    best = dict(s1[best_key]["ppo"])
    out["stage1_best"] = {"key": best_key, **best}
    windows = list(WINDOWS)
    if imitation:  # 4.1 rule: skip the middle window when cloning shows the window length does not matter
        c = imitation["clones"]
        a, b = c["lstm_w64"]["aggregate_mean"]["suite_val"], c["lstm_w256"]["aggregate_mean"]["suite_val"]
        if abs(a - b) < 0.05:
            windows = [64, 256]
        out["window_rule"] = {"lstm_w64_clone": a, "lstm_w256_clone": b, "windows_run": windows}
    s2 = run_group(tier_name, stage2_jobs(tier_name, best, tuple(windows)), workers)
    out["stage2"] = s2
    wbest = max(s2, key=lambda k: s2[k]["mean"])
    out["winner"] = {"key": wbest, "ppo": s2[wbest]["ppo"]}
    # P3: window 256 against window 64, same seeds, bootstrap interval on the IQM difference
    if "lstm_window256" in s2 and "lstm_window64" in s2:
        g2 = stage2_jobs(tier_name, best, tuple(windows))
        a, b = score_group(tier_name, g2["lstm_window256"]), score_group(tier_name, g2["lstm_window64"])
        out["window256_minus_window64"] = S.iqm_difference_with_ci(a, b)
    ff_best = max([k for k in s1 if k.startswith("ff_")], key=lambda k: s1[k]["mean"])
    ff_v2 = tuned_ppo(tier_name, "ff")
    out["ff_equal_effort"] = {"best_extra": ff_best, "best_extra_mean": s1[ff_best]["mean"], "v2_config": ff_v2}
    write_json(os.path.join(v21_dir(tier_name, "diagnose"), "lstm_retune.json"), out)
    return out
