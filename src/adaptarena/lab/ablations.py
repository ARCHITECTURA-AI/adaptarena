"""Ablation ladder: six single-change ablations of the default configuration (arm B, sparse reward).

Rules (also in analysis_plan.yaml): 5 training seeds per variant, scored on the validation suite and never
on the locked test suite; every result is labelled exploratory; effects are differences in ladder-normalised
survival with stratified-bootstrap CIs, plus raw survival; a null effect is reported, a dead input is a finding.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from ..core import params as P
from ..core.distributions import CONDITIONS
from ..core.env import EnvConfig, SurvivalEnv
from . import stats as S
from .evaluate import evaluate
from .experiments import (get_val_suite, load_model, load_tier, run_jobs, tier_dir, train_job, write_csv,
                          write_json)
from .suites import per_seed_score

# name -> (family, env overrides, ppo overrides)
VARIANTS: Dict[str, tuple] = {
    "default": ("default", {}, {}),
    "gamma_0.9": ("discount", {}, {"gamma": 0.9}),
    "gamma_0.999": ("discount", {}, {"gamma": 0.999}),
    "vision_5": ("vision", {"vision": 5}, {}),
    "vision_3": ("vision", {"vision": 3}, {}),
    "no_prev_action": ("previous action", {"prev_action": False}, {}),
    "no_cost_stat": ("energy-cost stat", {"cost_stat": False}, {}),
    "reward_naive": ("reward", {"reward": "naive"}, {}),
    "reward_potential": ("reward", {"reward": "potential"}, {}),
    "timeout_as_death": ("truncation", {}, {"bootstrap_timeout": False}),
}
COMPARE = {  # variant -> its pairing for the "does it matter" difference (always against default)
    v: "default" for v in VARIANTS if v != "default"}


def jobs(tier_name: str) -> List[dict]:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    vpath = os.path.join(tdir, "suite_val.json")
    from .tune import tuned_ppo
    tuned = tuned_ppo(tier_name, "ff")
    out = []
    for name, (fam, env, ppo) in VARIANTS.items():
        for s in range(tier["ablation_seeds"]):
            out.append(dict(out_dir=os.path.join(tdir, "ablations", name, f"seed_{s}"), arm="B", model="ff", seed=s,
                            steps=tier["ablation_steps"], env=env, ppo={**tuned, **ppo}, val_suite=vpath))
    return out


def wasted_interacts(results, specs, env_cfg) -> float:
    """Valid interacts taken with the matching stat >= 90%: the signature of farming a naive shaping bonus."""
    waste = total = 0
    for r, spec in zip(results, specs):
        env = SurvivalEnv(env_cfg)
        env.reset(spec)
        for a in r.actions:
            if env.done:
                break
            e0, h0 = env.energy, env.hydration
            _, _, _, _, info = env.step(a)
            if a == P.INTERACT and info["valid"]:
                total += 1
                if max(e0, h0) >= 9000 and (env.energy > e0 - 100 or env.hydration > h0 - 100):
                    waste += 1
    return waste / max(total, 1)


def run(tier_name: str, workers: int = 4) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    suite = get_val_suite(tdir, tier, 0)
    run_jobs(train_job, jobs(tier_name), workers)
    scores: Dict[str, Dict[str, np.ndarray]] = {}
    raw: Dict[str, dict] = {}
    exploit: Dict[str, float] = {}
    for name, (fam, env, ppo) in VARIANTS.items():
        cfg = EnvConfig(**env)
        per = {c: [] for c in CONDITIONS}
        raws = {c: [] for c in CONDITIONS}
        ex = []
        for s in range(tier["ablation_seeds"]):
            m = load_model(os.path.join(tdir, "ablations", name, f"seed_{s}"), "ff")
            ev = suite.evaluate(m, cfg, record=("reward" in env) or name == "default")
            for c in CONDITIONS:
                per[c].append(per_seed_score(ev[c]))
                raws[c].append(float(np.mean(ev[c]["survival"])))
            if "reward" in env or name == "default":
                ex.append(np.mean([wasted_interacts(ev[c]["results"], suite.specs[c], cfg) for c in CONDITIONS]))
        scores[name] = {c: np.array(v) for c, v in per.items()}
        raw[name] = {c: float(np.mean(v)) for c, v in raws.items()}
        if ex:
            exploit[name] = float(np.mean(ex))
    rows, out = [], {"label": "exploratory: validation suite only, never the locked test suite",
                     "obs_versions": {n: EnvConfig(**v[1]).obs_version for n, v in VARIANTS.items()},
                     "variants": {}, "wasted_interact_fraction": exploit}
    for name, (fam, env, ppo) in VARIANTS.items():
        agg = S.iqm_with_ci(scores[name])
        entry = {"family": fam, "iqm": agg, "raw_survival_mean": float(np.mean(list(raw[name].values()))),
                 "raw_by_condition": raw[name]}
        if name != "default":
            entry["effect_vs_default"] = S.iqm_difference_with_ci(scores[name], scores["default"], B=4000)
            eff = entry["effect_vs_default"]
            entry["finding"] = ("no measurable effect (interval includes zero)" if eff["lo"] <= 0 <= eff["hi"]
                                else ("hurts" if eff["hi"] < 0 else "helps"))
        out["variants"][name] = entry
        rows.append({"variant": name, "family": fam, "iqm": agg["iqm"], "lo": agg["lo"], "hi": agg["hi"],
                     "raw_survival": entry["raw_survival_mean"],
                     "effect": entry.get("effect_vs_default", {}).get("diff", ""),
                     "effect_lo": entry.get("effect_vs_default", {}).get("lo", ""),
                     "effect_hi": entry.get("effect_vs_default", {}).get("hi", "")})
    write_json(os.path.join(tdir, "ablations", "ablations.json"), out)
    write_csv(os.path.join(tdir, "ablations", "ablations.csv"), rows)
    return out
