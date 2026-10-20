"""v2.1 Phase 1 audits (diagnostic tier: no claim is made from these tables).

1.1 specialist check: is arm A's pooled IQM of 0.010 a bug or a finding?
1.2 selection-bias audit: `best_val` is the maximum over ~13 noisy evaluations on the same 70-scenario suite that
    chose the checkpoint, so it runs high. Every run's selected checkpoint is re-scored on a different validation suite.
1.3 access-log audit: the locked suite was opened exactly as many times as recorded, each with the locked plan hash.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

from ..core.distributions import CONDITIONS, sample_specialist
from ..core.planner import plan
from ..core.scenario import build_scenario
from . import plan as planmod
from .evaluate import evaluate, ladder_normalise, random_mean, survival_array
from .experiments import get_val_suite, load_model, load_tier, tier_dir, write_json


def _score(model, specs, planner, rand):
    surv = survival_array(evaluate(model, specs))
    sc, ok = ladder_normalise(surv, rand, planner)
    return float(surv.mean()), float(np.mean(sc[ok])) if ok.any() else float("nan")


def specialist_check(tier_name: str = "full") -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    val = get_val_suite(tdir, tier, 0)
    rng = np.random.default_rng([0xA0D17, 1])
    train_like, planner, rand = [], [], []
    while len(train_like) < 20:  # the exact training distribution of arm A: layout 0, training-range timing/weather seeds
        s = sample_specialist(rng)
        r = plan(build_scenario(s))
        if r.certified:
            train_like.append(s)
    planner = np.full(20, 500.0)
    rand = random_mean(train_like, 10)
    fam, unseen = val.specs["familiar"], val.specs["unseen_layout"]
    out = {"sampler": {"layout_id": 0, "timing_seed": "drawn per episode in [0, 1e9)", "weather_seed": "drawn per episode in [0, 1e9)",
                       "source": "core/distributions.sample_specialist via lab/sources.make_source('A')"}, "seeds": []}
    for s in range(tier["seeds"]):
        d = os.path.join(tdir, "headline", "arm_A", f"seed_{s}")
        rows = list(csv.DictReader(open(os.path.join(d, "val_log.csv"))))
        m = load_model(d, "ff")
        t_raw, t_ls = _score(m, train_like, planner, rand)
        f_raw, f_ls = _score(m, fam, val.planner["familiar"], val.rand["familiar"])
        u_raw, u_ls = _score(m, unseen, val.planner["unseen_layout"], val.rand["unseen_layout"])
        out["seeds"].append({"seed": s, "val_update0": float(rows[0]["val_survival"]), "val_best": max(float(r["val_survival"]) for r in rows),
                             "train_like_raw": t_raw, "train_like_score": t_ls, "familiar_raw": f_raw, "familiar_score": f_ls,
                             "fresh_layouts_raw": u_raw, "fresh_layouts_score": u_ls})
    mean = lambda k: float(np.mean([x[k] for x in out["seeds"]]))
    out["mean"] = {k: mean(k) for k in out["seeds"][0] if k != "seed"}
    t, f = out["mean"]["train_like_score"], out["mean"]["familiar_score"]
    if t > 0.5 and f < 0.2:
        row = "Training scenario near 500, familiar near random: memorised one event sequence"
    elif t < 0.2:
        row = "Training scenario also near random: learning failure or wiring bug"
    elif t > 0.5 and f > 0.5:
        row = "Both high: the pooled 0.010 comes from the evaluation path (pooled IQM over seven conditions, six of which are unseen layouts)"
    else:
        row = "Intermediate: see the table"
    out["decision_row"] = row
    write_json(os.path.join(tdir, "v2_1", "audit", "specialist.json"), out)
    return out


def selection_audit(tier_name: str = "full") -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    s1 = get_val_suite(tdir, tier, 1)
    specs = s1.all_specs()
    groups = {"headline": ("headline/arm_*/seed_*", "ff"), "ablations": ("ablations/*/seed_*", "ff"),
              "parity": ("parity/own/seed_*", "ff"), "memory_lstm": ("memory/lstm/*/seed_*", "lstm"),
              "memory_ff": ("memory/ff/*/seed_*", "ff")}
    rows = []
    for g, (pat, kind) in groups.items():
        for d in sorted(glob.glob(os.path.join(tdir, pat))):
            if not os.path.exists(os.path.join(d, "model_best.pt")) or not os.path.exists(os.path.join(d, "done.json")):
                continue
            cfg = json.load(open(os.path.join(d, "done.json")))["job"]
            from ..core.env import EnvConfig
            m = load_model(d, kind)
            res = evaluate(m, specs, list(range(len(specs))), EnvConfig(**(cfg.get("env") or {})))
            rescored = float(survival_array(res).mean())
            best = json.load(open(os.path.join(d, "done.json")))["best_val"]
            rows.append({"group": g, "run": os.path.relpath(d, tdir), "best_val": best, "rescored_val_s1": rescored, "gap": best - rescored})
    out = {"rows": rows, "by_group": {}}
    for g in groups:
        gaps = [r["gap"] for r in rows if r["group"] == g]
        if gaps:
            out["by_group"][g] = {"n": len(gaps), "mean_gap": float(np.mean(gaps)), "sd_gap": float(np.std(gaps)),
                                  "mean_best_val": float(np.mean([r["best_val"] for r in rows if r["group"] == g])),
                                  "mean_rescored": float(np.mean([r["rescored_val_s1"] for r in rows if r["group"] == g]))}
    # which suite do the published tables use?
    out["suite_used_for_tables"] = {"headline": "suite_test.json (locked; never used for checkpoint selection)",
                                    "ablations": "suite_val.json, the same suite that chose each checkpoint (biased upward)",
                                    "memory_variants": "suite_test.json and suite_test_memory.json (locked)"}
    write_json(os.path.join(tdir, "v2_1", "audit", "selection.json"), out)
    # ablation effects recomputed on the unbiased suite
    from . import stats as S
    from .ablations import VARIANTS
    from .suites import per_seed_score
    sc = {}
    for name, (fam, env, ppo) in VARIANTS.items():
        per = {c: [] for c in CONDITIONS}
        from ..core.env import EnvConfig
        for sd in range(tier["ablation_seeds"]):
            m = load_model(os.path.join(tdir, "ablations", name, f"seed_{sd}"), "ff")
            ev = s1.evaluate(m, EnvConfig(**env))
            for c in CONDITIONS:
                per[c].append(per_seed_score(ev[c]))
        sc[name] = {c: np.array(v) for c, v in per.items()}
    table = {}
    for name in VARIANTS:
        if name == "default":
            continue
        e = S.iqm_difference_with_ci(sc[name], sc["default"], B=4000)
        table[name] = e
    out["ablation_effects_on_s1"] = table
    write_json(os.path.join(tdir, "v2_1", "audit", "selection.json"), out)
    return out


def access_log_audit(tier_name: str = "full") -> dict:
    tdir = tier_dir(tier_name)
    log = [json.loads(l) for l in open(os.path.join(tdir, "test_access_log.jsonl"))]
    locked = open(planmod.HASH).read().strip()
    out = {"n_accesses": len(log), "plan_hash": locked,
           "all_match": all(r.get("plan_hash") == locked for r in log), "purposes": [r["purpose"] for r in log]}
    write_json(os.path.join(tdir, "v2_1", "audit", "access_log.json"), out)
    return out


def run(tier_name: str = "full") -> dict:
    return {"specialist": specialist_check(tier_name), "selection": selection_audit(tier_name), "access": access_log_audit(tier_name)}
