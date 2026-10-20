"""Phase 5a of v2.1: the budget sweep (exploratory) and the mutation arms E, E-prime, F (confirmatory at 2M).

Training is resumable (`run_jobs` skips finished runs). Scoring on the fresh suite goes through the access harness
only (`v2_1.open_suite`), once per call, and the purpose is logged. The selected outcome row is computed by the rules
written in analysis_plan_v2_1.yaml and restated here; nothing is picked by hand.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from ..core.distributions import CONDITIONS
from . import stats as S
from .experiments import ARMS, load_model, load_tier, run_jobs, tier_dir, train_job, write_json
from .suites import per_seed_score
from .tune import tuned_ppo
from .v2_1 import open_suite, v21_dir

MUTATION_ARMS = {"E": "Regret + mutation", "Eprime": "Value loss + mutation (control)", "F": "Regret over 4 rollouts"}
SWEEP_ARMS = ("B", "D")
EVAL_EVERY = {"mid": 100, "high": 250}  # updates between validations; 25 at 2M, so ~39 evaluations in every run


def sweep_jobs(tier_name: str, levels=("mid", "high")) -> List[dict]:
    t = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    base = tuned_ppo(tier_name, "ff")
    jobs = []
    for level, steps in (("mid", t["budget_steps_mid"]), ("high", t["budget_steps_high"])):
        if level not in levels:
            continue
        for arm in SWEEP_ARMS:
            for s in range(t["budget_seeds"]):
                jobs.append(dict(out_dir=os.path.join(v21_dir(tier_name, "scale"), f"arm_{arm}", f"steps_{steps}", f"seed_{s}"),
                                 arm=arm, model="ff", seed=s, steps=steps, ppo=dict(base, eval_every=EVAL_EVERY[level]),
                                 val_suite=os.path.join(tdir, "suite_val.json")))
    return jobs


def mutation_jobs(tier_name: str) -> List[dict]:
    t = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    base = tuned_ppo(tier_name, "ff")
    return [dict(out_dir=os.path.join(v21_dir(tier_name, "scale"), "mutation", f"arm_{arm}", f"seed_{s}"), arm=arm,
                 model="ff", seed=s, steps=t["steps"], ppo=base, val_suite=os.path.join(tdir, "suite_val.json"))
            for arm in MUTATION_ARMS for s in range(t["mutation_seeds"])]


def v2_arm_dirs(tier_name: str, arm: str, n: int) -> List[str]:
    return [os.path.join(tier_dir(tier_name), "headline", f"arm_{arm}", f"seed_{s}") for s in range(n)]


def _run_stats(d: str) -> dict:
    out = {}
    p = os.path.join(d, "train_result.json")
    if os.path.exists(p):
        r = json.load(open(p))
        ex = dict(r.get("extra", {}))
        ex.pop("snapshots", None)
        out.update({k: ex[k] for k in ("repeat_fraction", "distinct_scenarios", "episodes", "score_rank_correlation",
                                       "planner_calls", "planner_seconds", "children", "children_accepted",
                                       "children_rejected_uncertified", "children_rejected_by_buffer",
                                       "uncertified_rate", "acceptance_rate", "replays", "fresh") if k in ex})
        out["wall_seconds"] = r.get("wall_seconds")
        out["steps"] = r.get("steps")
    return out


def _score_dirs(suite, dirs: List[str]) -> Dict[str, np.ndarray]:
    sc = {c: [] for c in CONDITIONS}
    for d in dirs:
        ev = suite.evaluate(load_model(d, "ff"))
        for c in CONDITIONS:
            sc[c].append(per_seed_score(ev[c]))
    return {c: np.array(v) for c, v in sc.items()}


def _table(scores: Dict[str, np.ndarray]) -> dict:
    return {"aggregate_iqm": S.iqm_with_ci(scores), "per_seed": S.per_seed_table(scores),
            "per_condition": {c: float(np.mean(scores[c])) for c in CONDITIONS}}


# ------------------------------------------------------------------------------------------ 5a.1 budget sweep
def sweep_outcome(iqm: Dict[str, Dict[int, float]], diff: Dict[int, dict], budgets: List[int]) -> str:
    """The four rows of the plan's budget-sweep table, applied in a fixed order. `iqm[arm][budget]` are point estimates."""
    b2, b8, b20 = budgets
    plateau = all(iqm[a][b20] - iqm[a][b8] <= 0.02 for a in SWEEP_ARMS)
    if plateau:
        return "plateau"          # neither arm improves from the middle to the high budget: the bottleneck is elsewhere
    if all(diff[b]["diff"] < 0 for b in budgets):
        return "b_stays_ahead"    # the v2 negative result is robust to budget
    both_rise = all(iqm[a][b20] > iqm[a][b2] for a in SWEEP_ARMS)
    if both_rise and diff[b20]["diff"] > 0 and diff[b20]["lo"] > 0:
        return "d_overtakes"      # regret helps once the learner is competent
    return "gap_closes"           # the interval on D minus B at the high budget includes zero: neither helps nor hurts


SWEEP_TEXT = {
    "d_overtakes": "Both arms rise and D overtakes B at the high budget: regret helps once the learner is competent, so the v2 result was budget-limited.",
    "gap_closes": "Both arms rise and the D minus B interval includes zero: the v2 gap closes with budget, and regret neither helps nor hurts.",
    "b_stays_ahead": "B stays ahead of D at every budget: the v2 negative result is robust to budget.",
    "plateau": "Neither arm improves from the middle to the high budget: the learner has plateaued, so the bottleneck is elsewhere (see Phase 4)."}


def evaluate_sweep(tier_name: str) -> dict:
    t = load_tier(tier_name)
    budgets = [t["steps"], t["budget_steps_mid"], t["budget_steps_high"]]
    n = t["budget_seeds"]
    suite = open_suite(tier_name, "Phase 5a.1: budget sweep, B and D at 2M, 8M and 20M (exploratory)")
    points, per_arm = {}, {}
    for arm in SWEEP_ARMS:
        for steps in budgets:
            dirs = v2_arm_dirs(tier_name, arm, n) if steps == t["steps"] else [
                os.path.join(v21_dir(tier_name, "scale"), f"arm_{arm}", f"steps_{steps}", f"seed_{s}") for s in range(n)]
            sc = _score_dirs(suite, dirs)
            per_arm[(arm, steps)] = sc
            points[f"{arm}@{steps}"] = {**_table(sc), "run_stats": [_run_stats(d) for d in dirs]}
    diffs = {b: S.iqm_difference_with_ci(per_arm[("D", b)], per_arm[("B", b)]) for b in budgets}
    iqm = {a: {b: S.agg_iqm(per_arm[(a, b)]) for b in budgets} for a in SWEEP_ARMS}
    key = sweep_outcome(iqm, diffs, budgets)
    out = {"tier_label": "exploratory", "suite": "suite_test_v2_1.json", "seeds": n, "budgets": budgets, "points": points,
           "d_minus_b": {str(b): diffs[b] for b in budgets}, "outcome": key, "outcome_text": SWEEP_TEXT[key],
           "claim_status": "descriptive; with this few seeds no claim is labeled supported"}
    write_json(os.path.join(v21_dir(tier_name, "scale"), "sweep.json"), out)
    return out


# ------------------------------------------------------------------------------------------ 5a.2 mutation arms
MUTATION_TEXT = {
    "e_beats_b_and_eprime": "E beats B and E-prime: regret scoring plus mutation helps in this environment.",
    "e_beats_b_ties_eprime": "E beats B but ties E-prime: mutation helps and the planner adds nothing over value loss.",
    "e_ties_or_loses": "E ties or loses to B: the v2 conclusion stands under a fairer buffer."}


def mutation_outcome(e_vs_b: dict, e_vs_ep: dict) -> str:
    if e_vs_b["lo"] > 0 and e_vs_ep["lo"] > 0:
        return "e_beats_b_and_eprime"
    if e_vs_b["lo"] > 0:
        return "e_beats_b_ties_eprime"
    return "e_ties_or_loses"


def status(diff: dict) -> str:
    if diff["lo"] > 0:
        return "supported"
    if diff["hi"] < 0:
        return "not supported (reversed)"
    return "not supported (interval includes zero)"


def evaluate_mutation(tier_name: str) -> dict:
    t = load_tier(tier_name)
    n = t["mutation_seeds"]
    suite = open_suite(tier_name, "Phase 5a.2: mutation arms E, E-prime, F against B and D (confirmatory: E vs B, E vs E-prime)")
    arm_dirs = {"B": v2_arm_dirs(tier_name, "B", n), "D": v2_arm_dirs(tier_name, "D", n)}
    for a in MUTATION_ARMS:
        arm_dirs[a] = [os.path.join(v21_dir(tier_name, "scale"), "mutation", f"arm_{a}", f"seed_{s}") for s in range(n)]
    scores = {a: _score_dirs(suite, d) for a, d in arm_dirs.items()}
    tables = {a: {**_table(scores[a]), "run_stats": [_run_stats(d) for d in arm_dirs[a]]} for a in scores}
    cmp_ = {}
    for name, (x, y), tier_label in (("E_vs_B", ("E", "B"), "confirmatory"), ("E_vs_Eprime", ("E", "Eprime"), "confirmatory"),
                                     ("E_vs_D", ("E", "D"), "exploratory"), ("F_vs_D", ("F", "D"), "exploratory"),
                                     ("F_vs_B", ("F", "B"), "exploratory")):
        d = S.iqm_difference_with_ci(scores[x], scores[y])
        p = S.prob_improvement_with_ci(scores[x], scores[y])
        cmp_[name] = {"tier": tier_label, "iqm_difference": d, "prob_improvement": p, "status": status(d)}
    key = mutation_outcome(cmp_["E_vs_B"]["iqm_difference"], cmp_["E_vs_Eprime"]["iqm_difference"])
    best = max(tables[a]["aggregate_iqm"]["iqm"] for a in tables)
    out = {"tier_label": "confirmatory (E vs B, E vs E-prime); exploratory (the rest)", "suite": "suite_test_v2_1.json",
           "seeds": n, "steps": t["steps"], "arms": tables, "comparisons": cmp_, "outcome": key,
           "outcome_text": MUTATION_TEXT[key], "headroom_rule": {
               "best_arm_iqm": best, "learner_limited": bool(best < 0.5),
               "text": ("Every learned arm stays below 0.5 of the planner-random ladder: the experiment is learner-limited, "
                        "and the arms cannot be separated at this competence.") if best < 0.5 else "At least one arm reaches 0.5 of the ladder."}}
    write_json(os.path.join(v21_dir(tier_name, "scale"), "mutation.json"), out)
    return out


def run(tier_name: str, workers: int = 4, which=("mutation", "sweep")) -> None:
    """Train (resumable), then score through the harness."""
    if "mutation" in which:
        run_jobs(train_job, mutation_jobs(tier_name), workers)
        evaluate_mutation(tier_name)
    if "sweep-mid" in which:  # train the middle budget only; scoring waits for the high budget
        run_jobs(train_job, sweep_jobs(tier_name, ("mid",)), workers)
    if "sweep" in which:
        run_jobs(train_job, sweep_jobs(tier_name), workers)
        evaluate_sweep(tier_name)
