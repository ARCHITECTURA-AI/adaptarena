"""README results block and v2.1 ledger rows, generated from results/<tier>/. No number is typed by hand.

`write_readme` rewrites the text between <!-- RESULTS:START --> and <!-- RESULTS:END -->. The v2 block stays, labelled as
published; v2.1 appears as its own subsection. A stage that has not run is stated as "not run", never filled in.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from . import plan as planmod
from .experiments import load_tier, tier_dir

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
ARM_NAMES = {"A": "A: specialist", "B": "B: uniform randomization", "C": "C: replay by value loss", "D": "D: replay by planner regret",
             "E": "E: regret replay + mutation", "Eprime": "E-prime: value-loss replay + mutation (control)", "F": "F: regret over 4 rollouts"}
ABL_LABEL = {"gamma_0.9": "gamma 0.9", "gamma_0.999": "gamma 0.999", "vision_3": "3x3 vision", "vision_5": "5x5 vision",
             "no_prev_action": "removing the previous action", "no_cost_stat": "removing the energy-cost stat",
             "reward_naive": "naive reward shaping", "reward_potential": "potential-based shaping", "timeout_as_death": "time-out treated as death"}


def _j(path: str) -> Optional[dict]:
    return json.load(open(path)) if os.path.exists(path) else None


def _f(x, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _ci(d: dict, k: str = "iqm", dd: int = 3) -> str:
    return f"{_f(d[k], dd)} [{_f(d['lo'], dd)}, {_f(d['hi'], dd)}]"


def _join(items: List[str]) -> str:
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1] if items else "none"


# ------------------------------------------------------------------------------------------------ v2 (as published)
def v2_block(tier_name: str) -> str:
    tdir = tier_dir(tier_name)
    t = load_tier(tier_name)
    hl = _j(os.path.join(tdir, "headline", "stats.json"))
    if hl is None:
        return "## Result\n\nThe headline experiment has not been run for this tier.\n"
    dg = _j(os.path.join(tdir, "headline", "diagnosis.json")) or {}
    par = _j(os.path.join(tdir, "parity", "parity.json")) or {}
    mem = _j(os.path.join(tdir, "memory", "memory.json")) or {}
    abl = _j(os.path.join(tdir, "ablations", "ablations.json")) or {}
    spec = _j(os.path.join(tdir, "v2_1", "audit", "specialist.json")) or {}
    ledger = {x["id"]: x for x in hl["ledger"]}
    dec = hl["decision"]
    head = {"D_loses_to_B": "**Planner-regret replay did not beat uniform randomization; it lost.**",
            "D_beats_B_and_C": "**Planner-regret replay beat uniform randomization and value-loss replay.**",
            "D_ties_C": "**Planner-regret replay beat uniform randomization but not value-loss replay.**",
            "D_ties_B": "**Planner-regret replay showed no measurable benefit over uniform randomization.**"}.get(dec["outcome"], f"**{dec['text']}**")
    out = [f"### v2 result, as published (full tier: {t['seeds']} seeds per arm, {t['steps'] / 1e6:g}M transitions each, "
           f"locked test suite of {len(hl['conditions'])} conditions x {t['scenarios_per_condition']} certified scenarios)", "",
           f"{head} The pre-declared outcome rule fired as `{dec['outcome']}`.", "",
           "| Arm | Aggregate ladder-normalised IQM [95% CI] |", "| --- | --- |"]
    for a, name in hl["arms"].items():
        label = ARM_NAMES.get(a, a)
        if a == "A" and spec.get("mean"):
            m = spec["mean"]
            label += f" (pooled over all {len(hl['conditions'])} conditions; {_f(m['train_like_score'], 2)} on its own layout, {_f(m['fresh_layouts_score'], 2)} on unseen layouts)"
        out.append(f"| {label} | {_ci(hl['per_arm'][a]['aggregate_iqm'])} |")
    out.append("")
    for k, nm in (("D_vs_B", "D vs B"), ("D_vs_C", "D vs C")):
        c = hl["comparisons"][k]
        pi = f", P(improvement) {_ci(c['prob_improvement'], 'p')}" if k == "D_vs_B" else ""
        out.append(f"- {nm} (pre-declared): IQM difference {_ci(c['iqm_diff'], 'diff')}{pi}: **{ledger[k]['status']}**.")
    if dg:
        rep = dg["arms"].get("D", {}).get("replay_fraction")
        rho = dg.get("noise_ceiling_spearman_sampled_rollouts") or []
        out.append(f"- Why (exploratory diagnosis in the report): every training scenario is planner-solvable ({_f(dg.get('training_distribution_certified_fraction'), 2)} certified), "
                   f"so regret collapses to the policy's shortfall; ~{(rep or 0) * 100:.0f}% of replay-arm episodes are repeats; scenario parameters explain almost none of what is hard "
                   f"(CV R² {_f(dg.get('cv_r2_survival_from_parameters'), 2)}); the score is noisy (rank correlation {_f(sum(rho) / len(rho) if rho else None, 2)} across sampled rollouts).")
    if par:
        out.append(f"- Parity gate with Stable-Baselines3 {'passed' if par.get('passed') else 'FAILED'}: own PPO IQM survival {_f(par['ci']['own']['iqm'], 1)}, "
                   f"SB3 {_f(par['ci']['sb3']['iqm'], 1)}, curves overlap at {par['curve_overlap_fraction'] * 100:.0f}% of points.")
    if mem:
        preds = mem["predictions"]
        hits = sum(p["status"] == "hit" for p in preds)
        out.append(f"- Memory track: {hits} of {len(preds)} locked predictions hit; the LSTM did not beat the feedforward policy where memory was predicted to help "
                   f"({t['memory_seeds']} seeds vs {t['seeds']}, hyper-parameters borrowed from the feedforward sweep by a dated amendment).")
    if abl:
        hurts = [ABL_LABEL.get(n, n) for n, v in abl["variants"].items() if n != "default" and v["finding"] == "hurts"]
        none = [ABL_LABEL.get(n, n) for n, v in abl["variants"].items() if n != "default" and v["finding"] != "hurts"]
        out.append(f"- Ablations (exploratory, scored on the validation suite that chose each checkpoint; see v2.1 audit below): {_join(hurts)} hurt; {_join(none)} had no measurable effect.")
    lad = hl["ladder"].get("memory_script", {})
    if lad:
        vals = list(lad.values())
        out.append(f"- Scripted baselines are strong: the memory script reaches {min(vals):.1f} to {max(vals):.1f} of the planner on every condition. The baseline ladder is on the site and in the report.")
    out += ["", "Limits: one small custom environment; only this benchmark; a stated hypothesis that lost. Nothing here says regret-guided training is worse in general. "
            "Hardware numbers are from one named 4-core Xeon.", ""]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ v2.1
def _status(d: dict) -> str:
    if d["lo"] > 0:
        return "supported"
    return "not supported (reversed)" if d["hi"] < 0 else "not supported (interval includes zero)"


def v21_block(tier_name: str) -> str:
    tdir = tier_dir(tier_name)
    v = os.path.join(tdir, "v2_1")
    t = load_tier(tier_name)
    out = ["### v2.1 (plan `analysis_plan_v2_1.yaml`; every number below regenerates from `results/full/v2_1/`)", "",
           "Tier labels: **confirmatory** = pre-declared on the fresh suite; **exploratory** = validation suites or descriptive; **diagnostic** = no claim.", ""]
    # confirmatory: mutation arms
    mu = _j(os.path.join(v, "scale", "mutation.json"))
    out.append(f"#### Mutation arms (confirmatory at {t['steps'] / 1e6:g}M, {t['mutation_seeds']} seeds, fresh suite `suite_test_v2_1.json`)")
    if mu:
        out += ["", "| Arm | Aggregate ladder-normalised IQM [95% CI] |", "| --- | --- |"]
        for a, row in mu["arms"].items():
            out.append(f"| {ARM_NAMES.get(a, a)} | {_ci(row['aggregate_iqm'])} |")
        out.append("")
        for k in ("E_vs_B", "E_vs_Eprime"):
            c = mu["comparisons"][k]
            out.append(f"- {k.replace('_', ' ').replace('Eprime', 'E-prime')} ({c['tier']}): IQM difference {_ci(c['iqm_difference'], 'diff')}, "
                       f"P(improvement) {_ci(c['prob_improvement'], 'p')}: **{c['status']}**.")
        out.append(f"- Pre-declared outcome row: **{mu['outcome_text']}**")
        out.append(f"- Headroom rule: {mu['headroom_rule']['text']}")
        for k in ("E_vs_D", "F_vs_D", "F_vs_B"):
            c = mu["comparisons"][k]
            out.append(f"- {k.replace('_', ' ')} (exploratory): {_ci(c['iqm_difference'], 'diff')}.")
    else:
        out.append("\nNot run yet.")
    # budget sweep
    sw = _j(os.path.join(v, "scale", "sweep.json"))
    out += ["", f"#### Budget sweep (exploratory, descriptive; {t['budget_seeds']} seeds)"]
    if sw:
        out += ["", "| Budget | B IQM | D IQM | D minus B |", "| --- | --- | --- | --- |"]
        for b in sw["budgets"]:
            out.append(f"| {b / 1e6:g}M | {_ci(sw['points'][f'B@{b}']['aggregate_iqm'])} | {_ci(sw['points'][f'D@{b}']['aggregate_iqm'])} | {_ci(sw['d_minus_b'][str(b)], 'diff')} |")
        out += ["", f"{sw['outcome_text']} ({sw['claim_status']}.)"]
    else:
        out.append("\nNot run yet.")
    # memory diagnostics
    pr = _j(os.path.join(v, "diagnose", "predictions.json"))
    out += ["", "#### Memory diagnostics (diagnostic and exploratory; validation suites)"]
    im = _j(os.path.join(v, "diagnose", "imitation.json"))
    if im:
        sc = im["script"]["suite_val"]["aggregate"]
        parts = [f"{k}: {_f(c['aggregate_mean']['suite_val'], 2)}" for k, c in im["clones"].items()]
        out.append(f"\nImitation ceiling ({im['episodes']} script episodes, {im['samples']} samples): memory script {_f(sc, 2)}; clones {', '.join(parts)} "
                   "(mean ladder-normalised score, `suite_val.json`).")
        if im.get("dagger"):
            dg = ", ".join(f"{k}: " + " -> ".join(_f(r["scores"]["suite_val"]["aggregate"], 2) for r in rs) for k, rs in im["dagger"].items())
            out.append(f"DAgger-lite rounds (round 0 to {len(next(iter(im['dagger'].values()))) - 1}): {dg}.")
    if pr:
        out += ["", "| Prediction | Criterion | Grade |", "| --- | --- | --- |"]
        for k in ("P1", "P2", "P3"):
            out.append(f"| {k} | {pr[k]['criterion']} | **{pr[k]['grade']}** |")
        out += ["", pr["verdict"]]
    elif not im:
        out.append("\nNot run yet.")
    # SSM / int8 / gamma
    ssm = _j(os.path.join(v, "ssm_int8", "ssm.json"))
    out += ["", "#### SSM (rung 8), int8, gamma (exploratory)"]
    if ssm:
        pc = ssm["params"]
        out.append(f"\nSSM parameters {pc['ssm']:,} vs LSTM {pc['lstm']:,}. " + (
            f"Trained: validation IQM {_ci(ssm['suite_val.json']['aggregate_iqm'])}." if ssm["tune"]["trained"] else ssm["tune"]["verdict"] + "."))
    else:
        out.append("\nSSM: not run yet.")
    q = _j(os.path.join(v, "ssm_int8", "int8.json"))
    if q:
        for name, r in q["rungs"].items():
            out.append(f"- int8 {name}: retention {_ci(r['retention'], 'ratio')} (float32 {_f(r['retention']['float32_iqm'])} vs int8 {_f(r['retention']['int8_iqm'])}), "
                       f"action agreement {_f(r['action_agreement_mean'], 3)}, size {r['size_fp32_bytes'] / 1e3:.0f} kB -> {r['size_int8_bytes'] / 1e3:.0f} kB.")
    else:
        out.append("- int8: not run yet.")
    g = _j(os.path.join(v, "gamma", "gamma.json"))
    if g:
        rows = [k.replace("_", " ") for k, on in g["rows"].items() if on]
        out.append(f"- Gamma re-run with return normalization ({g['seeds']} seeds, {g['steps'] / 1e3:g}k steps): gamma 0.9 penalty {_ci(g['gamma_0.9_penalty']['normalized'], 'diff')} "
                   f"(v2 unnormalized {_ci(g['gamma_0.9_penalty']['v2'], 'diff')}); rows that fired: {_join(rows)}.")
    else:
        out.append("- Gamma re-run: not run yet.")
    # audit + throughput
    au = _j(os.path.join(v, "audit", "selection.json"))
    out += ["", "#### Audit and throughput (diagnostic)", ""]
    if au:
        changed = [ABL_LABEL.get(n, n) for n, e in au["ablation_effects_on_s1"].items()
                   if (e["lo"] > 0 or e["hi"] < 0) != _v2_found_effect(tdir, n)]
        out.append(f"- Selection audit: ablation tables were scored on the suite that chose each checkpoint. Re-scored on `suite_val_s1.json`, the finding changes for {_join(changed)}. "
                   "Both ledger rows are kept, with dates.")
    tp = _j(os.path.join(planmod.REPO, "results", "throughput.json"))
    if tp and "training" in tp:
        tr = tp["training"]
        out.append(f"- Training throughput on {tr['machine']}: {_f(tr['single_job_steps_per_second'], 0)} steps/s for one job, "
                   f"{_f(tr['four_parallel_jobs_total_steps_per_second'], 0)} steps/s aggregate with 4 parallel jobs. {tr['decision']}")
    rs = _j(os.path.join(v, "rescore", "rescore.json"))
    if rs:
        out.append("- Existing arms re-scored on the fresh suite (yardstick, no new claim): " + "; ".join(
            f"{ARM_NAMES.get(k.replace('arm_', ''), k)} {_ci(a['aggregate_iqm'])}" for k, a in rs["arms"].items() if k.startswith("arm_")) + ".")
    out.append("")
    return "\n".join(out)


def _v2_found_effect(tdir: str, name: str) -> bool:
    abl = _j(os.path.join(tdir, "ablations", "ablations.json")) or {}
    return abl.get("variants", {}).get(name, {}).get("finding") == "hurts"


def results_block(tier_name: str = "full") -> str:
    return "\n".join([START, "## Results", "", v21_block(tier_name), "---", "", v2_block(tier_name), END])


def write_readme(tier_name: str = "full", path: Optional[str] = None) -> str:
    path = path or os.path.join(planmod.REPO, "README.md")
    s = open(path).read()
    i, j = s.index(START), s.index(END) + len(END)
    new = s[:i] + results_block(tier_name) + s[j:]
    open(path, "w").write(new)
    return new


# ------------------------------------------------------------------------------------------------ ledger rows
def v21_claims(tier_name: str) -> List[dict]:
    """Ledger rows for v2.1, each with `plan` and `tier`. Pre-declared comparisons get a row with their status."""
    v = os.path.join(tier_dir(tier_name), "v2_1")
    rows: List[dict] = []
    mu = _j(os.path.join(v, "scale", "mutation.json"))
    if mu:
        for k, c in mu["comparisons"].items():
            rows.append({"id": f"v2.1:{k}", "plan": "v2.1", "tier": c["tier"], "pre_declared": c["tier"] == "confirmatory",
                         "status": c["status"], "claim": f"{k.replace('_vs_', ' vs ').replace('Eprime', 'E-prime')} on the fresh suite at 2M, IQM difference across all seven conditions.",
                         "evidence": c["iqm_difference"], "link": "v2_1/scale/mutation.json"})
        rows.append({"id": "v2.1:mutation_outcome", "plan": "v2.1", "tier": "confirmatory", "pre_declared": True,
                     "status": mu["outcome"], "claim": mu["outcome_text"], "evidence": mu["headroom_rule"], "link": "v2_1/scale/mutation.json"})
    sw = _j(os.path.join(v, "scale", "sweep.json"))
    if sw:
        for b in sw["budgets"]:
            rows.append({"id": f"v2.1:sweep_D_minus_B@{b}", "plan": "v2.1", "tier": "exploratory", "pre_declared": False,
                         "status": "descriptive", "claim": f"Arm D minus arm B at {b / 1e6:g}M transitions ({sw['seeds']} seeds; no claim is labeled supported).",
                         "evidence": sw["d_minus_b"][str(b)], "link": "v2_1/scale/sweep.json"})
    pr = _j(os.path.join(v, "diagnose", "predictions.json"))
    if pr:
        for k in ("P1", "P2", "P3"):
            rows.append({"id": f"v2.1:{k}", "plan": "v2.1", "tier": "diagnostic" if k == "P1" else "exploratory", "pre_declared": True,
                         "status": pr[k]["grade"], "claim": pr[k]["criterion"], "evidence": {x: y for x, y in pr[k].items() if x not in ("criterion", "grade")},
                         "link": "v2_1/diagnose/predictions.json"})
    lf = _j(os.path.join(v, "diagnose", "lstm_final.json"))
    if lf:
        d = lf["suite_val_s1.json"]["lstm_minus_ff"]
        rows.append({"id": "v2.1:lstm_retuned_vs_ff", "plan": "v2.1", "tier": "exploratory", "pre_declared": False, "status": _status(d),
                     "claim": "Retuned LSTM minus feedforward arm B on the validation suite not used for selection.", "evidence": d,
                     "link": "v2_1/diagnose/lstm_final.json"})
    g = _j(os.path.join(v, "gamma", "gamma.json"))
    if g:
        for k, on in g["rows"].items():
            rows.append({"id": f"v2.1:gamma:{k}", "plan": "v2.1", "tier": "exploratory", "pre_declared": False,
                         "status": "observed" if on else "not observed", "claim": k.replace("_", " ") + " (return normalization on).",
                         "evidence": g["gamma_0.9_penalty"]["normalized"] if "0.9" in k or "penalty" in k else g["gamma_0.999_gain"]["normalized"],
                         "link": "v2_1/gamma/gamma.json"})
    q = _j(os.path.join(v, "ssm_int8", "int8.json"))
    if q:
        for name, r in q["rungs"].items():
            rows.append({"id": f"v2.1:int8:{name}", "plan": "v2.1", "tier": "exploratory", "pre_declared": False, "status": "descriptive",
                         "claim": f"int8 retention for {name} (ratio of aggregate IQMs; hardware: {q['hardware']}).", "evidence": r["retention"],
                         "link": "v2_1/ssm_int8/int8.json"})
    au = _j(os.path.join(v, "audit", "selection.json"))
    if au:
        for n, e in au["ablation_effects_on_s1"].items():
            rows.append({"id": f"v2.1:audit:ablation:{n}", "plan": "v2.1", "tier": "exploratory", "pre_declared": False,
                         "status": "supported" if e["lo"] > 0 or e["hi"] < 0 else "not supported",
                         "claim": f"Ablation '{n}' re-scored on suite_val_s1.json (the suite that did not choose the checkpoints). The v2 row on the selection suite is kept.",
                         "evidence": e, "link": "v2_1/audit/selection.json"})
    return rows


def tag_v2_claims(claims: List[dict]) -> List[dict]:
    for c in claims:
        c.setdefault("plan", "v2")
        c.setdefault("tier", "confirmatory" if c.get("pre_declared") else "exploratory")
    return claims
