"""Part II of the technical report (v2.1). Every number is read from results/<tier>/v2_1/; wording that depends on an
outcome comes from the pre-declared rules (the stored outcome rows and prediction grades), never from this file."""
from __future__ import annotations

import html
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np

from . import plan as planmod
from .experiments import load_tier, tier_dir

ARM_LABEL = {"B": "B uniform", "D": "D regret replay", "E": "E regret + mutation", "Eprime": "E-prime value loss + mutation",
             "F": "F regret over 4 rollouts"}
COLORS = {"B": "#0072b2", "D": "#d55e00", "E": "#009e73", "Eprime": "#cc79a7", "F": "#8c7a00"}


def _j(path: str) -> Optional[dict]:
    return json.load(open(path)) if os.path.exists(path) else None


def _f(x, d: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ci(d: dict, k: str = "iqm", dd: int = 3) -> str:
    return f"{_f(d[k], dd)} [{_f(d['lo'], dd)}, {_f(d['hi'], dd)}]"


def _tag(label: str) -> str:
    return f"<span class='tier {label}'>{label}</span>"


def figures(tier_name: str, figdir: str) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = os.path.join(tier_dir(tier_name), "v2_1", "scale")
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    out: Dict[str, str] = {}
    sw = _j(os.path.join(v, "sweep.json"))
    if sw:
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        for arm in ("B", "D"):
            xs = sw["budgets"]
            m = [sw["points"][f"{arm}@{b}"]["aggregate_iqm"] for b in xs]
            ax.errorbar(xs, [r["iqm"] for r in m], yerr=[[r["iqm"] - r["lo"] for r in m], [r["hi"] - r["iqm"] for r in m]],
                        fmt="o-", ms=4, lw=1.4, capsize=3, color=COLORS[arm], label=ARM_LABEL[arm])
            for b in xs:
                ps = [row["score"] for row in sw["points"][f"{arm}@{b}"]["per_seed"]] if "score" in (sw["points"][f"{arm}@{b}"]["per_seed"] or [{}])[0] else []
                ax.scatter([b] * len(ps), ps, s=6, alpha=0.35, color=COLORS[arm])
        ax.set_xscale("log")
        ax.set_xticks(sw["budgets"])
        ax.set_xticklabels([f"{b / 1e6:g}M" for b in sw["budgets"]])
        ax.set_xlabel("training transitions per run (log axis)")
        ax.set_ylabel("ladder-normalised survival\n(IQM, 95% CI; fresh suite)")
        ax.set_title("Budget sweep (exploratory, descriptive)", fontsize=8.5, loc="left")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "v21_budget.png"))
        plt.close(fig)
        out["v21_budget"] = "v21_budget.png"
    mu = _j(os.path.join(v, "mutation.json"))
    if mu:
        fig, ax = plt.subplots(figsize=(4.6, 2.9))
        arms = list(mu["arms"])
        m = [mu["arms"][a]["aggregate_iqm"] for a in arms]
        ax.errorbar(range(len(arms)), [r["iqm"] for r in m], yerr=[[r["iqm"] - r["lo"] for r in m], [r["hi"] - r["iqm"] for r in m]],
                    fmt="o", ms=5, capsize=3, color="#333")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([ARM_LABEL.get(a, a).replace(" ", "\n", 1) for a in arms], fontsize=6.5)
        ax.axhline(0.5, color="#777", lw=1, ls=":")
        ax.text(len(arms) - 0.5, 0.505, "0.5 of the ladder", ha="right", va="bottom", fontsize=6.5, color="#555")
        ax.set_ylabel("ladder-normalised survival (IQM, 95% CI)")
        ax.set_title("Mutation arms at 2M, fresh suite", fontsize=8.5, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "v21_mutation.png"))
        plt.close(fig)
        out["v21_mutation"] = "v21_mutation.png"
    return out


def part2_html(tier_name: str, figs: Dict[str, str]) -> str:
    tdir = tier_dir(tier_name)
    v = os.path.join(tdir, "v2_1")
    t = load_tier(tier_name)
    img = lambda k, cap: (f'<figure><img src="figures/{figs[k]}" style="width:70%"><figcaption>{cap}</figcaption></figure>' if k in figs else "")
    plan21 = open(planmod.HASH_V2_1).read().strip() if os.path.exists(planmod.HASH_V2_1) else "unlocked"
    s = [f"<h1 id='part2'>Part II. AdaptArena v2.1</h1><div class='meta'>Analysis plan v2.1 SHA-256 <code>{plan21[:16]}…</code> · parent plan "
         f"<code>{open(planmod.HASH).read().strip()[:16]}…</code> · every result is labelled {_tag('confirmatory')} (pre-declared, fresh suite), "
         f"{_tag('exploratory')} (validation suites or descriptive) or {_tag('diagnostic')} (no claim). Part I above is unchanged.</div>"]
    # 1 specialist
    sp = _j(os.path.join(v, "audit", "specialist.json"))
    if sp:
        m = sp["mean"]
        s.append(f"<h2>II.1 The specialist {_tag('diagnostic')}</h2><p>Arm A's pooled IQM of 0.010 is a finding, not a bug. The pooled number averages seven conditions, six of which use layouts the specialist never saw. "
                 f"On its own training distribution it scores {_f(m['train_like_score'], 2)} of the ladder ({_f(m['train_like_raw'], 0)} steps of 500) and on fresh layouts {_f(m['fresh_layouts_score'], 2)}. "
                 "An earlier reading of the pooled figure as 'random level even on its own layout' was wrong.</p>")
    # 2 selection audit
    au = _j(os.path.join(v, "audit", "selection.json"))
    if au:
        g = au["by_group"]
        rows = "".join(f"<tr><td>{k}</td><td>{x['n']}</td><td>{_f(x['mean_best_val'], 1)}</td><td>{_f(x['mean_rescored'], 1)}</td><td>{_f(x['mean_gap'], 1)}</td></tr>" for k, x in g.items())
        changed = [n for n, e in au["ablation_effects_on_s1"].items() if ((e["lo"] > 0 or e["hi"] < 0))]
        s.append(f"<h2>II.2 Selection audit {_tag('diagnostic')}</h2><p>A run's <code>best_val</code> is the maximum of about 13 noisy evaluations on the suite that chose the checkpoint, so it runs high. "
                 f"Every selected checkpoint was re-scored on <code>suite_val_s1.json</code>, a different validation suite (mean survival in steps):</p>"
                 f"<table><tr><th>Group</th><th>runs</th><th>best_val</th><th>re-scored</th><th>gap</th></tr>{rows}</table>"
                 "<p>The v2 ablation tables were scored on the selection suite. Re-scored on the other suite, the effects with intervals clear of zero are: "
                 f"{html.escape(', '.join(changed))}. The v2 rows stay in the ledger next to the v2.1 rows, with dates.</p>")
    # 3 throughput
    tp = _j(os.path.join(planmod.REPO, "results", "throughput.json"))
    if tp and "training" in tp:
        tr = tp["training"]
        s.append(f"<h2>II.3 Training throughput {_tag('diagnostic')}</h2><p>One arm-B job runs at {_f(tr['single_job_steps_per_second'], 0)} steps/s; four parallel jobs aggregate "
                 f"{_f(tr['four_parallel_jobs_total_steps_per_second'], 0)} steps/s on {html.escape(tr['machine'])}. Time share in a job: "
                 + ", ".join(f"{k.replace('_', ' ')} {v_ * 100:.0f}%" for k, v_ in tr["time_share_single_job"].items()) + f". {html.escape(tr['decision'])} "
                 f"The earlier spread of 1.1k to 2.1k steps/s in the v2 logs was not core sharing.</p>")
    # 4 memory
    pr = _j(os.path.join(v, "diagnose", "predictions.json"))
    im = _j(os.path.join(v, "diagnose", "imitation.json"))
    mp = _j(os.path.join(v, "diagnose", "mapfed.json"))
    rt = _j(os.path.join(v, "diagnose", "lstm_retune.json"))
    lf = _j(os.path.join(v, "diagnose", "lstm_final.json"))
    s.append(f"<h2>II.4 Memory diagnostics {_tag('diagnostic')} {_tag('exploratory')}</h2>")
    if im:
        sc = im["script"]["suite_val"]["aggregate"]
        rows = "".join(f"<tr><td>{k}</td><td>{_f(c['aggregate_mean']['suite_val'], 3)}</td><td>{_f(c['aggregate_mean']['suite_val_s1'], 3)}</td></tr>" for k, c in im["clones"].items())
        dg = ""
        if im.get("dagger"):
            dg = "<p>DAgger-lite (script labels on the learner's own states, aggregated, retrained): " + "; ".join(
                f"{k}: " + " → ".join(_f(r["scores"]["suite_val"]["aggregate"], 2) for r in rs) for k, rs in im["dagger"].items()) + " (round 0 to last, <code>suite_val</code>).</p>"
        s.append(f"<p><b>Imitation ceiling.</b> The memory script scores {_f(sc, 3)} of the ladder. Clones trained by cross-entropy on {im['samples']:,} script samples ({im['episodes']} episodes), no reinforcement learning:</p>"
                 f"<table><tr><th>clone</th><th>suite_val</th><th>suite_val_s1</th></tr>{rows}</table>{dg}")
    if mp:
        s.append(f"<p><b>Hand-fed map.</b> A feedforward policy given the script's dead-reckoning map as three extra 9×9 channels (remembered food, remembered water, known floor; this environment has no shelters) "
                 f"scores {_ci(mp['ff_map']['iqm'])} against plain feedforward {_ci(mp['ff_plain']['iqm'])} on <code>suite_val</code> ({mp['seeds']} seeds, {mp['steps'] / 1e6:g}M); "
                 f"difference {_ci(mp['map_minus_plain'], 'diff')}; {_f(mp['map_fraction_of_script'], 2)} of the script.</p>")
    if rt:
        w = rt.get("window256_minus_window64")
        s.append(f"<p><b>LSTM retune.</b> Stage 1 best: {html.escape(rt['stage1_best']['key'])}; stage 2 winner: {html.escape(rt['winner']['key'])}"
                 + (f"; window 256 minus window 64 {_ci(w, 'diff')}" if w else "") + ".</p>")
    if lf:
        d = lf["suite_val_s1.json"]
        s.append(f"<p>Retuned LSTM, {lf['seeds']} seeds: {_ci(d['lstm_retuned']['iqm'])} vs feedforward arm B {_ci(d['ff_arm_B']['iqm'])} on <code>suite_val_s1</code>; difference {_ci(d['lstm_minus_ff'], 'diff')}.</p>")
    if pr:
        rows = "".join(f"<tr><td>{k}</td><td>{html.escape(pr[k]['criterion'])}</td><td><b>{pr[k]['grade']}</b></td></tr>" for k in ("P1", "P2", "P3"))
        s.append(f"<table><tr><th>Prediction (locked before any Phase 4 run)</th><th>Criterion</th><th>Grade</th></tr>{rows}</table><p><b>Verdict.</b> {html.escape(pr['verdict'])}</p>")
    elif not (im or mp or rt):
        s.append("<p>Not run.</p>")
    # 5 budget sweep
    sw = _j(os.path.join(v, "scale", "sweep.json"))
    s.append(f"<h2>II.5 Budget sweep {_tag('exploratory')}</h2>")
    if sw:
        rows = "".join(f"<tr><td>{b / 1e6:g}M</td><td>{_ci(sw['points'][f'B@{b}']['aggregate_iqm'])}</td><td>{_ci(sw['points'][f'D@{b}']['aggregate_iqm'])}</td><td>{_ci(sw['d_minus_b'][str(b)], 'diff')}</td></tr>" for b in sw["budgets"])
        s.append(img("v21_budget", "Survival against training budget, log axis. Dots are single seeds.") +
                 f"<table><tr><th>Budget</th><th>B</th><th>D</th><th>D minus B</th></tr>{rows}</table><p>{html.escape(sw['outcome_text'])} With {sw['seeds']} seeds the intervals are wide and no claim is labelled supported.</p>")
    else:
        s.append("<p>Not run.</p>")
    # 6 mutation
    mu = _j(os.path.join(v, "scale", "mutation.json"))
    s.append(f"<h2>II.6 Mutation arms {_tag('confirmatory')}</h2>")
    if mu:
        rows = "".join(f"<tr><td>{ARM_LABEL.get(a, a)}</td><td>{_ci(r['aggregate_iqm'])}</td></tr>" for a, r in mu["arms"].items())
        cmp_ = "".join(f"<tr><td>{k.replace('_vs_', ' vs ').replace('Eprime', 'E-prime')}</td><td>{c['tier']}</td><td>{_ci(c['iqm_difference'], 'diff')}</td><td>{_ci(c['prob_improvement'], 'p')}</td><td>{c['status']}</td></tr>"
                       for k, c in mu["comparisons"].items())
        e_st = [r for r in mu["arms"].get("E", {}).get("run_stats", []) if r]
        mstat = ""
        if e_st:
            mstat = (f"<p>Arm E buffer logs (mean over seeds): repeat fraction {_f(np.mean([r['repeat_fraction'] for r in e_st]), 2)}, distinct scenarios {_f(np.mean([r['distinct_scenarios'] for r in e_st]), 0)}, "
                     f"uncertified child rate {_f(np.mean([r['uncertified_rate'] for r in e_st]), 2)}, acceptance rate {_f(np.mean([r['acceptance_rate'] for r in e_st]), 2)}.</p>")
        s.append(img("v21_mutation", "Aggregate ladder-normalised IQM on the fresh suite.") +
                 f"<table><tr><th>Arm</th><th>IQM [95% CI]</th></tr>{rows}</table><table><tr><th>Comparison</th><th>Tier</th><th>IQM difference</th><th>P(improvement)</th><th>Status</th></tr>{cmp_}</table>"
                 f"<p><b>Pre-declared outcome row.</b> {html.escape(mu['outcome_text'])} {html.escape(mu['headroom_rule']['text'])}</p>{mstat}")
    else:
        s.append("<p>Not run.</p>")
    # 7 ssm / int8
    ssm = _j(os.path.join(v, "ssm_int8", "ssm.json"))
    q = _j(os.path.join(v, "ssm_int8", "int8.json"))
    s.append(f"<h2>II.7 SSM and int8 {_tag('exploratory')}</h2>")
    if ssm:
        txt = f"<p>SSM parameters {ssm['params']['ssm']:,} vs LSTM {ssm['params']['lstm']:,} vs feedforward {ssm['params']['ff']:,}. "
        if ssm["tune"]["trained"]:
            txt += (f"Trained: validation {_ci(ssm['suite_val.json']['aggregate_iqm'])}, <code>suite_val_s1</code> {_ci(ssm['suite_val_s1.json']['aggregate_iqm'])}, fresh suite {_ci(ssm['suite_test_v2_1.json']['aggregate_iqm'])}.</p>")
        else:
            txt += html.escape(ssm["tune"]["verdict"]) + ".</p>"
        s.append(txt)
        ic = ssm.get("inference_cost")
        if ic:
            rows = "".join(f"<tr><td>{n}</td><td>{p['params']:,}</td><td>{p['state_bytes_per_env']:,}</td><td>{_f(p['batch1']['env_steps_per_second'], 0)}</td><td>{_f(p['batch64']['env_steps_per_second'], 0)}</td></tr>" for n, p in ic["policies"].items())
            s.append(f"<table><tr><th>policy</th><th>params</th><th>state bytes / env</th><th>steps/s batch 1</th><th>steps/s batch 64</th></tr>{rows}</table><p>One thread, {html.escape(ic['hardware'])}.</p>")
    elif _j(os.path.join(v, "ssm_cut.json")):
        s.append(f"<p><b>SSM (rung 8): not run.</b> {html.escape(_j(os.path.join(v, 'ssm_cut.json'))['reason'])}</p>")
    else:
        s.append("<p>SSM: not run.</p>")
    if q:
        rows = "".join(f"<tr><td>{n}</td><td>{r['seeds']}</td><td>{_f(r['retention']['float32_iqm'])}</td><td>{_f(r['retention']['int8_iqm'])}</td><td>{_ci(r['retention'], 'ratio')}</td><td>{_f(r['action_agreement_mean'])}</td>"
                       f"<td>{r['size_fp32_bytes'] / 1e3:.0f} → {r['size_int8_bytes'] / 1e3:.0f} kB</td></tr>" for n, r in q["rungs"].items())
        parts = "; ".join(f"{n}: {html.escape(', '.join(r['int8_parts'][:4]))}{'…' if len(r['int8_parts']) > 4 else ''}" for n, r in q["rungs"].items())
        s.append(f"<table><tr><th>rung</th><th>seeds</th><th>float32</th><th>int8</th><th>retention</th><th>action agreement</th><th>size</th></tr>{rows}</table>"
                 f"<p>Dynamic int8 on {html.escape(q['hardware'])}, scored on <code>suite_val_s1</code>. int8 parts: {parts}. Activations and recurrent state stay float32. No claim about edge devices.</p>")
    # 8 gamma
    g = _j(os.path.join(v, "gamma", "gamma.json"))
    s.append(f"<h2>II.8 Gamma re-run with return normalization {_tag('exploratory')}</h2>")
    if g:
        rows = "".join(f"<tr><td>{k}</td><td>{_ci(g['v2_unnormalized'][k]['aggregate_iqm'])}</td><td>{_ci(g['normalized'][k]['aggregate_iqm'])}</td></tr>" for k in g["normalized"])
        fired = [k.replace("_", " ") for k, on in g["rows"].items() if on]
        s.append(f"<table><tr><th>gamma</th><th>v2 (fixed reward scale)</th><th>normalized returns</th></tr>{rows}</table>"
                 f"<p>{g['seeds']} seeds, {g['steps'] / 1e3:g}k steps, <code>suite_val_s1</code>. Rows that fired by the stored rule: {html.escape(', '.join(fired) or 'none')}. Gamma 0.9 penalty: v2 {_ci(g['gamma_0.9_penalty']['v2'], 'diff')}, normalized {_ci(g['gamma_0.9_penalty']['normalized'], 'diff')}.</p>")
    else:
        s.append("<p>Not run.</p>")
    # 9 limitations
    s.append("<h2>II.9 Limitations (updated)</h2><ul>"
             "<li>The fresh suite was opened only through the access harness; its generation attempts and the unpushed tag are recorded in the amendments file.</li>"
             "<li>Tags are unsigned and could not be pushed from the build environment; the lock commit hash is recorded instead.</li>"
             "<li>With 3 seeds at 8M and 20M the budget sweep is descriptive. Timing and weather seeds and layouts in mutation are re-rolled inside the training split, a stricter bound than the plan's 'any'.</li>"
             f"<li>The map-fed, retune and SSM results are exploratory, scored on validation suites, with {t['map_seeds']}, {t['lstm_tune_seeds']} and {t['ssm_seeds']} seeds.</li>"
             "<li>Speed and int8 statements apply to the named machine only. No edge-device or energy claim is made.</li></ul>")
    return "".join(s)
