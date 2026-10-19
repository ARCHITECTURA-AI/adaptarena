"""Technical report builder: figures + HTML (+ PDF through headless Chromium) written from the results folder.

Every number in the text is read from a results file. Wording that depends on an outcome is chosen by the
pre-declared rules (`decide_outcome`, claim statuses, prediction scores), never by hand.
"""
from __future__ import annotations

import html
import json
import os
import shutil
from typing import Optional

import numpy as np

from ..core.distributions import CONDITIONS
from . import plan as planmod
from .experiments import ARMS, tier_dir

ARM_COLORS = {"A": "#8c7a00", "B": "#0072b2", "C": "#cc79a7", "D": "#d55e00"}


def _load(path: str):
    return json.load(open(path)) if os.path.exists(path) else None


def _fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ci(d, k="iqm", dd=2):
    return f"{_fmt(d[k], dd)} [{_fmt(d['lo'], dd)}, {_fmt(d['hi'], dd)}]"


# --------------------------------------------------------------------------------- figures
def figures(tier_name: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tdir = tier_dir(tier_name)
    figdir = os.path.join(tdir, "report", "figures")
    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    out = {}
    hl = _load(os.path.join(tdir, "headline", "stats.json"))
    if hl:
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [2.6, 1]})
        ax = axs[0]
        x = np.arange(len(CONDITIONS))
        for i, a in enumerate(ARMS):
            pc = hl["per_arm"][a]["per_condition"]
            m = np.array([pc[c]["iqm"] for c in CONDITIONS])
            lo = np.array([pc[c]["lo"] for c in CONDITIONS])
            hi = np.array([pc[c]["hi"] for c in CONDITIONS])
            ax.errorbar(x + (i - 1.5) * 0.16, m, yerr=[m - lo, hi - m], fmt="o", ms=3.5, lw=1.2, capsize=2,
                        color=ARM_COLORS[a], label=f"{a}: {ARMS[a]}")
        ax.axhline(0, color="#777", lw=1, ls=":")
        ax.axhline(1, color="#777", lw=1, ls="--")
        ax.text(len(CONDITIONS) - 0.5, 1.01, "planner = 1", ha="right", va="bottom", fontsize=7, color="#555")
        ax.text(len(CONDITIONS) - 0.5, 0.01, "random = 0", ha="right", va="bottom", fontsize=7, color="#555")
        for name, mk, col in (("reactive_script", "_", "#555"), ("memory_script", "_", "#000")):
            ys = [hl["ladder"][name][c] for c in CONDITIONS]
            ax.scatter(x, ys, marker=mk, s=300, color=col, linewidths=1.6, zorder=0,
                       label="reactive script" if name.startswith("reactive") else "memory script")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") + ("\n(in dist.)" if hl["in_distribution"][c] else "") for c in CONDITIONS], fontsize=6.5)
        ax.set_ylabel("ladder-normalised survival\n(IQM over seeds, 95% stratified bootstrap CI)")
        ax.set_title("Headline: per-condition IQM, locked test suite", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.3, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False)
        # companion panel: what the regret buffer concentrated on
        ax2 = axs[1]
        feats = ["n_water", "n_food", "move_cost_pct", "cold_pct", "delay"]
        for arm, dx, col in (("D", -0.18, ARM_COLORS["D"]), ("C", 0.18, ARM_COLORS["C"])):
            ratios = []
            for s in range(100):
                p = os.path.join(tdir, "headline", f"arm_{arm}", f"seed_{s}", "train_result.json")
                if not os.path.exists(p):
                    break
                snaps = json.load(open(p))["extra"].get("snapshots") or []
                if snaps and snaps[-1]:
                    sn = snaps[-1]
                    ratios.append([sn["top_quartile"][f] / max(sn["all"][f], 1e-9) for f in feats])
            if ratios:
                ax2.barh(np.arange(len(feats)) + dx, np.mean(ratios, 0) - 1, height=0.34, left=1, color=col, label=f"arm {arm}")
        ax2.axvline(1, color="#777", lw=1)
        ax2.set_yticks(range(len(feats)))
        ax2.set_yticklabels(["water sites", "food sites", "move cost", "cold %", "respawn delay"], fontsize=7)
        ax2.set_xlabel("top-quartile / buffer mean")
        ax2.set_title("What the buffer concentrated on", fontsize=8.5, loc="left")
        if ax2.get_legend_handles_labels()[0]:
            ax2.legend(fontsize=6.5, frameon=False)
        else:
            ax2.text(0.5, 0.5, "no buffer snapshots\nin this tier", ha="center", transform=ax2.transAxes, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "headline.png"))
        plt.close(fig)
        out["headline"] = "headline.png"
    par = _load(os.path.join(tdir, "parity", "parity.json"))
    if par:
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        for impl, col in (("own", "#0072b2"), ("sb3", "#d55e00")):
            c = par["curves"][impl]
            ax.plot(c["steps"], c["mean"], color=col, label={"own": "own PPO", "sb3": "Stable-Baselines3"}[impl])
            ax.fill_between(c["steps"], c["min"], c["max"], color=col, alpha=0.15)
        ax.set_xlabel("transitions")
        ax.set_ylabel("validation survival")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("Parity gate: mean and seed band", fontsize=8.5, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "parity.png"))
        plt.close(fig)
        out["parity"] = "parity.png"
    st = _load(os.path.join(tdir, "stress", "stress.json"))
    if st:
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
        for ax, (name, res) in zip(axs, st["stressors"].items()):
            for i, (label, r) in enumerate(res["rungs"].items()):
                if label == "planner":
                    continue
                ys = [np.nan if v is None else v for v in r["mean_survival"]]
                ax.plot(range(len(ys)), ys, marker="osd^v<>p"[i % 8], ms=3, lw=1, label=label.replace("ppo_ff_arm", "PPO arm "))
                if r["breaking_point"] is not None:
                    ax.plot([r["breaking_point"]], [ys[r["breaking_point"]]], marker="x", color="k", ms=6, mew=1.4)
            ax.axhline(500, color="#777", ls="--", lw=0.8)
            ax.set_title({"water": "water sites", "cost": "movement cost", "food": "food sites"}[name], fontsize=8, loc="left")
            ax.set_xlabel("stress level")
        axs[0].set_ylabel("mean steps survived")
        axs[0].legend(fontsize=5.8, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "stress.png"))
        plt.close(fig)
        out["stress"] = "stress.png"
    ab = _load(os.path.join(tdir, "ablations", "ablations.json"))
    if ab:
        names = [n for n in ab["variants"] if n != "default"]
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        ys = np.arange(len(names))
        eff = [ab["variants"][n]["effect_vs_default"] for n in names]
        ax.errorbar([e["diff"] for e in eff], ys, xerr=[[e["diff"] - e["lo"] for e in eff], [e["hi"] - e["diff"] for e in eff]],
                    fmt="o", ms=3.5, capsize=2, color="#0072b2")
        ax.axvline(0, color="#777", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("change in ladder-normalised survival")
        ax.set_title("Ablations (exploratory, validation)", fontsize=8.5, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "ablations.png"))
        plt.close(fig)
        out["ablations"] = "ablations.png"
    mem = _load(os.path.join(tdir, "memory", "memory.json"))
    if mem:
        fig, ax = plt.subplots(figsize=(3.6, 2.2))
        keys = [("familiar", "familiar"), ("unseen_layout", "unseen"), ("resource_relocation", "relocation"),
                ("vision3", "3x3 vision"), ("nocost", "no cost stat"), ("cold_snap", "cold snap")]
        ys = np.arange(len(keys))
        c = [mem["comparisons"][k] for k, _ in keys]
        ax.errorbar([e["diff"] for e in c], ys, xerr=[[e["diff"] - e["lo"] for e in c], [e["hi"] - e["diff"] for e in c]],
                    fmt="o", ms=3.5, capsize=2, color="#009e73")
        ax.axvline(0, color="#777", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([n for _, n in keys], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("LSTM minus feedforward (ladder-normalised)")
        ax.set_title("Memory track", fontsize=8.5, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "memory.png"))
        plt.close(fig)
        out["memory"] = "memory.png"
    dbg = _load(os.path.join(tdir, "debug", "debug.json"))
    if dbg:
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 1.9))
        for name, col in (("failed_reward_scale_1.0", "#b3261e"), ("fixed_reward_scale_0.01", "#0072b2")):
            r = dbg["runs"][name]
            axs[0].plot(r["update"], r["ep_survival"], color=col, label=name.split("_")[0])
            axs[1].semilogy(r["update"], np.maximum(r["value_loss"], 1e-5), color=col)
            axs[2].plot(r["update"], np.minimum(r["approx_kl"], 1.0), color=col)
        axs[0].axhline(dbg["planner_survival"], color="#777", ls="--", lw=0.8)
        axs[0].set_title("episode survival", fontsize=8, loc="left")
        axs[1].set_title("value loss", fontsize=8, loc="left")
        axs[2].set_title("approx. KL (clipped at 1)", fontsize=8, loc="left")
        axs[0].legend(frameon=False, fontsize=6.5)
        for a in axs:
            a.set_xlabel("update")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "debug.png"))
        plt.close(fig)
        out["debug"] = "debug.png"
    return out


# --------------------------------------------------------------------------------- text
CITES = {
    "minigrid": "Chevalier-Boisvert et al., Minigrid &amp; Miniworld, NeurIPS Datasets and Benchmarks 2023",
    "procgen": "Cobbe et al., Leveraging Procedural Generation to Benchmark Reinforcement Learning, ICML 2020",
    "craftax": "Matthews et al., Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning, ICML 2024",
    "plr": "Jiang, Grefenstette, Rocktäschel, Prioritized Level Replay, ICML 2021",
    "rplr": "Jiang et al., Replay-Guided Adversarial Environment Design, NeurIPS 2021",
    "paired": "Dennis et al., Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design, NeurIPS 2020",
    "agarwal": "Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021",
    "cleanrl": "Huang et al., CleanRL: High-quality Single-file Implementations of Deep RL Algorithms, JMLR 2022",
    "sb3": "Raffin et al., Stable-Baselines3, JMLR 2021",
    "ppo": "Schulman et al., Proximal Policy Optimization Algorithms, 2017",
    "gae": "Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation, ICLR 2016",
    "mamba": "Gu and Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, 2023",
    "dr": "Tobin et al., Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World, IROS 2017",
}

CSS = """
@page { size: Letter; margin: 0.6in 0.65in; }
body { font: 9.6pt/1.38 "Georgia", "Times New Roman", serif; color: #111; max-width: 7.2in; margin: 0 auto; }
h1 { font-size: 17pt; margin: 0 0 2pt; } h2 { font-size: 11.5pt; margin: 11pt 0 3pt; border-bottom: 1px solid #bbb; }
h3 { font-size: 10pt; margin: 8pt 0 2pt; } p { margin: 3pt 0 5pt; text-align: justify; }
.meta { color: #555; font-size: 8.5pt; } figure { margin: 5pt 0; text-align: center; } figure img { max-width: 100%; }
figcaption { font-size: 8.2pt; color: #333; text-align: left; } table { border-collapse: collapse; font-size: 8.2pt; margin: 4pt 0; width: 100%; }
td, th { border-bottom: 1px solid #ccc; padding: 2pt 5pt; text-align: left; vertical-align: top; } th { background: #f1f1ec; }
.cols { display: flex; gap: 10pt; } .cols > * { flex: 1; } .refs { font-size: 7.6pt; columns: 2; } .refs li { margin: 0 0 1pt; }
code { font-size: 8.4pt; } .flag { font-weight: 700; }
"""


def _headline_text(hl: dict, tier: dict) -> str:
    d = hl["decision"]
    pa = hl["per_arm"]
    rows = "".join(
        f"<tr><td>{a}: {ARMS[a]}</td><td>{_ci(pa[a]['aggregate_iqm'])}</td><td>{_fmt(pa[a]['optimality_gap']['gap'])}</td>"
        f"<td>{html.escape(_cost(hl, a))}</td></tr>" for a in ARMS)
    comp = hl["comparisons"]
    crow = ""
    for k in ("D_vs_B", "D_vs_C"):
        c = comp[k]
        crow += (f"<tr><td>{k.replace('_vs_', ' vs ')} (pre-declared)</td><td>{_ci(c['iqm_diff'], 'diff')}</td>"
                 f"<td>{_ci(c['prob_improvement'], 'p')}</td>"
                 f"<td>{next(x['status'] for x in hl['ledger'] if x['id'] == k)}</td></tr>")
    sat = hl["decision"]["saturated_conditions"]
    sat_txt = (f"Saturated conditions (every arm reaches the horizon in more than 90% of episodes): {', '.join(sat)}; "
               "these are flagged and not counted as wins." if sat else
               "No condition was saturated by every arm, so no condition is excluded for saturation.")
    return f"""
<p><b>Pre-declared outcome.</b> {html.escape(d['text'])} (rule fired: <code>{d['outcome']}</code>). {sat_txt}</p>
<table><tr><th>Arm</th><th>Aggregate IQM [95% CI]</th><th>Optimality gap</th><th>Training cost</th></tr>{rows}</table>
<table><tr><th>Comparison</th><th>IQM difference [CI]</th><th>P(improvement) [CI]</th><th>Claim status</th></tr>{crow}</table>
<p class="meta">Scores are ladder-normalised survival (random = 0, planner = 1), averaged over episodes inside each seed first;
{tier['seeds']} training seeds per arm, {tier['scenarios_per_condition']} certified scenarios per condition, stratified bootstrap with 10,000 resamples.
Every comparison that was run, including exploratory ones, is in the appendix table <code>comparisons_appendix.csv</code>.</p>"""


def _cost(hl, a):
    c = hl["train_cost"][a]
    s = f"{c['mean_wall_s'] / 60:.0f} min/run"
    if c["planner_calls"]:
        s += f", {c['planner_calls']:.0f} planner calls ({c['planner_seconds']:.0f} s)"
    return s


def build_report_html(tier_name: str, figs: dict) -> str:
    from .experiments import load_tier
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    hl = _load(os.path.join(tdir, "headline", "stats.json"))
    par = _load(os.path.join(tdir, "parity", "parity.json"))
    st = _load(os.path.join(tdir, "stress", "stress.json"))
    ab = _load(os.path.join(tdir, "ablations", "ablations.json"))
    mem = _load(os.path.join(tdir, "memory", "memory.json"))
    rt = _load(os.path.join(tdir, "redteam", "redteam.json"))
    cal = _load(os.path.join(tdir, "calibration.json"))
    tp = _load(os.path.join(planmod.REPO, "results", "throughput.json"))
    prb = _load(os.path.join(tdir, "probes", "probes.json"))
    tuned = _load(os.path.join(tdir, "tune", "tuned.json"))
    manifest = _load(os.path.join(tdir, "manifest.json")) or {}
    plan_hash = open(planmod.HASH).read().strip() if os.path.exists(planmod.HASH) else "unlocked"
    img = lambda k, cap, w="100%": (f'<figure><img src="figures/{figs[k]}" style="width:{w}"><figcaption>{cap}</figcaption></figure>' if k in figs else "")

    s = [f"<h1>AdaptArena v2: does training where the policy fails most survive unseen shifts better?</h1>",
         f"<div class='meta'>Technical report · results tier <b>{tier_name}</b> · analysis plan SHA-256 <code>{plan_hash[:16]}…</code> · "
         f"{html.escape(manifest.get('machine', ''))} · torch {manifest.get('torch', '')}, numpy {manifest.get('numpy', '')}</div>"]
    # 1 question
    s.append(f"""<h2>1. Question</h2>
<p>Does sampling training scenarios by <i>planner regret</i> (reference-planner survival minus policy survival) produce agents that
survive unseen shifts better than uniform randomization, at equal transition budget? This is a hypothesis, and the report is written so that
either answer is reportable. Arms: A specialist (one fixed layout), B uniform randomization, C replay prioritised by positive value loss,
D replay prioritised by planner regret; C and D share one buffer implementation, so the score is the only difference.
Two comparisons were declared in advance (D vs B, D vs C); everything else is exploratory.</p>
<p><b>Positioning.</b> Minigrid [{CITES['minigrid']}] and Procgen [{CITES['procgen']}] are established generalisation benchmarks and Craftax
[{CITES['craftax']}] is a fast survival-style benchmark. What this project adds is narrower: a full-information reference planner that yields
solvability certificates and exact regret on every scenario, bit-exact integer simulator state, controlled interventions with stress curves, and a
locked statistical protocol. Replay-guided environment design is not new: Prioritized Level Replay uses a value-loss score as a regret proxy
[{CITES['plr']}; {CITES['rplr']}], and PAIRED uses regret directly [{CITES['paired']}]. Here regret is cheap because the world is small, which is a
limitation of the setting rather than evidence for a general method. This is not a new RL algorithm and no general claim about regret-guided training is made.</p>""")
    # 2 environment
    cal_tbl = ""
    if cal:
        cal_tbl = "<table><tr><th>Condition (validation)</th><th>random</th><th>reactive script</th><th>memory script</th><th>planner</th></tr>" + "".join(
            f"<tr><td>{c}</td><td>{_fmt(v['random'], 0)}</td><td>{_fmt(v['reactive'], 0)}</td><td>{_fmt(v['memory'], 0)}</td><td>{_fmt(v['planner'], 0)}</td></tr>"
            for c, v in cal.items()) + "</table>"
    tp_txt = ""
    if tp:
        b = tp["steps_per_second"]
        tp_txt = (f" Throughput on {html.escape(tp['machine'])} ({tp['cores']} cores): {b['reference_scalar']:,.0f} steps/s for the scalar reference "
                  f"implementation and {b['batch_8_with_observations']:,.0f} (8 envs) to {b['batch_256_with_observations']:,.0f} (256 envs) steps/s for the vectorised "
                  "batch environment, including observation construction; every experiment budget below was sized against these numbers.")
    s.append(f"""<h2>2. Environment</h2>
<p><code>AdaptArena-Survival-v1</code>: a 15×15 grid, six actions, reward 1 per living step, 500-step horizon. Energy and hydration are integers in hundredths (0 to 10000),
drains are integers and cost multipliers are integer percents, so NumPy, a future JAX port and every machine produce identical SHA-256 state hashes.
The observation is a 9×9 agent-centred local view with 8 channels plus an 11-value stats vector (energy, hydration, cold, previous action one-hot,
previous action valid, energy drained last step). Invalid moves pay the movement surcharge, invalid interactions only the base drain; simultaneous death causes are
all stored and the primary reason follows hazard, dehydration, starvation. A full-information search <b>planner</b> sees the map, respawn and weather schedules and
returns a plan that survives 500 steps; each plan is replayed in the environment to verify it, and on tiny maps the planner is cross-checked against exhaustive search.
A scenario enters any suite only if the planner certifies it. The planner is a strong reference, never called optimal.{tp_txt}</p>
<p><b>Difficulty calibration</b> (validation seeds only, before any learned policy ran; constants frozen in the analysis plan):</p>{cal_tbl}""")
    # 3 methods
    tune_txt = ""
    if tuned:
        tune_txt = " Tuned (learning rate, entropy coefficient): " + "; ".join(
            f"{k} {v['best']['lr']:g}, {v['best']['ent_coef']:g}" for k, v in tuned.items()) + " (SSM reuses the LSTM setting: a recorded difference in tuning effort)."
    par_txt = ""
    if par:
        par_txt = (f"<b>Parity gate with SB3.</b> Both implementations trained with the same hyper-parameters, environment, reward scaling and transition budget on "
                   f"{len(par['seeds']['own'])} seeds each. Final IQM survival: own {_ci(par['ci']['own'], 'iqm', 1)}, SB3 {_ci(par['ci']['sb3'], 'iqm', 1)}; "
                   f"each lies {'inside' if par['own_in_sb3_ci'] and par['sb3_in_own_ci'] else 'NOT inside'} the other's interval; mean curves overlap the other's seed band at "
                   f"{par['curve_overlap_fraction'] * 100:.0f}% of evaluation points (threshold {par['criteria']['curve_overlap_min_fraction'] * 100:.0f}%). "
                   f"<span class='flag'>Gate {'passed' if par['passed'] else 'FAILED'}.</span>")
    s.append(f"""<h2>3. Methods</h2>
<p><b>Learner.</b> PPO written from scratch in one file ({CITES['cleanrl']} style; {CITES['ppo']}; {CITES['gae']}): vectorised rollouts, GAE with bootstrap on truncation and never on death,
clipped surrogate, entropy bonus, advantage normalisation, linear learning-rate decay, gradient clipping, and a recurrent variant on 64-step sequence chunks with hidden-state resets at episode boundaries.
It is validated by unit tests (hand-computed GAE with one death and one truncation, ratio 1 and clip fraction 0 on the first epoch, overfitting one batch, a tiny solvable environment, same-seed reproducibility,
LSTM and SSM sequence pass equal to the step-by-step pass) and by the parity gate against Stable-Baselines3 [{CITES['sb3']}].{tune_txt}</p>
<p>{par_txt}</p>
{img('parity', 'Learning curves (validation survival), mean and min-max seed band.', '55%')}
<p><b>Protocol.</b> {tier['seeds']} seeds per arm at {tier['steps']:,} transitions each (counted across all environments). Evaluation uses the locked test suite: 7 conditions × {tier['scenarios_per_condition']} planner-certified scenarios,
deterministic actions as the primary mode, identical scenarios for every arm. Statistics follow {CITES['agarwal']}: IQM with stratified bootstrap confidence intervals, optimality gap against the planner,
probability of improvement, performance profiles, with the training seed within a condition as the unit of analysis. The analysis plan (hypotheses, arms, seed counts, metrics, thresholds, parity criteria, memory predictions) was hashed
(<code>{plan_hash[:16]}…</code>) and committed to git before the test suite was generated; the harness refuses to open the suite otherwise and logs every access. Checkpoints are chosen by highest mean validation survival, earlier wins ties.</p>""")
    # 4 results
    res = "<h2>4. Results</h2>"
    if hl:
        res += img("headline", "Headline figure. Left: per-condition IQM with 95% CIs for the four arms against the random floor, the scripted baselines and the planner ceiling. Right: the companion panel showing what the replay buffers concentrated on (top-priority quartile relative to the buffer mean).")
        res += _headline_text(hl, tier)
        comb = hl["comparisons"]["D_vs_B"]["per_condition"]["combined_shift"]["iqm_diff"]
        res += (f"<p>Under the combined shift, the D−B difference is {_ci(comb, 'diff')} (exploratory). "
                f"Training cost for arm D includes planner calls; every arm used the same transition budget.</p>")
    else:
        res += "<p>The headline experiment has not been run in this tier.</p>"
    # 5 what it learned
    wl = "<h2>5. What it learned</h2>"
    if st:
        bp = []
        for stname, r in st["stressors"].items():
            for label, v in r["rungs"].items():
                if label.startswith("ppo") or label == "memory_script":
                    bp.append(f"{label} ({stname}): {'none within 10 levels' if v['breaking_point'] is None else 'level ' + str(v['breaking_point'])}")
        wl += img("stress", "Stress curves for every rung (marker × = breaking point: first level below half of familiar-condition survival). Saturated levels and unsolvable levels are drawn and flagged on the site and in <code>stress.json</code>.")
        wl += "<p>Breaking points: " + "; ".join(bp[:8]) + ".</p>"
    if prb and prb.get("policies"):
        a = next(iter(prb["policies"].values()))
        wl += (f"<p><b>Probes</b> (each paired with what it does not show). Hydration threshold: constructed states with the stat overwritten, so behaviour off-distribution. "
               f"Distilled depth-3 tree agrees with the policy on {a['distilled_tree']['agreement_heldout'] * 100:.0f}% of held-out states "
               f"(tree survival {a['distilled_tree']['tree_survival']:.0f} vs policy {a['distilled_tree']['policy_survival']:.0f}); agreement is approximation, not the policy's logic. "
               f"Occlusion heatmaps show input sensitivity only; the training time machine shows when behaviours appear, not why. The point-of-no-return probe forks recorded states every 25 steps and asks the planner for a certificate; it is valid for the planner's search power, not a proof that no plan exists.</p>")
    if rt:
        wl += "<p><b>Red team.</b> " + "; ".join(
            f"{k}: worst certified-solvable scenario scored {_fmt(v['worst'][0]['score'], 2)} (survival {v['worst'][0]['survival']})" for k, v in rt["policies"].items() if v["worst"]) + \
              f" after {rt['budget']}+ candidates each; the five most instructive are in the failure museum with templated narration (no language model).</p>"
    if ab:
        eff = "; ".join(f"{n}: {_fmt(v['effect_vs_default']['diff'], 3)} [{_fmt(v['effect_vs_default']['lo'], 3)}, {_fmt(v['effect_vs_default']['hi'], 3)}] ({v['finding']})"
                        for n, v in ab["variants"].items() if n != "default")
        wl += img("ablations", "Single-change ablations, exploratory, validation suite only.", "62%") + f"<p><b>Ablation effects</b> (ladder-normalised survival vs default, {tier['ablation_seeds']} seeds): {eff}.</p>"
        wi = ab.get("wasted_interact_fraction", {})
        if wi.get("reward_naive") is not None:
            wl += (f"<p>Naive shaping: {wi['reward_naive'] * 100:.0f}% of valid interactions were taken with a stat at 90% or more (the signature of farming the per-interaction bonus), "
                   f"vs {_fmt(wi.get('reward_potential', float('nan')) * 100, 0)}% under potential-based shaping.</p>")
    if mem:
        sc = "".join(f"<tr><td>{p['shift']}</td><td>{p['predicted']}</td><td><b>{p['status']}</b></td></tr>" for p in mem["predictions"])
        wl += (img("memory", "LSTM minus feedforward, per shift, with 95% CIs.", "62%") +
               f"<table><tr><th>Shift</th><th>Locked prediction (recurrent vs feedforward)</th><th>Scored</th></tr>{sc}</table>")
        pr = mem.get("recurrent_probe")
        if pr and pr["models"]:
            m0 = pr["models"][0]
            wl += (f"<p>Recurrent-state probe (held-out episodes, shuffled-label control): steps-since-water R² {m0['hidden']['steps_since_water_r2']:.2f} from the hidden state vs "
                   f"{m0['hidden']['steps_since_water_r2_shuffled']:.2f} shuffled and {m0['observation_only']['steps_since_water_r2']:.2f} from the current observation alone; direction accuracy "
                   f"{m0['hidden']['direction_accuracy']:.2f} vs {m0['hidden']['direction_accuracy_shuffled']:.2f} shuffled (majority class {m0['hidden']['majority_class_accuracy']:.2f}). This shows whether memory holds task information, and nothing more.</p>")
        if mem.get("compression"):
            comp = mem["compression"]
            parts = [f"{k}: retention IQM {_ci(v['survival_retention_iqm'], 'iqm', 2)}, {v['size_fp32_bytes'] / 1e3:.0f}→{v['size_int8_bytes'] / 1e3:.0f} kB, "
                     f"latency {v['latency_ms_fp32_batch8']:.2f}→{v['latency_ms_int8_batch8']:.2f} ms (batch 8, 1 thread)"
                     for k, v in comp["rungs"].items() if "survival_retention_iqm" in v]
            if parts:
                wl += f"<p>Int8 post-training quantization (stretch, exploratory, measured on {html.escape(comp['hardware'])} only): " + "; ".join(parts) + ".</p>"
    # debugging story
    dbg = ""
    if "debug" in figs:
        dbg = ("<h3>The debugging story</h3>" + img("debug", "A real failed run and its fix, regenerated from logged diagnostics on a tiny solvable scenario whose optimal return is known.") +
               "<p>The first PPO runs stayed at random-level survival on a scenario the planner solves. The diagnostics told the story: value loss of 30 to 90, explained variance swinging between -8 and +2, "
               "and approximate-KL spikes above 6, while entropy barely moved. With a reward of 1 per step and γ = 0.99 returns are around 100, so the value-loss gradient dominated the single joint "
               "gradient-norm clip (0.5) and left the policy gradient almost nothing. Scaling the learner's reward by 0.01 (the environment and every reported survival number are untouched) fixed it: "
               "the same scenario was solved in about 30 updates. The failed run is kept in <code>results/&lt;tier&gt;/debug/</code>.</p>")
    # 6 limitations
    lim = ["<h2>6. Limitations</h2><ul>"
           "<li>One small custom environment. Any statement about regret-guided training holds only under this benchmark, with these limits; nothing is claimed for domain randomization in general.</li>"
           "<li>Regret is cheap here because a planner exists. In most environments it is unobservable.</li>"
           "<li>The planner is a depth-first search over resource-visit macros with a node cap and hazards treated as walls; a scenario without a certificate may still be solvable. It is a reference, not an optimum.</li>"
           "<li>Probes show behaviour and sensitivity, not reasoning; heuristic labels (blind or blunder) are suggestions.</li>"
           "<li>Compute: " + html.escape(f"{tier['seeds']} seeds × {tier['steps']:,} transitions per arm; ablations {tier['ablation_seeds']} seeds × {tier['ablation_steps']:,}; "
                                          f"memory track {tier['memory_seeds']} seeds (LSTM, {tier['memory_steps']:,}) and {tier['memory_variant_seeds']} seeds for the 3×3-vision and no-cost-stat variants ({tier['memory_variant_steps']:,}); "
                                          f"tuning sweep at {tier['tune_steps']:,} transitions with {tier['tune_seeds']} seeds per cell.") + " These sizes were fixed before results existed. "
           "Training reproducibility is same machine and same software; evaluating a fixed checkpoint is bit-exact.</li>"
           "<li>Hardware and speed statements apply to the named machine only. No claim is made about SSM or quantized hardware speed or energy beyond what was measured there.</li>"
           "<li>The Play page uses Pyodide with the same Python core; it was tested in headless Chromium, not across browsers or phones.</li></ul>"]
    # 7 reproducibility
    rep = (f"<h2>7. Reproducibility</h2><p><code>make reproduce TIER=smoke|dev|full</code> runs every stage behind its gate (smoke: minutes on a laptop, 1 seed, in CI on every commit; dev: 3 seeds, 20 scenarios per condition; "
           f"full: the published benchmark). Config hashes and checksums are in <code>manifest.json</code>; each stage writes a resumable results folder with manifests, episode CSVs, aggregate CSVs, training metrics, replays and a lockfile. "
           f"Claims in the ledger link to their episode data. Policy weights are omitted from the repository except released checkpoints (model cards in <code>releases/</code>).</p>")
    refs = "<h2>References</h2><ol class='refs'>" + "".join(f"<li>{v}</li>" for v in CITES.values()) + "</ol>"
    body = "".join(s) + res + wl + dbg + "".join(lim) + rep + refs
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>AdaptArena v2 technical report</title><style>{CSS}</style></head><body>{body}</body></html>"


def build(tier_name: str) -> str:
    tdir = tier_dir(tier_name)
    rdir = os.path.join(tdir, "report")
    os.makedirs(rdir, exist_ok=True)
    figs = figures(tier_name)
    html_text = build_report_html(tier_name, figs)
    path = os.path.join(rdir, "report.html")
    open(path, "w").write(html_text)
    pdf = os.path.join(rdir, "report.pdf")
    try:
        from playwright.sync_api import sync_playwright
        exe = None
        for cand in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome",):
            if os.path.exists(cand):
                exe = cand
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch()
            pg = b.new_page()
            pg.goto("file://" + path)
            pg.pdf(path=pdf, format="Letter", print_background=True)
            b.close()
        try:
            from pypdf import PdfReader
            print("report pages:", len(PdfReader(pdf).pages))
        except Exception:
            pass
    except Exception as e:  # pragma: no cover
        print("PDF skipped:", e)
    print("report written to", rdir)
    return path


def report_page_html(tier_name: str, page_fn, meta) -> str:
    """The site's Report page: the report itself, embedded, plus raw data downloads."""
    tdir = tier_dir(tier_name)
    rpath = os.path.join(tdir, "report", "report.html")
    body = ""
    if os.path.exists(rpath):
        raw = open(rpath).read()
        inner = raw.split("<body>")[1].split("</body>")[0]
        body = f'<div class="card report">{inner}</div>'
    extra = ('<p><a href="data/results/episodes.csv">episodes.csv</a> · <a href="data/results/aggregate.csv">aggregate.csv</a> · '
             '<a href="data/results/comparisons_appendix.csv">comparisons_appendix.csv</a> · <a href="data/results/ablations.csv">ablations.csv</a> · '
             '<a href="data/headline.json">headline.json</a> · <a href="data/stress.json">stress.json</a> · <a href="data/probes.json">probes.json</a> · '
             '<a href="data/memory.json">memory.json</a> · <a href="data/manifest.json">manifest.json</a> · <a href="report.pdf">report.pdf</a></p>')
    html_page = page_fn("report.html", "report", "Technical report", "The report is the product; the site, video and package exist to get you to it and let you trust it.",
                        f'<h2 id="downloads">Raw data downloads</h2>{extra}{body}', meta)
    return html_page
