"""Static showcase site builder: results folder -> `site/dist/` (no server, read-only files)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List

import numpy as np

from ..core import params as P
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario, with_intervention
from . import plan as planmod
from .experiments import get_val_suite, load_model, load_tier, policy_zoo, tier_dir
from .replays import record_replay, replay_from_actions, strip_private, verify_replay

SITE = os.path.join(planmod.REPO, "site")
PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"
INTERVENTIONS = [
    {"id": "none", "label": "No intervention", "step": None},
    {"id": "drought", "label": "Drought: two water sites vanish at step 150", "step": 150, "spec": ("drought", 150, 2)},
    {"id": "cost", "label": "Cost spike: movement costs 250% from step 150", "step": 150, "spec": ("cost", 150, 250)},
    {"id": "cold", "label": "Cold spell: steps 120-200", "step": 120, "spec": ("cold", 120, 80)},
    {"id": "reloc", "label": "Relocation: water moves at step 200", "step": 200, "spec": ("reloc", 200, 0)},
    {"id": "hazard", "label": "Hazard spill: 6 tiles turn deadly at step 150", "step": 150, "spec": ("hazard", 150, 6)},
]
RUNG_LABELS = {"random": "Rung 1: uniform random", "reactive_script": "Rung 2: reactive script",
               "memory_script": "Rung 3: memory script", "ppo_ff_armB": "Rung 5: PPO feedforward (arm B)",
               "ppo_ff_armD": "Rung 5: PPO feedforward (arm D)", "ppo_lstm": "Rung 7: PPO + LSTM",
               "ppo_ssm": "Rung 8: PPO + SSM", "planner": "Rung 9: reference planner"}


def _layout_hash(spec: ScenarioSpec) -> str:
    sc = build_scenario(spec)
    h = hashlib.sha256()
    for a in (sc.terrain, sc.food_pos, sc.water_pos, np.array(sc.start), sc.food_delay, sc.water_delay, sc.cold):
        h.update(np.ascontiguousarray(a, dtype="<i4").tobytes())
    return h.hexdigest()


def _planner_replay(spec: ScenarioSpec) -> dict:
    res = plan(build_scenario(spec))
    rep = replay_from_actions(spec, res.actions, "Reference planner")
    rep["certified"] = res.certified
    return rep


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _copy_if(src: str, dst: str) -> bool:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        return True
    return False


NAV = [("index.html", "Play"), ("compare.html", "Compare"), ("stress.html", "Stress"), ("probes.html", "Probes"),
       ("museum.html", "Museum"), ("ledger.html", "Ledger"), ("report.html", "Report")]

PAGES = {
    "index.html": ("play", "Play as the agent",
                   "Survive 500 steps in the same fog-of-war view the policies get. Your result is ranked against the ladder on exactly the same scenario.",
                   ""),
    "compare.html": ("compare", "Compare policies side by side",
                     "Three arenas share one timeline. Pick policies and an intervention; dead policies freeze on their final frame with the cause of death.",
                     ""),
    "stress.html": ("stress", "Stress curves",
                    "Survival against intensity for every rung of the ladder, with breaking points marked. Survival is right-censored at the 500-step horizon, so it is drawn as Kaplan-Meier curves.",
                    ""),
    "probes.html": ("probes", "What each policy learned",
                    "Probes that show what a policy did, each with a note on what it does not show. Nothing here reads an agent's thoughts.",
                    ""),
    "museum.html": ("museum", "Failure museum",
                    "The five most instructive verified failures found by the red team search. The reference planner certified every scenario solvable, so each failure is real.",
                    ""),
    "ledger.html": ("ledger", "Claim ledger",
                    "Each claim, its status, and the evidence behind it. The analysis plan was hashed and committed before the locked test suite was generated.",
                    ""),
}


def _hero(meta: dict) -> str:
    h = meta.get("hero")
    if not h:
        return ""
    return f"""<section class="hero reveal">
<span class="eyebrow"><i></i> Live simulator · runs in your browser</span>
<h1>Can you survive what the <span class="grad">agents</span> couldn't?</h1>
<p class="lede">A 15×15 world, a 9×9 window of fog, 500 steps. Play the exact environment that trains the agents, then see where you rank against a random walker, a scripted forager, trained PPO policies and a planner that sees everything.</p>
<div class="stats">
<div class="stat s1"><b data-to="{h['random']:.0f}">0</b><span>steps: the average life of a random agent</span></div>
<div class="stat s2"><b data-to="{h['planner']:.0f}">0</b><span>steps survived by the full-information planner, on every certified scenario</span></div>
<div class="stat s3"><b data-to="{h['best']:.3f}" data-dec="3">0</b><span>ladder score of the best learned arm (random = 0, planner = 1). Planner-regret curriculum lost to uniform sampling.</span></div>
</div></section>"""


def _page(name: str, page: str, title: str, lede: str, extra: str, meta: dict) -> str:
    cur = ' aria-current="page"'
    nav = "".join(f'<a href="{h}"{cur if h == name else ""}>{t}</a> ' for h, t in NAV)
    hero = _hero(meta) if page == "play" else ""
    head = "" if page == "play" else f'<div class="reveal"><h1>{title}</h1><p class="lede">{lede}</p></div>'
    return f"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AdaptArena v2 · {title}</title>
<meta name="description" content="{lede}">
<meta name="theme-color" content="#070a12">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3E%3Ccircle cx=%278%27 cy=%278%27 r=%277%27 fill=%27%235db4ff%27/%3E%3C/svg%3E">
<link rel="stylesheet" href="assets/style.css">
<script>try{{var t=localStorage.getItem("aa-theme");if(t)document.documentElement.setAttribute("data-theme",t)}}catch(e){{}}</script></head>
<body data-page="{page}">
<a class="skip" href="#app">Skip to content</a>
<header class="site"><div class="in"><a class="brand" href="index.html"><span class="logo" aria-hidden="true">A</span><span>AdaptArena v2<small>survival lab</small></span></a><nav aria-label="Main">{nav}</nav><button id="theme-toggle" type="button" aria-label="Switch to light theme">&#9728;</button></div></header>
<main>
{hero}{head}
{extra}
<div id="app"><noscript>This page needs JavaScript. All data is static files in <code>data/</code>.</noscript></div>
</main>
<footer>Results tier: <strong>{meta['tier']}</strong> · plan {meta['plan_hash'][:12]} · every published number links to its episode data in <a href="report.html#downloads">the raw downloads</a>.</footer>
<script src="assets/app.js" defer></script>
</body></html>"""


def build(tier_name: str, out: str = None) -> str:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    out = out or os.path.join(SITE, "dist")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "assets"))
    data = os.path.join(out, "data")
    npm = shutil.which("npm")
    if npm:
        if not os.path.exists(os.path.join(SITE, "node_modules")):
            subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=SITE, check=True)
        subprocess.run([npm, "run", "build"], cwd=SITE, check=True, capture_output=True)
    shutil.copy(os.path.join(SITE, "assets", "app.js"), os.path.join(out, "assets", "app.js"))
    shutil.copy(os.path.join(SITE, "assets", "style.css"), os.path.join(out, "assets", "style.css"))

    # python wheel for Pyodide
    pydir = os.path.join(out, "py")
    os.makedirs(pydir)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", pydir, planmod.REPO], check=True,
                   capture_output=True)
    wheel = [f for f in os.listdir(pydir) if f.endswith(".whl")][0]

    suite = get_val_suite(tdir, tier, 0)
    zoo = policy_zoo(tdir, tier, include_memory=True, max_seeds=tier["seeds"])
    learned = {k: v for k, v in zoo.items() if v["kind"] == "model"}
    seed0 = {k: v["policies"][0] for k, v in learned.items()}
    scripts = ["random", "reactive_script", "memory_script"]
    plan_hash = open(planmod.HASH).read().strip() if os.path.exists(planmod.HASH) else "unlocked"
    meta = {"tier": tier_name, "plan_hash": plan_hash}
    try:
        hl = json.load(open(os.path.join(tdir, "headline", "stats.json")))
        cal = json.load(open(os.path.join(tdir, "calibration.json")))
        meta["hero"] = {"random": float(np.mean([v["random"] for v in cal.values()])), "planner": 500.0,
                        "best": max(v["aggregate_iqm"]["iqm"] for v in hl["per_arm"].values())}
    except (OSError, KeyError, ValueError):
        pass

    # ---- challenge scenarios (validation split, never the locked suite)
    picks = [("familiar", 1), ("familiar", 2), ("unseen_layout", 1), ("scarce_water", 1), ("higher_move_cost", 1), ("combined_shift", 1)]
    challenges = []
    from .evaluate import evaluate, random_mean, survival_array
    for cond, i in picks:
        i = min(i, len(suite.specs[cond]) - 1)
        spec = suite.specs[cond][i]
        rungs = {"random": float(random_mean([spec], 10)[0])}
        for name in scripts[1:]:
            rungs[name] = float(survival_array(evaluate(name, [spec]))[0])
        for label, entry in learned.items():
            vals = [float(survival_array(evaluate(m, [spec]))[0]) for m in entry["policies"]]
            rungs[label] = float(np.median(vals))
        rungs["planner"] = float(plan(build_scenario(spec)).survival)
        challenges.append({"id": f"{cond}-{i}", "label": f"{cond.replace('_', ' ')} #{i}", "spec": spec.to_json(),
                           "rungs": rungs, "layout_hash": _layout_hash(spec)})
    ssm_run = "ppo_ssm" in learned
    _write(os.path.join(data, "ladder.json"), {"challenges": challenges, "rung_labels": RUNG_LABELS, "wheel": wheel,
                                              "pyodide": PYODIDE_URL,
                                              "rung_status": {"ppo_ssm": "run" if ssm_run else "not run"}})

    # ---- compare replays: scenarios x interventions x policies
    comp_sc = [("familiar", 0), ("unseen_layout", 0), ("combined_shift", 0)]
    policies = [{"id": n, "label": RUNG_LABELS.get(n, n), "rung": {"random": 1, "reactive_script": 2, "memory_script": 3}.get(n, 0)}
                for n in scripts]
    for n in learned:
        policies.append({"id": n, "label": RUNG_LABELS.get(n, n), "rung": zoo[n]["rung"]})
    policies.append({"id": "planner", "label": RUNG_LABELS["planner"], "rung": 9})
    index = {"scenarios": [{"id": f"{c}-{i}", "label": f"{c.replace('_', ' ')} #{i}"} for c, i in comp_sc],
             "interventions": [{k: v for k, v in iv.items() if k != "spec"} for iv in INTERVENTIONS], "policies": policies}
    _write(os.path.join(data, "compare", "index.json"), index)
    for c, i in comp_sc:
        base = suite.specs[c][min(i, len(suite.specs[c]) - 1)]
        for iv in INTERVENTIONS:
            spec = base if "spec" not in iv else with_intervention(base, *iv["spec"])
            reps: Dict[str, dict] = {}
            for n in scripts:
                reps[n] = strip_private(record_replay(spec, n, label=RUNG_LABELS[n]))
            for n, m in seed0.items():
                reps[n] = strip_private(record_replay(spec, m, label=RUNG_LABELS.get(n, n), with_probs=True))
            reps["planner"] = _planner_replay(spec)
            for n, r in reps.items():
                if n != "planner" and not verify_replay(r):
                    raise RuntimeError(f"replay verification failed for {n}")
            _write(os.path.join(data, "compare", f"{c}-{i}__{iv['id']}.json"), reps)
    first = next(iter(seed0.values()), None)
    fb = record_replay(suite.specs["familiar"][0], first if first is not None else "memory_script", with_probs=first is not None,
                       label="Curated replay")
    _write(os.path.join(data, "fallback_replay.json"), strip_private(fb))

    # ---- evidence data
    for src, dst in (("stress/stress.json", "stress.json"), ("probes/probes.json", "probes.json"),
                     ("museum/museum.json", "museum.json"), ("memory/memory.json", "memory.json"),
                     ("ablations/ablations.json", "ablations.json"), ("headline/stats.json", "headline.json"),
                     ("parity/parity.json", "parity.json"), ("redteam/redteam.json", "redteam.json"),
                     ("calibration.json", "calibration.json"), ("manifest.json", "manifest.json")):
        _copy_if(os.path.join(tdir, src), os.path.join(data, dst))
    for src in ("headline/episodes.csv", "headline/aggregate.csv", "headline/comparisons_appendix.csv", "ablations/ablations.csv"):
        _copy_if(os.path.join(tdir, src), os.path.join(data, "results", os.path.basename(src)))
    _copy_if(os.path.join(planmod.REPO, "analysis_plan.yaml"), os.path.join(data, "analysis_plan.yaml"))
    figdir = os.path.join(tdir, "report", "figures")
    if os.path.isdir(figdir):
        shutil.copytree(figdir, os.path.join(out, "figures"))
    _copy_if(os.path.join(tdir, "report", "report.pdf"), os.path.join(out, "report.pdf"))
    _write(os.path.join(data, "ledger.json"), build_ledger(tier_name, tdir, plan_hash))
    _write(os.path.join(data, "meta.json"), meta)

    for name, (page, title, lede, extra) in PAGES.items():
        with open(os.path.join(out, name), "w") as f:
            f.write(_page(name, page, title, lede, extra, meta))
    from .report import report_page_html
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(report_page_html(tier_name, _page, meta))
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns)
    print(f"site built in {out} ({size / 1e6:.1f} MB)")
    return out


def build_ledger(tier_name: str, tdir: str, plan_hash: str) -> dict:
    hl = json.load(open(os.path.join(tdir, "headline", "stats.json"))) if os.path.exists(os.path.join(tdir, "headline", "stats.json")) else None
    mem = json.load(open(os.path.join(tdir, "memory", "memory.json"))) if os.path.exists(os.path.join(tdir, "memory", "memory.json")) else None
    abl = json.load(open(os.path.join(tdir, "ablations", "ablations.json"))) if os.path.exists(os.path.join(tdir, "ablations", "ablations.json")) else None
    claims = list(hl["ledger"]) if hl else []
    if abl:
        for name, v in abl["variants"].items():
            if name == "default":
                continue
            e = v["effect_vs_default"]
            claims.append({"id": f"ablation:{name}", "pre_declared": False, "status": "supported" if e["lo"] > 0 or e["hi"] < 0 else "not supported",
                           "claim": f"Ablation '{name}' changes ladder-normalised survival relative to the default (exploratory, validation suite; observed: {v['finding']}).",
                           "evidence": e, "link": "ablations.json"})
    from .readme_gen import tag_v2_claims, v21_claims
    claims = tag_v2_claims(claims) + v21_claims(tier_name)
    amend = [json.loads(l) for l in open(planmod.AMEND)] if os.path.exists(planmod.AMEND) else []
    log_path = os.path.join(tdir, "test_access_log.jsonl")
    access = [json.loads(l) for l in open(log_path)] if os.path.exists(log_path) else []
    git_commit = planmod.git_commit_of_hash()
    notes = []
    mfile = os.path.join(tdir, "manifest.json")
    return {"claims": claims, "plan_hash": plan_hash, "git_commit": git_commit, "tier": tier_name,
            "decision": hl["decision"] if hl else {"outcome": "pending", "text": "pending", "saturated_conditions": []},
            "predictions": [{"shift": p["shift"], "predicted": p["predicted"], "status": p["status"]} for p in mem["predictions"]] if mem else [],
            "amendments": amend, "access_log": access, "ladder": hl["ladder"] if hl else {}, "notes": notes}
