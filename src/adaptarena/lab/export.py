"""Experiment export package and released-checkpoint model cards.

Layout (kept from v1): manifest, methodology, scenario configs, policy metadata, episode metrics CSV, aggregate
metrics CSV, training metrics CSV, replays, dependency lockfile and checksums. Policy weights are omitted from the
export (see `releases/` for the released checkpoints) but their hashes and the retrieval steps are documented.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil
from typing import List

import yaml

from . import plan as planmod
from .experiments import ARMS, load_tier, tier_dir, write_json

REPO = planmod.REPO


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


METHODOLOGY = """# Methodology

Environment `{env}`: integer-state survival world (15x15, 6 actions, reward 1 per living step, 500-step horizon).
Seed streams: layout (`layout_id`), resource timing (`timing_seed`), weather (`weather_seed`), action sampling (per-episode action seed), training (`seed`).

## Training
- Algorithm: PPO written from scratch (`src/adaptarena/lab/ppo.py`): GAE (gamma 0.99, lambda 0.95), clip 0.2, entropy {ent}, value coefficient 0.5, gradient clipping 0.5,
  linear LR decay from {lr}, 8 environments x 256 steps per update, 10 epochs, 4 minibatches, advantage normalisation, reward scaled by 0.01 inside the learner.
- Termination vs truncation: bootstrap from V(final observation) on horizon truncation, never on death.
- Transition budget per run: {steps:,}, counted across all environments. Seeds per arm: {seeds}.
- Checkpoint rule: highest mean validation survival, earlier checkpoint wins ties.
- Recurrent policies: sequence chunks of 64 steps with hidden-state resets at episode boundaries; 2 sequence minibatches.
- Tuning: one sweep per method (3 learning rates x 3 entropy coefficients x {tune_seeds} seeds, {tune_steps:,} transitions) scored on validation only.

## Evaluation partitions
- training layouts 0..9999; validation layouts 200000+; locked test layouts 100000+; stress 300000+; red team 400000+. Timing and weather seeds are in disjoint ranges.
- Locked test suite: 7 conditions x {scen} planner-certified scenarios, deterministic action selection (primary), sampled actions secondary and never mixed in one aggregate.
- The harness opens the locked suite only if the SHA-256 of `analysis_plan.yaml` matches the committed `analysis_plan.sha256`; every access is logged.

## Interventions
Precomputed schedule edits `(name, step, argument)`: `cost` (movement-cost percent from a step), `cold` (forced cold spell), `drought` / `blight` (water / food sites vanish),
`reloc` (water sites move to alternate cells), `hazard` (tiles become deadly). Hazards are off in every benchmark suite and used only in the hazard-spill demo.

## Metrics
- Survival: steps survived within the 500-step horizon (right-censored). Ladder-normalised survival: (policy - random mean) / (planner - random mean) per scenario, random mean over action seeds 0..9.
- Aggregates: IQM over training seeds (episodes averaged inside a seed first), stratified bootstrap CIs (10,000 resamples), optimality gap, probability of improvement, performance profiles.

## Known limitations
See the technical report, section 6.
"""


def export(tier_name: str) -> str:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    out = os.path.join(tdir, "export")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    tuned = {}
    tp = os.path.join(tdir, "tune", "tuned.json")
    if os.path.exists(tp):
        tuned = json.load(open(tp)).get("ff", {}).get("best", {})
    open(os.path.join(out, "methodology.md"), "w").write(METHODOLOGY.format(
        env="AdaptArena-Survival-v1", ent=tuned.get("ent_coef", 0.01), lr=tuned.get("lr", 3e-4), steps=tier["steps"],
        seeds=tier["seeds"], tune_seeds=tier["tune_seeds"], tune_steps=tier["tune_steps"], scen=tier["scenarios_per_condition"]))
    for name in ("suite_val.json", "suite_test.json", "suite_test_memory.json", "calibration.json", "manifest.json",
                 "test_access_log.jsonl"):
        p = os.path.join(tdir, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, name))
    for src in ("headline/episodes.csv", "headline/aggregate.csv", "headline/comparisons_appendix.csv", "headline/stats.json",
                "ablations/ablations.csv", "ablations/ablations.json", "memory/memory.json", "stress/stress.json",
                "probes/probes.json", "redteam/redteam.json", "museum/museum.json", "parity/parity.json", "tune/tuned.json"):
        p = os.path.join(tdir, src)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, src.replace("/", "_")))
    for name in ("analysis_plan.yaml", "analysis_plan.sha256"):
        shutil.copy(os.path.join(REPO, name), os.path.join(out, name))
    if os.path.exists(planmod.AMEND):
        shutil.copy(planmod.AMEND, os.path.join(out, "plan_amendments.jsonl"))
    lock = os.path.join(REPO, "requirements-lock.txt")
    if os.path.exists(lock):
        shutil.copy(lock, os.path.join(out, "requirements-lock.txt"))
    # training metrics (all runs) and policy metadata with weight hashes
    rows, meta = [], []
    for log in sorted(glob.glob(os.path.join(tdir, "**", "train_log.csv"), recursive=True)):
        run = os.path.relpath(os.path.dirname(log), tdir)
        for r in csv.DictReader(open(log)):
            rows.append({"run": run, **r})
        best = os.path.join(os.path.dirname(log), "model_best.pt")
        res = os.path.join(os.path.dirname(log), "train_result.json")
        if os.path.exists(res):
            tr = json.load(open(res))
            meta.append({"run": run, "model": tr["model"], "n_params": tr["n_params"], "best_update": tr["best_update"],
                         "best_val": tr["best_val"], "steps": tr["steps"], "config": tr["config"],
                         "weights_sha256": sha256(best) if os.path.exists(best) else None,
                         "weights": "omitted for size; regenerate with `adaptarena reproduce` or see releases/ for released checkpoints"})
    if rows:
        with open(os.path.join(out, "training_metrics.csv"), "w", newline="") as f:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write_json(os.path.join(out, "policy_metadata.json"), meta)
    # replays used by the site
    site_data = os.path.join(REPO, "site", "dist", "data", "compare")
    if os.path.isdir(site_data):
        shutil.copytree(site_data, os.path.join(out, "replays"))
    sums = {}
    for p in sorted(glob.glob(os.path.join(out, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            sums[os.path.relpath(p, out)] = sha256(p)
    with open(os.path.join(out, "checksums.sha256"), "w") as f:
        for k, v in sums.items():
            f.write(f"{v}  {k}\n")
    return out


CARD = """# Model card: {name}

- **Policy**: {model} ({n_params:,} parameters), own PPO implementation, rung {rung} of the ladder.
- **Training**: arm {arm} ({arm_name}), seed {seed}, {steps:,} transitions, checkpoint = highest mean validation survival (update {best_update}).
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `{sha}`.
- **Environment**: `AdaptArena-Survival-v1`, observation {obs}.
- **Intended use**: reproducing the published benchmark and exploring the site's probes. Frozen policy: it does not learn at inference time.
- **Result on the locked test suite**: ladder-normalised aggregate IQM for this arm {iqm}.
- **Limitations**: trained and evaluated on one small custom environment; behaviour outside the stated distributions is not characterised; probes show behaviour, not reasoning.
- **Config hash**: `{cfg_hash}`; analysis plan `{plan}`.
"""


def release(tier_name: str) -> List[str]:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    hl = None
    hp = os.path.join(tdir, "headline", "stats.json")
    if os.path.exists(hp):
        hl = json.load(open(hp))
    plan_hash = open(planmod.HASH).read().strip()
    made = []
    specs = [(f"ppo_ff_arm{a}_seed0", os.path.join(tdir, "headline", f"arm_{a}", "seed_0"), a) for a in ARMS]
    specs.append(("ppo_lstm_seed0", os.path.join(tdir, "memory", "lstm", "default", "seed_0"), "B"))
    for name, d, arm in specs:
        w, r = os.path.join(d, "model_best.pt"), os.path.join(d, "train_result.json")
        if not (os.path.exists(w) and os.path.exists(r)):
            continue
        tr = json.load(open(r))
        dst = os.path.join(REPO, "releases", name)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(w, os.path.join(dst, "model.pt"))
        cfg_hash = hashlib.sha256(json.dumps(tr["config"], sort_keys=True).encode()).hexdigest()[:16]
        iqm = "n/a"
        if hl and name.startswith("ppo_ff"):
            a = hl["per_arm"][arm]["aggregate_iqm"]
            iqm = f"{a['iqm']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"
        open(os.path.join(dst, "model_card.md"), "w").write(CARD.format(
            name=name, model=tr["model"], n_params=tr["n_params"], rung=7 if "lstm" in name else 5, arm=arm, arm_name=ARMS[arm],
            seed=0, steps=tr["steps"], best_update=tr["best_update"], sha=sha256(w), obs="obs-v9", iqm=iqm,
            cfg_hash=cfg_hash, plan=plan_hash[:16]))
        made.append(dst)
    return made
