"""Experiment orchestration: tiers, training jobs, the parity gate and the headline experiment.

Everything writes into `results/<tier>/` as plain files (CSV/JSON/torch checkpoints). A job is
skipped when its `done.json` exists, so any stage can be resumed.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import yaml

from ..core import params as P
from ..core.distributions import CONDITIONS
from ..core.env import EnvConfig
from . import plan as planmod
from . import stats as S
from .suites import Suite, build_suite_file, open_locked_test_suite, per_seed_score

REPO = planmod.REPO
ARMS = {"A": "Specialist", "B": "Uniform randomization", "C": "Replay by value loss", "D": "Replay by planner regret"}


# ------------------------------------------------------------------------------ tiers
def load_tier(name: str) -> dict:
    with open(os.path.join(REPO, "configs", "tiers.yaml")) as f:
        t = yaml.safe_load(f)[name]
    t["name"] = name
    return t


def tier_dir(name: str) -> str:
    d = os.path.join(REPO, "results", name)
    os.makedirs(d, exist_ok=True)
    return d


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def write_csv(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def manifest(tdir: str, tier: dict) -> dict:
    import numpy, scipy, torch
    m = {"tier": tier["name"], "python": sys.version.split()[0], "platform": platform.platform(),
         "machine": platform.machine(), "cpu_count": os.cpu_count(), "numpy": numpy.__version__,
         "torch": torch.__version__, "scipy": scipy.__version__, "env_id": P.ENV_ID,
         "plan_hash": open(planmod.HASH).read().strip() if os.path.exists(planmod.HASH) else None,
         "tier_config": tier, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import stable_baselines3
        m["stable_baselines3"] = stable_baselines3.__version__
    except Exception:
        pass
    write_json(os.path.join(tdir, "manifest.json"), m)
    return m


def get_val_suite(tdir: str, tier: dict, seed: int = 0) -> Suite:
    name = "suite_val.json" if seed == 0 else f"suite_val_s{seed}.json"
    path = os.path.join(tdir, name)
    if not os.path.exists(path):
        print(f"building validation suite (seed {seed})")
        build_suite_file("val", tier["val_per_condition"], path, seed=seed)
    return Suite.load(path)


# ------------------------------------------------------------------------------ jobs
def run_jobs(fn: Callable[[dict], None], jobs: List[dict], workers: int = 4) -> None:
    todo = [j for j in jobs if not os.path.exists(os.path.join(j["out_dir"], "done.json"))]
    print(f"{len(todo)}/{len(jobs)} jobs to run on {workers} workers", flush=True)
    if not todo:
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, _ in enumerate(pool.imap_unordered(fn, todo), 1):
            print(f"  finished {i}/{len(todo)}", flush=True)


def _env_config(d: Optional[dict]) -> EnvConfig:
    return EnvConfig(**(d or {}))


def train_job(job: dict) -> None:
    """One training run. job keys: out_dir, arm, model, seed, steps, env, ppo, val_suite."""
    import torch
    torch.set_num_threads(1)
    from .models import make_model
    from .ppo import PPOConfig, train
    from .sources import make_source

    out = job["out_dir"]
    if os.path.exists(os.path.join(out, "done.json")):
        return
    suite = Suite.load(job["val_suite"])
    val_specs = suite.all_specs()
    env_cfg = _env_config(job.get("env"))
    kw = dict(total_steps=job["steps"], seed=job["seed"])
    kw.update(job.get("ppo", {}))
    if job["model"] != "ff":
        kw.setdefault("n_minibatches", 2)
    cfg = PPOConfig(**kw)
    torch.manual_seed(job["seed"])
    model = make_model(job["model"])
    source = make_source(job.get("arm", "B"), job["seed"])
    t0 = time.time()
    res = train(model, cfg, source, env_cfg, val_specs, out)
    write_json(os.path.join(out, "done.json"), {"wall": time.time() - t0, "job": job,
                                                "best_val": res.best_val, "best_update": res.best_update})


def load_model(run_dir: str, kind: str, which: str = "best"):
    import torch
    from .models import make_model
    m = make_model(kind)
    m.load_state_dict(torch.load(os.path.join(run_dir, f"model_{which}.pt"), map_location="cpu"))
    m.eval()
    return m


# ------------------------------------------------------------------------------ parity gate
def sb3_job(job: dict) -> None:
    """Train Stable-Baselines3 PPO with the same hyper-parameters, env and budget as own PPO (arm B)."""
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ..gym_env import AdaptArenaSurvivalEnv
    from .evaluate import evaluate, survival_array

    out = job["out_dir"]
    os.makedirs(out, exist_ok=True)
    suite = Suite.load(job["val_suite"])
    val_specs = suite.all_specs()
    scale = 0.01  # same reward scaling as own PPO

    class Flat(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0.0, 1.0, (P.OBS_DIM,), np.float32)

        def observation(self, o):
            return np.concatenate([o["grid"].reshape(-1), o["stats"]]).astype(np.float32)

        def step(self, a):
            o, r, te, tr, info = self.env.step(a)
            return self.observation(o), r * scale, te, tr, info

    class Adapter(nn.Module):
        recurrent = False
        name = "sb3"

        def __init__(self, sb3):
            super().__init__()
            self.sb3 = sb3.policy

        def initial_state(self, n):
            return None

        def step(self, obs, state, first):
            dist = self.sb3.get_distribution(obs)
            return dist.distribution.logits, self.sb3.predict_values(obs).squeeze(-1), None

        sequence = step

    n_envs = 8
    venv = DummyVecEnv([lambda: Flat(AdaptArenaSurvivalEnv()) for _ in range(n_envs)])
    venv.seed(job["seed"])
    total = job["steps"]
    model = PPO("MlpPolicy", venv, learning_rate=lambda p: 3e-4 * p, n_steps=256, batch_size=512, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                normalize_advantage=True, seed=job["seed"], device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128]), activation_fn=nn.Tanh))
    best = {"v": -1.0}
    rows = []
    t0 = time.time()

    def validate(steps):
        res = evaluate(Adapter(model), val_specs, list(range(len(val_specs))), deterministic=True)
        v = float(survival_array(res).mean())
        rows.append({"steps": steps, "val_survival": v})
        if v > best["v"]:
            best["v"] = v
            model.save(os.path.join(out, "model_best"))

    class CB(BaseCallback):
        def __init__(self):
            super().__init__()
            self.k = 0

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            self.k += 1
            if self.k % 25 == 0:
                validate(self.num_timesteps)

    validate(0)
    model.learn(total_timesteps=total, callback=CB())
    validate(model.num_timesteps)
    model.save(os.path.join(out, "model_final"))
    write_csv(os.path.join(out, "val_log.csv"), rows)
    write_json(os.path.join(out, "done.json"), {"wall": time.time() - t0, "job": job, "best_val": best["v"],
                                                "sps": model.num_timesteps / (time.time() - t0)})


def run_parity(tier_name: str, workers: int = 4) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    manifest(tdir, tier)
    val = get_val_suite(tdir, tier, 0)
    get_val_suite(tdir, tier, 1)
    vpath = os.path.join(tdir, "suite_val.json")
    own = [dict(out_dir=os.path.join(tdir, "parity", "own", f"seed_{s}"), arm="B", model="ff", seed=s,
                steps=tier["parity_steps"], val_suite=vpath) for s in range(tier["parity_seeds"])]
    sb3 = [dict(out_dir=os.path.join(tdir, "parity", "sb3", f"seed_{s}"), seed=s, steps=tier["parity_steps"],
                val_suite=vpath) for s in range(tier["parity_seeds"])]
    run_jobs(train_job, own, workers)
    run_jobs(sb3_job, sb3, workers)
    return analyse_parity(tier_name)


def analyse_parity(tier_name: str) -> dict:
    import torch
    from stable_baselines3 import PPO
    from .evaluate import evaluate, survival_array
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    plan_cfg = yaml.safe_load(open(os.path.join(REPO, "analysis_plan.yaml")))
    crit = plan_cfg["parity_gate"]
    ev = get_val_suite(tdir, tier, 1)  # fresh scenarios, not used for checkpoint selection
    specs = ev.all_specs()
    scores: Dict[str, np.ndarray] = {}
    for impl in ("own", "sb3"):
        vals = []
        for s in range(tier["parity_seeds"]):
            d = os.path.join(tdir, "parity", impl, f"seed_{s}")
            if impl == "own":
                m = load_model(d, "ff")
                res = evaluate(m, specs, list(range(len(specs))))
            else:
                from .experiments import _sb3_adapter
                res = evaluate(_sb3_adapter(PPO.load(os.path.join(d, "model_best"), device="cpu")), specs,
                               list(range(len(specs))))
            vals.append(float(survival_array(res).mean()))
        scores[impl] = np.array(vals)
    rng = np.random.default_rng(0)
    out = {"seeds": {k: v.tolist() for k, v in scores.items()}}
    ci = {}
    for k, v in scores.items():
        boot = S._iqm_rows(v[rng.integers(0, len(v), size=(S.N_BOOT, len(v)))])
        lo, hi = S.ci(boot, crit["ci_level"])
        ci[k] = {"iqm": S.iqm(v), "lo": lo, "hi": hi}
    out["ci"] = ci
    out["own_in_sb3_ci"] = ci["sb3"]["lo"] <= ci["own"]["iqm"] <= ci["sb3"]["hi"]
    out["sb3_in_own_ci"] = ci["own"]["lo"] <= ci["sb3"]["iqm"] <= ci["own"]["hi"]
    # learning curves: val_survival vs steps, mean and seed band
    curves = {}
    for impl in ("own", "sb3"):
        per = []
        for s in range(tier["parity_seeds"]):
            rows = list(csv.DictReader(open(os.path.join(tdir, "parity", impl, f"seed_{s}", "val_log.csv"))))
            per.append([float(r["val_survival"]) for r in rows])
            steps = [int(float(r["steps"])) for r in rows]
        n = min(len(p) for p in per)
        arr = np.array([p[:n] for p in per])
        curves[impl] = {"steps": steps[:n], "mean": arr.mean(0).tolist(), "min": arr.min(0).tolist(),
                        "max": arr.max(0).tolist()}
    n = min(len(curves["own"]["mean"]), len(curves["sb3"]["mean"]))
    inside = []
    for i in range(1, n):  # skip step 0 (untrained)
        a, b = curves["own"], curves["sb3"]
        inside.append(b["min"][i] <= a["mean"][i] <= b["max"][i] and a["min"][i] <= b["mean"][i] <= a["max"][i])
    out["curves"] = curves
    out["curve_overlap_fraction"] = float(np.mean(inside)) if inside else float("nan")
    out["criteria"] = crit
    out["passed"] = bool(out["own_in_sb3_ci"] and out["sb3_in_own_ci"]
                         and out["curve_overlap_fraction"] >= crit["curve_overlap_min_fraction"])
    write_json(os.path.join(tdir, "parity", "parity.json"), out)
    return out


def _sb3_adapter(sb3_model):
    import torch.nn as nn

    class Adapter(nn.Module):
        recurrent = False
        name = "sb3"

        def __init__(self, sb3):
            super().__init__()
            self.sb3 = sb3.policy

        def initial_state(self, n):
            return None

        def step(self, obs, state, first):
            dist = self.sb3.get_distribution(obs)
            return dist.distribution.logits, self.sb3.predict_values(obs).squeeze(-1), None

        sequence = step

    return Adapter(sb3_model)


# ------------------------------------------------------------------------------ headline
def headline_jobs(tier_name: str) -> List[dict]:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    vpath = os.path.join(tdir, "suite_val.json")
    from .tune import tuned_ppo
    tuned = tuned_ppo(tier_name, "ff")
    jobs = []
    for arm in ARMS:
        for s in range(tier["seeds"]):
            jobs.append(dict(out_dir=os.path.join(tdir, "headline", f"arm_{arm}", f"seed_{s}"), arm=arm,
                             model="ff", seed=s, steps=tier["steps"], ppo=tuned, val_suite=vpath))
    return jobs


def run_headline_training(tier_name: str, workers: int = 4) -> None:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    parity_path = os.path.join(tdir, "parity", "parity.json")
    if not os.path.exists(parity_path):
        raise SystemExit("run the parity gate first: adaptarena parity --tier " + tier_name)
    if not json.load(open(parity_path))["passed"]:
        print("WARNING: the parity gate did not pass; the headline run is blocked by the plan unless "
              "an amendment is recorded.", flush=True)
        if not os.environ.get("ADAPTARENA_FORCE_HEADLINE"):
            raise SystemExit("parity gate failed (set ADAPTARENA_FORCE_HEADLINE=1 after writing an amendment)")
    get_val_suite(tdir, tier, 0)
    run_jobs(train_job, headline_jobs(tier_name), workers)


def _episode_rows(arm_label: str, seed: int, ev: dict, mode: str) -> List[dict]:
    rows = []
    for c, d in ev.items():
        for k, r in enumerate(d["results"]):
            rows.append({"policy": arm_label, "seed": seed, "condition": c, "idx": k, "sid": r.sid,
                         "survival": r.survival, "cause": r.cause, "mode": mode, "action_seed": r.action_seed,
                         "final_hash": r.final_hash, "normalised": float(d["score"][k]) if d["valid"][k] else ""})
    return rows


def evaluate_headline(tier_name: str) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    suite = open_locked_test_suite(tdir, "headline evaluation", tier["scenarios_per_condition"], generate=True)
    episodes: List[dict] = []
    scores: Dict[str, Dict[str, list]] = {a: {c: [] for c in CONDITIONS} for a in ARMS}
    reach: Dict[str, Dict[str, list]] = {a: {c: [] for c in CONDITIONS} for a in ARMS}
    extra = {}
    for arm in ARMS:
        for s in range(tier["seeds"]):
            d = os.path.join(tdir, "headline", f"arm_{arm}", f"seed_{s}")
            m = load_model(d, "ff")
            ev = suite.evaluate(m)
            episodes += _episode_rows(f"arm_{arm}", s, ev, "deterministic")
            for c in CONDITIONS:
                scores[arm][c].append(per_seed_score(ev[c]))
                reach[arm][c].append(float((ev[c]["survival"] >= P.HORIZON).mean()))
            tr = json.load(open(os.path.join(d, "train_result.json")))
            extra.setdefault(arm, []).append({"seed": s, "wall": tr["wall_seconds"], "sps": tr["sps"],
                                              "best_update": tr["best_update"], **{k: v for k, v in tr["extra"].items()
                                                                                 if k != "snapshots"}})
    # ladder references on the same suite
    refs = {"random": suite.random_scores(), "reactive_script": suite.script_scores("reactive_script"),
            "memory_script": suite.script_scores("memory_script")}
    for name, ev in refs.items():
        episodes += _episode_rows(name, 0, {c: {**ev[c], "results": _FakeResults(ev[c]["survival"], suite.specs[c])}
                                            for c in CONDITIONS}, "scripted")
    write_csv(os.path.join(tdir, "headline", "episodes.csv"), episodes)
    out = headline_stats(tier_name, scores, reach, refs, suite, extra)
    return out


class _FakeResults(list):
    def __init__(self, survival, specs):
        from .evaluate import EpisodeResult
        super().__init__(EpisodeResult(sid=s.sid, survival=int(v), horizon=500, cause="", mode="scripted",
                                       action_seed=i, final_hash="", energy=0, hydration=0)
                         for i, (v, s) in enumerate(zip(survival, specs)))


def headline_stats(tier_name: str, scores, reach, refs, suite, extra) -> dict:
    tdir = tier_dir(tier_name)
    sc = {a: {c: np.array(v, dtype=float) for c, v in cd.items()} for a, cd in scores.items()}
    out: dict = {"arms": ARMS, "conditions": list(CONDITIONS), "in_distribution": suite.data["conditions"] and
                 {c: suite.data["conditions"][c]["in_distribution"] for c in CONDITIONS}}
    out["per_arm"] = {}
    for a in ARMS:
        out["per_arm"][a] = {
            "aggregate_iqm": S.iqm_with_ci(sc[a]),
            "optimality_gap": S.optimality_gap_with_ci(sc[a]),
            "per_condition": {c: S.iqm_with_ci({c: sc[a][c]}) for c in CONDITIONS},
            "profile": S.performance_profile(sc[a], np.linspace(0, 1, 41)),
        }
    out["comparisons"] = {}
    for x, y, primary in (("D", "B", True), ("D", "C", True), ("D", "A", False), ("C", "B", False), ("B", "A", False)):
        key = f"{x}_vs_{y}"
        out["comparisons"][key] = {
            "primary": primary,
            "iqm_diff": S.iqm_difference_with_ci(sc[x], sc[y]),
            "prob_improvement": S.prob_improvement_with_ci(sc[x], sc[y]),
            "per_condition": {c: {"iqm_diff": S.iqm_difference_with_ci({c: sc[x][c]}, {c: sc[y][c]}, B=4000),
                                  "p": S.prob_improvement(sc[x][c], sc[y][c])} for c in CONDITIONS},
        }
    # saturation: a condition where every arm reaches the horizon in >90% of episodes is not a win
    sat = {}
    for c in CONDITIONS:
        frac = {a: float(np.mean(reach[a][c])) for a in ARMS}
        sat[c] = {"reach_horizon": frac, "all_saturated": all(v > 0.9 for v in frac.values())}
    out["saturation"] = sat
    out["ladder"] = {name: {c: float(np.mean(ev[c]["score"][ev[c]["valid"]])) for c in CONDITIONS}
                     for name, ev in refs.items()}
    out["ladder_raw_survival"] = {name: {c: float(np.mean(ev[c]["survival"])) for c in CONDITIONS}
                                  for name, ev in refs.items()}
    out["train_cost"] = {a: {"mean_wall_s": float(np.mean([e["wall"] for e in extra[a]])),
                             "planner_calls": float(np.mean([e.get("planner_calls", 0) for e in extra[a]])),
                             "planner_seconds": float(np.mean([e.get("planner_seconds", 0) for e in extra[a]]))}
                         for a in ARMS}
    out["per_seed"] = {a: S.per_seed_table(sc[a]) for a in ARMS}
    out["decision"] = decide_outcome(out)
    out["ledger"] = headline_claims(out)
    write_json(os.path.join(tdir, "headline", "stats.json"), out)
    write_csv(os.path.join(tdir, "headline", "aggregate.csv"),
              [{"arm": a, "iqm": out["per_arm"][a]["aggregate_iqm"]["iqm"], "lo": out["per_arm"][a]["aggregate_iqm"]["lo"],
                "hi": out["per_arm"][a]["aggregate_iqm"]["hi"], "optimality_gap": out["per_arm"][a]["optimality_gap"]["gap"]}
               for a in ARMS])
    write_csv(os.path.join(tdir, "headline", "comparisons_appendix.csv"),
              [{"comparison": k, "primary": v["primary"], "iqm_diff": v["iqm_diff"]["diff"], "lo": v["iqm_diff"]["lo"],
                "hi": v["iqm_diff"]["hi"], "p_improve": v["prob_improvement"]["p"], "p_lo": v["prob_improvement"]["lo"],
                "p_hi": v["prob_improvement"]["hi"]} for k, v in out["comparisons"].items()] +
              [{"comparison": f"{k}:{c}", "primary": False, "iqm_diff": d["iqm_diff"]["diff"], "lo": d["iqm_diff"]["lo"],
                "hi": d["iqm_diff"]["hi"], "p_improve": d["p"], "p_lo": "", "p_hi": ""}
               for k, v in out["comparisons"].items() for c, d in v["per_condition"].items()])
    return out


def decide_outcome(out: dict) -> dict:
    """Apply the outcome table written into the analysis plan before any test run."""
    c = out["comparisons"]
    db, dc = c["D_vs_B"]["iqm_diff"], c["D_vs_C"]["iqm_diff"]
    clear = lambda d: d["lo"] > 0
    loses = lambda d: d["hi"] < 0
    if clear(db) and clear(dc):
        k, text = "D_beats_B_and_C", ("Planner regret helps in this environment: arm D beats both uniform randomization and "
                                      "value-loss replay with intervals clear of zero, under this benchmark's limits.")
    elif loses(db):
        k, text = "D_loses_to_B", "Uniform randomization wins here; see the diagnosis of why regret misled."
    elif clear(db) and not clear(dc) and not loses(dc):
        k, text = "D_ties_C", "Planner access adds nothing over value loss here (D beats B, but D and C are indistinguishable)."
    elif not clear(db) and not loses(db):
        k, text = "D_ties_B", "No measurable benefit of planner-regret sampling over uniform randomization at this seed count."
    else:
        k, text = "mixed", "Mixed: see the pre-declared comparisons."
    sat = [cn for cn, v in out["saturation"].items() if v["all_saturated"]]
    return {"outcome": k, "text": text, "saturated_conditions": sat}


def status_from_interval(diff: dict) -> str:
    if diff["lo"] > 0:
        return "supported"
    if diff["diff"] > 0:
        return "mixed"
    return "not supported"


def headline_claims(out: dict) -> list:
    claims = []
    for key, text in (("D_vs_B", "Planner-regret replay (D) retains more ladder-normalised survival than uniform randomization (B)."),
                      ("D_vs_C", "Planner-regret replay (D) retains more ladder-normalised survival than value-loss replay (C).")):
        v = out["comparisons"][key]
        st = status_from_interval(v["iqm_diff"])
        if st == "supported" and v["prob_improvement"]["lo"] <= 0.5:
            st = "mixed"
        claims.append({"id": key, "claim": text, "pre_declared": True, "status": st,
                       "evidence": {"iqm_diff": v["iqm_diff"], "prob_improvement": v["prob_improvement"]},
                       "link": "headline/stats.json"})
    comb = out["comparisons"]["D_vs_B"]["per_condition"]["combined_shift"]
    claims.append({"id": "D_vs_B_combined", "pre_declared": False, "claim":
                   "Under the combined shift, D retained more survival than B (exploratory).",
                   "status": status_from_interval(comb["iqm_diff"]), "evidence": comb["iqm_diff"], "link": "headline/comparisons_appendix.csv"})
    return claims


# ------------------------------------------------------------------------------ policy zoo
def policy_zoo(tdir: str, tier: dict, include_memory: bool = True, max_seeds: Optional[int] = None) -> dict:
    """Every rung of the ladder that can be run on a scenario list.

    Returns {label: {"rung": int, "kind": "script"|"model", "policies": [name or model, ...]}}.
    The planner (rung 9) is handled separately because it needs the scenario, not an observation.
    """
    zoo: dict = {"random": {"rung": 1, "kind": "script", "policies": ["random"]},
                 "reactive_script": {"rung": 2, "kind": "script", "policies": ["reactive_script"]},
                 "memory_script": {"rung": 3, "kind": "script", "policies": ["memory_script"]}}
    seeds = range(tier["seeds"] if max_seeds is None else min(max_seeds, tier["seeds"]))
    for arm in ("B", "D"):
        ms = []
        for s in seeds:
            d = os.path.join(tdir, "headline", f"arm_{arm}", f"seed_{s}")
            if os.path.exists(os.path.join(d, "model_best.pt")):
                ms.append(load_model(d, "ff"))
        if ms:
            zoo[f"ppo_ff_arm{arm}"] = {"rung": 5, "kind": "model", "policies": ms}
    if include_memory:
        for kind, rung in (("lstm", 7), ("ssm", 8)):
            ms = []
            for s in range(tier["memory_seeds"]):
                d = os.path.join(tdir, "memory", kind, "default", f"seed_{s}")
                if os.path.exists(os.path.join(d, "model_best.pt")):
                    ms.append(load_model(d, kind))
            if ms:
                zoo[f"ppo_{kind}"] = {"rung": rung, "kind": "model", "policies": ms}
    return zoo


def run_zoo_entry(entry: dict, specs, seeds=None, env_config=None):
    """Evaluate every policy of a zoo entry; returns a list (one per policy) of EpisodeResult lists."""
    from .evaluate import evaluate
    seeds = list(range(len(specs))) if seeds is None else seeds
    return [evaluate(p, specs, seeds, env_config) for p in entry["policies"]]
