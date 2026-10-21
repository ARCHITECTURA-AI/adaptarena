"""Phase 5b of v2.1: SSM (rung 8), int8 post-training quantization, and the gamma re-run with return normalization.

All three are exploratory or diagnostic and every output carries its tier label. Training is resumable. Validation
scoring uses `suite_val.json` (selection) and `suite_val_s1.json` (not used for selection); the fresh test suite is
opened only through the harness, once, for the SSM, and only to report it as exploratory.
"""
from __future__ import annotations

import copy
import os
import time
from typing import Dict, List

import numpy as np

from ..core import params as P
from ..core.batch import BatchEnv
from ..core.distributions import CONDITIONS
from ..core.scenario import build_scenario
from . import stats as S
from .experiments import load_model, load_tier, run_jobs, tier_dir, train_job, write_json
from .models import FFPolicy, LSTMPolicy, SSMPolicy, count_params
from .suites import Suite, per_seed_score
from .tune import tuned_ppo
from .v2_1 import open_suite, v21_dir

SSM_LRS = (3e-4, 1e-3, 3e-3)
GAMMAS = (0.9, 0.99, 0.999)


def _d(tier_name: str, *parts) -> str:
    return os.path.join(v21_dir(tier_name, "ssm_int8"), *parts)


def _job(tier_name: str, out_dir: str, model: str, seed: int, steps: int, ppo: dict, arm: str = "B") -> dict:
    return dict(out_dir=out_dir, arm=arm, model=model, seed=seed, steps=steps, ppo=ppo,
                val_suite=os.path.join(tier_dir(tier_name), "suite_val.json"))


# ------------------------------------------------------------------------------------------ 5b.1 SSM
def lstm_config(tier_name: str) -> dict:
    """The best LSTM config and chunk length from Phase 4 (lstm_retune.json); the v2 tuned config if Phase 4 has not run."""
    import json
    p = os.path.join(v21_dir(tier_name, "diagnose"), "lstm_retune.json")
    if os.path.exists(p):
        r = json.load(open(p))
        return dict(r["winner"]["ppo"])
    return dict(tuned_ppo(tier_name, "lstm"))


def ssm_tune_jobs(tier_name: str, base: dict) -> Dict[str, List[dict]]:
    t = load_tier(tier_name)
    steps = t["lstm_tune_steps"] // 2  # 500k at the full tier
    return {f"ssm_lr{lr:g}": [_job(tier_name, _d(tier_name, "ssm", "tune", f"lr{lr:g}", f"seed_{s}"), "ssm", s, steps,
                                   dict(base, lr=lr)) for s in range(3 if t["lstm_tune_seeds"] >= 3 else t["lstm_tune_seeds"])]
            for lr in SSM_LRS}


def ssm_final_jobs(tier_name: str, cfg: dict) -> List[dict]:
    t = load_tier(tier_name)
    return [_job(tier_name, _d(tier_name, "ssm", "final", f"seed_{s}"), "ssm", s, t["steps"], cfg) for s in range(t["ssm_seeds"])]


def _mean_survival(suite: Suite, model) -> float:
    ev = suite.evaluate(model)
    return float(np.mean(np.concatenate([ev[c]["survival"] for c in CONDITIONS])))


def ssm_stop_rule(tier_name: str, workers: int = 4) -> dict:
    """One tuning pass: 3 learning rates, 3 seeds. If no seed beats random level by more than 20 steps, stop: 'did not train'."""
    base = {k: v for k, v in lstm_config(tier_name).items() if k not in ("lr",)}
    groups = ssm_tune_jobs(tier_name, base)
    run_jobs(train_job, [j for js in groups.values() for j in js], workers)
    suite = Suite.load(os.path.join(tier_dir(tier_name), "suite_val.json"))
    random_level = float(np.mean(np.concatenate([suite.rand[c] for c in CONDITIONS])))
    table, best = {}, None
    for key, js in groups.items():
        surv = [_mean_survival(suite, load_model(j["out_dir"], "ssm")) for j in js]
        table[key] = {"per_seed_mean_survival": surv, "lr": js[0]["ppo"]["lr"], "mean": float(np.mean(surv))}
        if best is None or table[key]["mean"] > table[best]["mean"]:
            best = key
    beat = [s for k in table for s in table[k]["per_seed_mean_survival"] if s > random_level + 20]
    out = {"tier_label": "exploratory", "random_level_steps": random_level, "margin_steps": 20, "tuning": table,
           "best": best, "trained": bool(beat), "seeds_beating_random": len(beat),
           "verdict": "trained" if beat else "did not train: no seed beat random level by more than 20 steps (logs kept; no further tuning)"}
    write_json(_d(tier_name, "ssm_tune.json"), out)
    return out


def ssm_run(tier_name: str, workers: int = 4) -> dict:
    tune = ssm_stop_rule(tier_name, workers)
    ls = lstm_config(tier_name)
    out = {"tier_label": "exploratory", "params": {"ssm": count_params(SSMPolicy()), "lstm": count_params(LSTMPolicy()),
                                                   "ff": count_params(FFPolicy())}, "tune": tune}
    if not tune["trained"]:
        write_json(_d(tier_name, "ssm.json"), out)
        return out
    cfg = dict(ls, lr=tune["tuning"][tune["best"]]["lr"])
    run_jobs(train_job, ssm_final_jobs(tier_name, cfg), workers)
    dirs = [j["out_dir"] for j in ssm_final_jobs(tier_name, cfg)]
    out["config"] = cfg
    for name in ("suite_val.json", "suite_val_s1.json"):
        suite = Suite.load(os.path.join(tier_dir(tier_name), name))
        out[name] = _score(suite, dirs)
    test = open_suite(tier_name, "Phase 5b.1: SSM rung 8 scored on the fresh suite (exploratory)")
    out["suite_test_v2_1.json"] = _score(test, dirs)
    out["inference_cost"] = inference_cost(tier_name)
    write_json(_d(tier_name, "ssm.json"), out)
    return out


def _score(suite: Suite, dirs: List[str], kind: str = "ssm") -> dict:
    sc = {c: [] for c in CONDITIONS}
    for d in dirs:
        ev = suite.evaluate(load_model(d, kind))
        for c in CONDITIONS:
            sc[c].append(per_seed_score(ev[c]))
    sc = {c: np.array(v) for c, v in sc.items()}
    return {"aggregate_iqm": S.iqm_with_ci(sc), "per_seed": S.per_seed_table(sc),
            "per_condition": {c: float(np.mean(sc[c])) for c in CONDITIONS}}


# ------------------------------------------------------------------------------------------ inference cost
def _state_bytes(model) -> int:
    st = model.initial_state(1)
    return 0 if st is None else int(sum(s.numel() * s.element_size() for s in st))


def _latency(model, batch: int, iters: int = 200) -> dict:
    import torch
    torch.set_num_threads(1)
    x, first = torch.rand(batch, P.OBS_DIM), torch.zeros(batch)
    st = model.initial_state(batch)
    with torch.no_grad():
        for _ in range(20):
            model.step(x, st, first)
        t0 = time.perf_counter()
        for _ in range(iters):
            model.step(x, st, first)
    dt = (time.perf_counter() - t0) / iters
    return {"ms_per_call": dt * 1000, "env_steps_per_second": batch / dt}


def inference_cost(tier_name: str) -> dict:
    """State size and speed at batch 1 and 64 for the three architectures, fp32, one thread, on the named machine."""
    from .throughput import cpu_name
    out = {"hardware": cpu_name(), "threads": 1, "policies": {}}
    for name, m in (("ff", FFPolicy()), ("lstm", LSTMPolicy()), ("ssm", SSMPolicy())):
        m.eval()
        out["policies"][name] = {"state_bytes_per_env": _state_bytes(m), "params": count_params(m),
                                 "batch1": _latency(m, 1), "batch64": _latency(m, 64)}
    return out


# ------------------------------------------------------------------------------------------ 5b.2 int8
def quantize(model):
    """Dynamic int8 quantization of the Linear layers (FF, LSTM encoder and heads, SSM projections) and LSTMCell.

    Parts that stay float32: the recurrent state (LSTM h and c, SSM state and conv buffer), activations between layers,
    LayerNorm, the SSM's convolution kernel, A_log and D, and all biases that dynamic quantization keeps in float.
    """
    import torch
    import torch.nn as nn
    layers = {nn.Linear} if isinstance(model, SSMPolicy) else {nn.Linear, nn.LSTMCell}
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), layers, dtype=torch.qint8)
    q.eval()
    return q


def int8_parts(q) -> List[str]:
    return sorted({f"{n or 'root'}:{type(m).__name__}" for n, m in q.named_modules() if "quantized" in type(m).__module__})


def agreement(full, quant, specs, env_config=None) -> float:
    """Fraction of steps where int8 picks the same action as float32, on the float32 policy's own rollouts."""
    import torch
    n = len(specs)
    env = BatchEnv(n, env_config)
    env.reset_all([build_scenario(s) for s in specs])
    live = np.ones(n, bool)
    sf, sq = full.initial_state(n), quant.initial_state(n)
    first = torch.ones(n)
    hit = tot = 0
    while live.any():
        flat = torch.from_numpy(env.flat_obs())
        with torch.no_grad():
            lf, _, sf = full.step(flat, sf, first)
            lq, _, sq = quant.step(flat, sq, first)
        af, aq = lf.argmax(-1).numpy(), lq.argmax(-1).numpy()
        hit += int(((af == aq) & live).sum())
        tot += int(live.sum())
        was = live.copy()
        env.step(np.where(live, af, P.WAIT))
        live &= ~(env.done & was)
        first = torch.from_numpy(env.done.astype(np.float32))
    return hit / max(tot, 1)


def _size(model) -> int:
    import io
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def retention(full: Dict[str, np.ndarray], quant: Dict[str, np.ndarray], B: int = 2000) -> dict:
    """int8 aggregate IQM divided by float32 aggregate IQM; paired bootstrap over seeds and conditions (same resamples)."""
    f = S.agg_iqm_boot(full, np.random.default_rng(11), B)
    q = S.agg_iqm_boot(quant, np.random.default_rng(11), B)
    ratio = q / np.maximum(f, 1e-6)
    lo, hi = S.ci(ratio)
    return {"ratio": S.agg_iqm(quant) / max(S.agg_iqm(full), 1e-6), "lo": lo, "hi": hi,
            "float32_iqm": S.agg_iqm(full), "int8_iqm": S.agg_iqm(quant)}


def int8_run(tier_name: str, groups: Dict[str, dict]) -> dict:
    """groups: name -> {"kind": ff|lstm|ssm, "dirs": [checkpoint dirs]}; scored on suite_val_s1.json (not used for selection)."""
    from .throughput import cpu_name
    tdir = tier_dir(tier_name)
    suite = Suite.load(os.path.join(tdir, "suite_val_s1.json"))
    specs = suite.all_specs()
    out = {"tier_label": "exploratory", "suite": "suite_val_s1.json", "hardware": cpu_name(),
           "method": "torch.ao.quantization.quantize_dynamic, qint8 weights, float32 activations and recurrent state",
           "no_conv_layers": "none of the three policies uses a convolution layer, so static quantization is not needed",
           "rungs": {}}
    for name, g in groups.items():
        models = [load_model(d, g["kind"]) for d in g["dirs"]]
        if not models:
            continue
        q = [quantize(m) for m in models]
        full_sc, q_sc = {c: [] for c in CONDITIONS}, {c: [] for c in CONDITIONS}
        agree = []
        for m, qm in zip(models, q):
            ef, eq = suite.evaluate(m), suite.evaluate(qm)
            for c in CONDITIONS:
                full_sc[c].append(per_seed_score(ef[c]))
                q_sc[c].append(per_seed_score(eq[c]))
            agree.append(agreement(m, qm, specs))
        full_sc = {c: np.array(v) for c, v in full_sc.items()}
        q_sc = {c: np.array(v) for c, v in q_sc.items()}
        out["rungs"][name] = {
            "seeds": len(models), "retention": retention(full_sc, q_sc), "float32_score": S.iqm_with_ci(full_sc),
            "int8_score": S.iqm_with_ci(q_sc), "action_agreement_mean": float(np.mean(agree)), "action_agreement_per_seed": agree,
            "int8_parts": int8_parts(q[0]), "size_fp32_bytes": _size(models[0]), "size_int8_bytes": _size(q[0]),
            "latency_batch1_fp32": _latency(models[0], 1), "latency_batch1_int8": _latency(q[0], 1)}
    write_json(_d(tier_name, "int8.json"), out)
    return out


# ------------------------------------------------------------------------------------------ 5b.3 gamma
def gamma_jobs(tier_name: str) -> Dict[str, List[dict]]:
    t = load_tier(tier_name)
    base = tuned_ppo(tier_name, "ff")
    return {f"gamma_{g:g}_norm": [_job(tier_name, os.path.join(v21_dir(tier_name, "gamma"), f"gamma_{g:g}_norm", f"seed_{s}"),
                                       "ff", s, t["ablation_steps"], dict(base, gamma=g, normalize_returns=True))
                                  for s in range(t["ablation_seeds"])] for g in GAMMAS}


def _v2_dirs(tier_name: str, name: str, n: int) -> List[str]:
    return [os.path.join(tier_dir(tier_name), "ablations", name, f"seed_{s}") for s in range(n)]


def gamma_run(tier_name: str, workers: int = 4) -> dict:
    t = load_tier(tier_name)
    groups = gamma_jobs(tier_name)
    run_jobs(train_job, [j for js in groups.values() for j in js], workers)
    suite = Suite.load(os.path.join(tier_dir(tier_name), "suite_val_s1.json"))  # selection-free validation suite

    def scores(dirs):
        sc = {c: [] for c in CONDITIONS}
        for d in dirs:
            ev = suite.evaluate(load_model(d, "ff"))
            for c in CONDITIONS:
                sc[c].append(per_seed_score(ev[c]))
        return {c: np.array(v) for c, v in sc.items()}
    new = {g: scores([j["out_dir"] for j in groups[f"gamma_{g:g}_norm"]]) for g in GAMMAS}
    old = {0.9: scores(_v2_dirs(tier_name, "gamma_0.9", t["ablation_seeds"])),
           0.99: scores(_v2_dirs(tier_name, "default", t["ablation_seeds"])),
           0.999: scores(_v2_dirs(tier_name, "gamma_0.999", t["ablation_seeds"]))}
    d_new = S.iqm_difference_with_ci(new[0.99], new[0.9])      # positive: gamma 0.9 hurts
    d_old = S.iqm_difference_with_ci(old[0.99], old[0.9])
    hi_new = S.iqm_difference_with_ci(new[0.999], new[0.99])   # positive: gamma 0.999 helps
    hi_old = S.iqm_difference_with_ci(old[0.999], old[0.99])
    shrinks = d_new["diff"] < 0.7 * d_old["diff"] or d_new["lo"] <= 0
    out = {"tier_label": "exploratory", "suite": "suite_val_s1.json", "steps": t["ablation_steps"], "seeds": t["ablation_seeds"],
           "normalized": {str(g): {"aggregate_iqm": S.iqm_with_ci(new[g])} for g in GAMMAS},
           "v2_unnormalized": {str(g): {"aggregate_iqm": S.iqm_with_ci(old[g])} for g in GAMMAS},
           "gamma_0.9_penalty": {"v2": d_old, "normalized": d_new},
           "gamma_0.999_gain": {"v2": hi_old, "normalized": hi_new},
           "rows": {"gamma_0.9_still_hurts": bool(d_new["lo"] > 0 and not shrinks),
                    "penalty_shrinks_or_vanishes": bool(shrinks),
                    "gamma_0.999_now_helps": bool(hi_new["lo"] > 0)},
           "rule": "penalty shrinks if the normalized 0.99-minus-0.9 difference is under 70% of the v2 one or its interval includes zero; "
                   "0.999 helps if the normalized 0.999-minus-0.99 interval is above zero"}
    write_json(os.path.join(v21_dir(tier_name, "gamma"), "gamma.json"), out)
    return out
