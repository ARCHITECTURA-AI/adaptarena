"""Memory track: does recurrence close the gap a reactive policy cannot?

Predictions were written into analysis_plan.yaml and committed before the first memory run; this module
scores each one as hit / partial / miss against the rule stated there. It also runs the recurrent-state
linear probe and the int8 compression stretch.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from ..core import params as P
from ..core.baselines import C as _C, MEMORY_EST_DELAY, MemoryScript
from ..core.batch import BatchEnv
from ..core.distributions import CONDITIONS, MEMORY_CONDITIONS
from ..core.env import EnvConfig, SurvivalEnv
from . import stats as S
from .experiments import (get_val_suite, load_model, load_tier, run_jobs, tier_dir, train_job, write_json)
from .suites import open_locked_test_suite, per_seed_score
from .tune import tuned_ppo

VARIANTS = {"vision3": {"vision": 3}, "nocost": {"cost_stat": False}}


def jobs(tier_name: str, include_ssm: bool) -> List[dict]:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    vpath = os.path.join(tdir, "suite_val.json")
    out = []
    kinds = ["lstm"] + (["ssm"] if include_ssm else [])
    for kind in kinds:
        for s in range(tier["memory_seeds"] if kind == "lstm" else min(5, tier["memory_seeds"])):
            out.append(dict(out_dir=os.path.join(tdir, "memory", kind, "default", f"seed_{s}"), arm="B", model=kind,
                            seed=s, steps=tier["memory_steps"], ppo=tuned_ppo(tier_name, kind), val_suite=vpath))
    for v, env in VARIANTS.items():
        for kind in ("ff", "lstm"):
            for s in range(tier["memory_variant_seeds"]):
                out.append(dict(out_dir=os.path.join(tdir, "memory", kind, v, f"seed_{s}"), arm="B", model=kind,
                                seed=s, steps=tier["memory_variant_steps"], env=env,
                                ppo=tuned_ppo(tier_name, kind), val_suite=vpath))
    return out


def _scores(suite, models, env_cfg, conds) -> Dict[str, np.ndarray]:
    per = {c: [] for c in conds}
    for m in models:
        ev = suite.evaluate(m, env_cfg)
        for c in conds:
            per[c].append(per_seed_score(ev[c]))
    return {c: np.array(v) for c, v in per.items()}


def _ff_default_models(tdir, tier):
    out = []
    for s in range(tier["seeds"]):
        d = os.path.join(tdir, "headline", f"arm_B", f"seed_{s}")
        if os.path.exists(os.path.join(d, "model_best.pt")):
            out.append(load_model(d, "ff"))
    return out


def _group(tdir, kind, variant, n):
    out = []
    for s in range(n):
        d = os.path.join(tdir, "memory", kind, variant, f"seed_{s}")
        if os.path.exists(os.path.join(d, "model_best.pt")):
            out.append(load_model(d, kind))
    return out


def run(tier_name: str, workers: int = 4, include_ssm: Optional[bool] = None) -> dict:
    if include_ssm is None:
        include_ssm = bool(os.environ.get("ADAPTARENA_SSM"))
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    get_val_suite(tdir, tier, 0)
    run_jobs(train_job, jobs(tier_name, include_ssm), workers)
    n_loc = tier["scenarios_per_condition"]
    main = open_locked_test_suite(tdir, "memory track: default conditions", n_loc, generate=True)
    reloc = open_locked_test_suite(tdir, "memory track: relocation condition", n_loc, generate=True,
                                   name="suite_test_memory.json", conditions=MEMORY_CONDITIONS)
    ff = _ff_default_models(tdir, tier)
    lstm = _group(tdir, "lstm", "default", tier["memory_seeds"])
    res: dict = {"seeds": {"ff_default": len(ff), "lstm_default": len(lstm)}}
    cmp = {}

    def diff(a, b):
        return S.iqm_difference_with_ci(a, b, B=4000)

    sc_l, sc_f = _scores(main, lstm, None, CONDITIONS), _scores(main, ff, None, CONDITIONS)
    sc_l_r, sc_f_r = _scores(reloc, lstm, None, MEMORY_CONDITIONS), _scores(reloc, ff, None, MEMORY_CONDITIONS)
    cmp["familiar"] = diff({"familiar": sc_l["familiar"]}, {"familiar": sc_f["familiar"]})
    cmp["unseen_layout"] = diff({"unseen_layout": sc_l["unseen_layout"]}, {"unseen_layout": sc_f["unseen_layout"]})
    cmp["resource_relocation"] = diff(sc_l_r, sc_f_r)
    cmp["cold_snap"] = diff({"cold_snap": sc_l["cold_snap"]}, {"cold_snap": sc_f["cold_snap"]})
    for v, env in VARIANTS.items():
        cfg = EnvConfig(**env)
        lv, fv = _group(tdir, "lstm", v, tier["memory_variant_seeds"]), _group(tdir, "ff", v, tier["memory_variant_seeds"])
        conds = ["familiar", "unseen_layout"] if v == "vision3" else ["higher_move_cost"]
        a, b = _scores(main, lv, cfg, conds), _scores(main, fv, cfg, conds)
        cmp[v] = diff(a, b)
        res.setdefault("variant_scores", {})[v] = {"lstm": {c: x.tolist() for c, x in a.items()},
                                                   "ff": {c: x.tolist() for c, x in b.items()}}
    res["comparisons"] = cmp
    res["raw_scores"] = {"lstm": {c: x.tolist() for c, x in sc_l.items()}, "ff": {c: x.tolist() for c, x in sc_f.items()},
                         "lstm_relocation": sc_l_r["resource_relocation"].tolist(), "ff_relocation": sc_f_r["resource_relocation"].tolist()}
    res["predictions"] = score_predictions(cmp)
    if include_ssm:
        ssm = _group(tdir, "ssm", "default", 5)
        if ssm:
            res["ssm"] = {"seeds": len(ssm), "iqm_all_conditions": S.iqm_with_ci(_scores(main, ssm, None, CONDITIONS))}
    res["recurrent_probe"] = recurrent_probe(lstm, get_val_suite(tdir, tier, 0)) if lstm else None
    res["compression"] = compression(tdir, tier, ff, lstm, _group(tdir, "ssm", "default", 5) if include_ssm else [])
    write_json(os.path.join(tdir, "memory", "memory.json"), res)
    return res


def score_predictions(cmp: dict) -> list:
    out = []

    def st(ok, partial):
        return "hit" if ok else ("partial" if partial else "miss")

    f, u = cmp["familiar"], cmp["unseen_layout"]
    out.append({"shift": "familiar and unseen layouts", "predicted": "recurrent ahead",
                "observed": {"familiar": f, "unseen_layout": u},
                "status": st(f["lo"] > 0 and u["lo"] > 0, f["diff"] > 0 or u["diff"] > 0)})
    r = cmp["resource_relocation"]["diff"]
    lead_u = u["diff"]
    if lead_u > 0:
        ratio = r / lead_u
        status = st(ratio < 0.5, ratio < 0.8)
    else:
        ratio, status = None, "miss"  # no lead to lose
    out.append({"shift": "resource relocation", "predicted": "recurrent loses most of its lead",
                "observed": {"relocation_lead": r, "unseen_lead": lead_u, "ratio": ratio}, "status": status})
    v = cmp["vision3"]
    out.append({"shift": "vision reduced to 3x3", "predicted": "recurrent clearly ahead", "observed": v,
                "status": st(v["lo"] > 0 and v["diff"] >= 0.10, v["diff"] > 0)})
    n = cmp["nocost"]
    out.append({"shift": "movement-cost increase, energy-cost stat removed", "predicted": "recurrent holds survival better",
                "observed": n, "status": st(n["lo"] > 0, n["diff"] > 0)})
    c = cmp["cold_snap"]
    out.append({"shift": "cold snap", "predicted": "no separation", "observed": c,
                "status": "hit" if c["lo"] <= 0 <= c["hi"] else "miss"})
    return out


# ----------------------------------------------------------------------------- recurrent-state probe
def recurrent_probe(models, suite, n_models: int = 3) -> dict:
    """Linear probes on the LSTM hidden state, with a shuffled-label control and an observation-only control."""
    import torch
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score

    specs = suite.all_specs()
    rng = np.random.default_rng(0)
    out = {"reading": "shows whether memory holds task information, and nothing more", "models": []}
    for m in models[:n_models]:
        H, X_obs, y_steps, y_dir, ep_ids = [], [], [], [], []
        for ep, spec in enumerate(specs):
            env = SurvivalEnv()
            obs = env.reset(spec)
            state, first = m.initial_state(1), torch.ones(1)
            last_seen, last_pos = None, None
            while not env.done:
                flat = torch.from_numpy(np.concatenate([obs["grid"].reshape(-1), obs["stats"]])[None])
                if obs["grid"][P.CH_WATER].any() or obs["grid"][P.CH_WATER_DEPLETED].any():
                    ys, xs = np.nonzero(obs["grid"][P.CH_WATER] + obs["grid"][P.CH_WATER_DEPLETED])
                    last_seen = env.t
                    last_pos = (env.r + int(ys[0]) - P.VIEW_R, env.c + int(xs[0]) - P.VIEW_R)
                with torch.no_grad():
                    logits, _, state = m.step(flat, state, first)
                first = torch.zeros(1)
                if last_seen is not None:
                    dy, dx = last_pos[0] - env.r, last_pos[1] - env.c
                    d = 4 if (dy == 0 and dx == 0) else (int(dy > 0) if abs(dy) >= abs(dx) else 2 + int(dx > 0))
                    H.append(state[0][0].numpy().copy())
                    X_obs.append(flat[0].numpy().copy())
                    y_steps.append(min(env.t - last_seen, 150))
                    y_dir.append(d)
                    ep_ids.append(ep)
                obs, *_ = env.step(int(logits.argmax(-1)))
        H, X_obs, y_steps, y_dir, ep_ids = map(np.array, (H, X_obs, y_steps, y_dir, ep_ids))
        test_eps = set(rng.choice(len(specs), len(specs) // 2, replace=False).tolist())
        te = np.isin(ep_ids, list(test_eps))
        tr = ~te
        if tr.sum() < 50 or te.sum() < 50:
            continue
        rec = {}
        for name, X in (("hidden", H), ("observation_only", X_obs)):
            reg = Ridge(alpha=1.0).fit(X[tr], y_steps[tr])
            clf = LogisticRegression(max_iter=300).fit(X[tr], y_dir[tr])
            ctrl_reg = Ridge(alpha=1.0).fit(X[tr], rng.permutation(y_steps[tr]))
            ctrl_clf = LogisticRegression(max_iter=300).fit(X[tr], rng.permutation(y_dir[tr]))
            rec[name] = {"steps_since_water_r2": float(r2_score(y_steps[te], reg.predict(X[te]))),
                         "steps_since_water_r2_shuffled": float(r2_score(y_steps[te], ctrl_reg.predict(X[te]))),
                         "direction_accuracy": float(accuracy_score(y_dir[te], clf.predict(X[te]))),
                         "direction_accuracy_shuffled": float(accuracy_score(y_dir[te], ctrl_clf.predict(X[te]))),
                         "majority_class_accuracy": float(np.bincount(y_dir[te]).max() / te.sum())}
        rec["n_train"], rec["n_test"] = int(tr.sum()), int(te.sum())
        out["models"].append(rec)
    return out


# ----------------------------------------------------------------------------- compression stretch
def _quantize(model):
    import torch
    import torch.nn as nn
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTMCell}, dtype=torch.qint8)


def _size_bytes(model) -> int:
    import io
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def compression(tdir, tier, ff, lstm, ssm) -> dict:
    """Post-training int8 quantization of rungs 5, 7, 8: survival retention with CIs, size, latency on named hardware."""
    import torch
    from .throughput import cpu_name
    suite = get_val_suite(tdir, tier, 0)
    out = {"label": "stretch goal, exploratory, validation suite", "hardware": cpu_name(), "rungs": {}}
    for rung, name, models in ((5, "ppo_ff", ff), (7, "ppo_lstm", lstm), (8, "ppo_ssm", ssm)):
        if not models:
            continue
        try:
            q = [_quantize(m) for m in models]
        except Exception as e:  # pragma: no cover
            out["rungs"][name] = {"error": str(e)}
            continue
        full = _scores(suite, models, None, CONDITIONS)
        quant = _scores(suite, q, None, CONDITIONS)
        retention = {c: quant[c] / np.maximum(full[c], 1e-6) for c in CONDITIONS}
        agg = S.iqm_with_ci(retention)
        x = torch.rand(8, P.OBS_DIM)

        def lat(m):
            st = m.initial_state(8)
            first = torch.zeros(8)
            m.step(x, st, first)
            t0 = time.perf_counter()
            for _ in range(200):
                m.step(x, st, first)
            return (time.perf_counter() - t0) / 200 * 1000

        torch.set_num_threads(1)
        out["rungs"][name] = {"survival_retention_iqm": agg, "size_fp32_bytes": _size_bytes(models[0]),
                              "size_int8_bytes": _size_bytes(q[0]), "latency_ms_fp32_batch8": lat(models[0]),
                              "latency_ms_int8_batch8": lat(q[0]), "seeds": len(models)}
    return out


# ----------------------------------------------------------------------------- hand-fed memory (Phase 4.2)
MAP_CHANNELS = 3  # remembered food sites, remembered water sites, known floor (this environment has no shelters)
MAP_DIM = MAP_CHANNELS * P.VIEW * P.VIEW


class MapTracker(MemoryScript):
    """The memory script's dead-reckoning map with its decision logic removed.

    `core/baselines.py` is frozen, so the map code is imported through subclassing, not copied. `update` does what the
    first half of `MemoryScript.act` does (move the dead-reckoned position when the previous action was a valid
    move, then fold the 9x9 view into the map) and stops there. The script's one action-coupled step, marking a
    site depleted when it chooses INTERACT, is not applied here: the next observation shows the true state.
    """
    name = "map_tracker"

    def update(self, obs: dict) -> None:
        g, s = obs["grid"], obs["stats"]
        if s[P.S_PREV_ACTION].any():
            pa = int(np.argmax(s[P.S_PREV_ACTION]))
            if pa < 4 and s[P.S_PREV_VALID] > 0.5:
                dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[pa]
                self.y += dy
                self.x += dx
        self._observe(g)
        self.t += 1

    def channels(self) -> np.ndarray:
        """(3, 9, 9) egocentric window centred on the current dead-reckoned cell.

        Sites: 1.0 if last seen ready; if last seen depleted, 0.1 + 0.4 * min(1, age / MEMORY_EST_DELAY), so a
        remembered-but-empty site is distinguishable from an unknown cell and ripens at the script's assumed delay.
        Known floor: 1.0 for every cell the walker has seen as floor.
        """
        out = np.zeros((MAP_CHANNELS, P.VIEW, P.VIEW), dtype=np.float32)
        for dy in range(-_C, _C + 1):
            for dx in range(-_C, _C + 1):
                cell = (self.y + dy, self.x + dx)
                if self.floor.get(cell, False):
                    out[2, _C + dy, _C + dx] = 1.0
                rec = self.sites.get(cell)
                if rec is None:
                    continue
                if rec["ready"]:
                    v = 1.0
                else:
                    age = max(0, self.t - 1 - (rec["seen_dep"] if rec["seen_dep"] is not None else self.t - 1))
                    v = 0.1 + 0.4 * min(1.0, age / MEMORY_EST_DELAY)
                out[0 if rec["kind"] == "food" else 1, _C + dy, _C + dx] = v
        return out


class MapBatchEnv(BatchEnv):
    """BatchEnv whose flat observation is the v2 vector (659) followed by the 3 map channels (243): obs-v9-map, 902 dims.

    The simulator is untouched. One MapTracker per environment is advanced inside `flat_obs`, once per new state
    (keyed on the step counter), so repeated calls on the same state are harmless.
    """

    def __init__(self, n, config=None, **kw):
        super().__init__(n, config, **kw)
        self._trk = [MapTracker() for _ in range(n)]
        for k in self._trk:
            k.reset(0)
        self._stamp = np.full(n, -1, dtype=np.int64)
        self._maps = np.zeros((n, MAP_DIM), dtype=np.float32)

    def reset_env(self, i, scenario):
        super().reset_env(i, scenario)
        self._trk[i].reset(0)
        self._stamp[i] = -1

    def flat_obs(self):
        o = self.observe()
        for i in range(self.n):
            if self._stamp[i] != self.t[i]:
                self._trk[i].update({"grid": o["grid"][i], "stats": o["stats"][i]})
                self._maps[i] = self._trk[i].channels().reshape(-1)
                self._stamp[i] = self.t[i]
        return np.concatenate([o["grid"].reshape(self.n, -1), o["stats"], self._maps], axis=1)


OBS_VARIANTS = {"v2": (None, P.OBS_DIM), "v9-map": (MapBatchEnv, P.OBS_DIM + MAP_DIM)}


def env_factory_for(variant: str):
    """None for the v2 observation (BatchEnv is the default), MapBatchEnv for obs-v9-map."""
    return OBS_VARIANTS[variant][0]
