"""Statistics following Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" (2021).

The unit of analysis is the training seed within a condition. Episodes inside a seed are averaged
first (by the caller), so 50 episodes are never counted as 50 independent trained models.
Bootstrap resamples are stratified by condition: seeds are resampled with replacement
independently inside each condition.

Inputs are dicts `{condition: array of per-seed scores}`.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import trim_mean

N_BOOT = 10_000
Scores = Dict[str, np.ndarray]


def iqm(x) -> float:
    return float(trim_mean(np.asarray(x, dtype=float).ravel(), 0.25))


def _iqm_rows(m: np.ndarray) -> np.ndarray:
    return trim_mean(m, 0.25, axis=1)


def _resample(scores: Scores, rng: np.random.Generator, B: int) -> Dict[str, np.ndarray]:
    """For each condition, B resamples of its seed scores -> (B, n_seeds)."""
    out = {}
    for c, v in scores.items():
        v = np.asarray(v, dtype=float)
        out[c] = v[rng.integers(0, len(v), size=(B, len(v)))]
    return out


def agg_iqm(scores: Scores) -> float:
    return iqm(np.concatenate([np.asarray(v, float).ravel() for v in scores.values()]))


def agg_iqm_boot(scores: Scores, rng, B: int = N_BOOT) -> np.ndarray:
    rs = _resample(scores, rng, B)
    return _iqm_rows(np.concatenate(list(rs.values()), axis=1))


def ci(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def iqm_with_ci(scores: Scores, seed: int = 0, B: int = N_BOOT) -> dict:
    rng = np.random.default_rng([0x57A7, seed])
    boot = agg_iqm_boot(scores, rng, B)
    lo, hi = ci(boot)
    return {"iqm": agg_iqm(scores), "lo": lo, "hi": hi}


def optimality_gap(scores: Scores, threshold: float = 1.0) -> float:
    allv = np.concatenate([np.asarray(v, float).ravel() for v in scores.values()])
    return float(np.mean(np.maximum(threshold - allv, 0.0)))


def optimality_gap_with_ci(scores: Scores, seed: int = 0, B: int = N_BOOT) -> dict:
    rng = np.random.default_rng([0x57A8, seed])
    rs = _resample(scores, rng, B)
    boot = np.maximum(1.0 - np.concatenate(list(rs.values()), axis=1), 0.0).mean(1)
    lo, hi = ci(boot)
    return {"gap": optimality_gap(scores), "lo": lo, "hi": hi}


def prob_improvement(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) with ties counted half, over all pairs of seeds."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def prob_improvement_with_ci(sx: Scores, sy: Scores, seed: int = 0, B: int = N_BOOT) -> dict:
    """Average over conditions of P(X>Y); stratified bootstrap on both arms independently."""
    rng = np.random.default_rng([0x57A9, seed])
    conds = list(sx)
    point = float(np.mean([prob_improvement(sx[c], sy[c]) for c in conds]))
    rx, ry = _resample(sx, rng, B), _resample(sy, rng, B)
    per = []
    for c in conds:
        a, b = rx[c], ry[c]
        gt = (a[:, :, None] > b[:, None, :]).mean((1, 2))
        eq = (a[:, :, None] == b[:, None, :]).mean((1, 2))
        per.append(gt + 0.5 * eq)
    boot = np.mean(per, axis=0)
    lo, hi = ci(boot)
    return {"p": point, "lo": lo, "hi": hi}


def iqm_difference_with_ci(sx: Scores, sy: Scores, seed: int = 0, B: int = N_BOOT) -> dict:
    rng = np.random.default_rng([0x57AA, seed])
    bx, by = agg_iqm_boot(sx, rng, B), agg_iqm_boot(sy, rng, B)
    d = bx - by
    lo, hi = ci(d)
    return {"diff": agg_iqm(sx) - agg_iqm(sy), "lo": lo, "hi": hi}


def performance_profile(scores: Scores, taus: Sequence[float], seed: int = 0, B: int = 2000) -> dict:
    """Fraction of (seed, condition) runs scoring above each tau, with a stratified bootstrap band."""
    rng = np.random.default_rng([0x57AB, seed])
    allv = np.concatenate([np.asarray(v, float).ravel() for v in scores.values()])
    taus = np.asarray(taus, float)
    point = (allv[None, :] > taus[:, None]).mean(1)
    rs = np.concatenate(list(_resample(scores, rng, B).values()), axis=1)  # (B, N)
    boot = (rs[:, None, :] > taus[None, :, None]).mean(2)  # (B, T)
    return {"tau": taus.tolist(), "frac": point.tolist(),
            "lo": np.quantile(boot, 0.025, axis=0).tolist(), "hi": np.quantile(boot, 0.975, axis=0).tolist()}


def per_seed_table(scores: Scores) -> list:
    rows = []
    for c, v in scores.items():
        for i, x in enumerate(v):
            rows.append({"condition": c, "seed": i, "score": float(x)})
    return rows


# ------------------------------------------------------------------ survival analysis
def kaplan_meier(times: np.ndarray, died: np.ndarray, horizon: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """KM estimate of P(survive beyond t). Episodes alive at the horizon are right-censored."""
    times, died = np.asarray(times, int), np.asarray(died, bool)
    t_grid = np.arange(0, horizon + 1)
    S = np.ones(horizon + 1)
    surv = 1.0
    for t in range(1, horizon + 1):
        # at risk at step t: still alive after t-1 steps, i.e. survival >= t-1
        at_risk = (times >= t - 1).sum()
        events = (died & (times == t - 1)).sum()
        if at_risk > 0:
            surv *= 1 - events / at_risk
        S[t] = surv
    return t_grid, S


def censored_fraction(times: np.ndarray, horizon: int = 500) -> float:
    return float((np.asarray(times) >= horizon).mean())


def breaking_point(levels: Sequence[float], mean_survival: Sequence[float], threshold: float):
    """Lowest intensity level whose survival falls below `threshold`; None if it never does."""
    for lv, s in zip(levels, mean_survival):
        if s < threshold:
            return lv
    return None
