"""Failure museum: point of no return, blind-or-blunder label and templated narration.

No language model is involved: every sentence is a fixed template filled from a logged value.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from ..core import params as P
from ..core.env import SurvivalEnv
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario
from .replays import record_replay, snapshot_env, state_at, strip_private

SNAP = 25


def _has_certificate(sc, actions, k) -> bool:
    env = state_at(sc, actions, k)
    if env.done and not env.alive:
        return False
    return plan(sc, max_nodes=2500, init=snapshot_env(env)).certified


def point_of_no_return(sc, actions) -> dict:
    """Fork the recorded state every 25 steps and ask the planner for a certificate."""
    last_k = len(actions)
    ks = list(range(0, last_k + 1, SNAP))
    good = None
    nxt = None
    for k in ks:
        if _has_certificate(sc, actions, k):
            good = k
        else:
            nxt = k
            if good is not None:
                break
    if good is None:
        return {"step": None, "snapshots_checked": len(ks)}
    if nxt is None:
        nxt = last_k
    lo, hi = good, nxt  # certificate at lo, none at hi: bisect
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _has_certificate(sc, actions, mid):
            lo = mid
        else:
            hi = mid
    return {"step": lo, "snapshots_checked": len(ks), "valid_for": "the planner's search power, not a proof that no plan exists"}


def _nearest(sc, env: SurvivalEnv, kind: str) -> Optional[int]:
    t = env.t
    if kind == "water":
        cells = [tuple(p) for i, p in enumerate(sc.water_positions(t)) if t < sc.water_until[i] and env.water_ready[i] <= t]
    else:
        cells = [tuple(p) for i, p in enumerate(sc.food_pos) if t < sc.food_until[i] and env.food_ready[i] <= t]
    if not cells:
        return None
    dist = {(env.r, env.c): 0}
    dq = deque([(env.r, env.c)])
    while dq:
        y, x = dq.popleft()
        if (y, x) in cells:
            return dist[(y, x)]
        for a in range(4):
            ny, nx = y + P.DR[a], x + P.DC[a]
            if 0 <= ny < P.GRID and 0 <= nx < P.GRID and not sc.terrain[ny, nx] and (ny, nx) not in dist:
                dist[(ny, nx)] = dist[(y, x)] + 1
                dq.append((ny, nx))
    return None


def _view_has(env: SurvivalEnv, kind: str) -> bool:
    g = env.observation()["grid"]
    return bool(g[P.CH_WATER if kind == "water" else P.CH_FOOD].any())


def _sentence(sc, env: SurvivalEnv, t: int, label: str) -> str:
    nw, nf = _nearest(sc, env, "water"), _nearest(sc, env, "food")
    vw, vf = _view_has(env, "water"), _view_has(env, "food")
    w = ("water in view" if vw else "no water in view") + (f", nearest ready water {nw} tiles away" if nw is not None else ", no ready water anywhere")
    f = ("food in view" if vf else "no food in view") + (f", nearest ready food {nf} tiles away" if nf is not None else ", no ready food anywhere")
    return f"Step {t}: hydration {env.hydration // 100}, energy {env.energy // 100}, {w}; {f}. ({label})"


def narrate(spec: ScenarioSpec, actions: List[int], cause: str, pnr: Optional[int], blind: Optional[str]) -> List[str]:
    sc = build_scenario(spec)
    lines = [f"Scenario {sc.sid}: {sc.nf} food sites, {sc.nw} water sites, movement cost {spec.move_cost_pct}%, "
             f"cold {spec.cold_pct}% of steps, {int(sc.terrain.sum())} wall tiles."]
    env = SurvivalEnv()
    env.reset(sc)
    first_low = {"hydration": None, "energy": None}
    marks = {}
    for t, a in enumerate(actions):
        if env.hydration < 3000 and first_low["hydration"] is None:
            first_low["hydration"] = t
            marks[t] = "hydration fell below 30"
        if env.energy < 3000 and first_low["energy"] is None:
            first_low["energy"] = t
            marks[t] = marks.get(t, "") or "energy fell below 30"
        if pnr is not None and t == pnr:
            marks[t] = "last step with a survival plan"
        if t in marks:
            lines.append(_sentence(sc, env, t, marks[t]))
        if env.done:
            break
        env.step(a)
    lines.append(_sentence(sc, env, env.t, f"died of {cause}"))
    if pnr is not None:
        lines.append(f"The reference planner could still certify survival at step {pnr} and could not afterwards; "
                     f"label: {blind} (heuristic suggestion).")
    return lines


def build_entry(spec: ScenarioSpec, policy, label: str) -> dict:
    rep = record_replay(spec, policy, label=label, with_probs=True)
    sc = build_scenario(spec)
    pnr = point_of_no_return(sc, rep["actions"])
    step = pnr["step"]
    blind = None
    if step is not None and not rep["alive"]:
        env = state_at(sc, rep["actions"], step)
        need = {"dehydration": ["water"], "starvation": ["food"]}.get(rep["cause"], ["water", "food"])
        seen = any(_view_has(env, k) and _nearest(sc, env, k) is not None for k in need)
        blind = "blunder" if seen else "blind"
    return {"label": label, "spec": spec.to_json(), "survival": rep["survival"], "cause": rep["cause"],
            "point_of_no_return": pnr, "blind_or_blunder": blind,
            "narration": narrate(spec, rep["actions"], rep["cause"], step, blind), "replay": strip_private(rep)}
