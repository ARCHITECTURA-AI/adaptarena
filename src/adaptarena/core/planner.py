"""Reference planner: full-information search for a plan that survives the horizon.

The planner sees the whole map, every respawn delay and the weather/cost schedules.
Between decisions the only useful things an agent can do are "walk the shortest path to
a site, wait there until it is ready, interact", so the search runs over those macro
moves with exact integer bookkeeping. A found plan is a *certificate*: `verify_plan`
replays it in the reference environment and checks the agent really survives.

The planner is a strong reference, not an optimality claim:
  * hazard tiles are treated as walls (conservative; a plan found is still valid),
  * the search is depth-first with a node cap, so a scenario that has a plan beyond the
    cap is reported as "no certificate", not as unsolvable.
On tiny maps with a generous cap it is cross-checked against brute force (tests).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import params as P
from .env import SurvivalEnv
from .scenario import Scenario


@dataclass
class PlanResult:
    certified: bool
    survival: int  # steps survived by the best plan found (== horizon when certified)
    actions: List[int]
    nodes: int
    capped: bool  # the node cap was hit before the search finished


def _bfs(terrain: np.ndarray, blocked: np.ndarray, src) -> np.ndarray:
    """Distance map from src; -1 for unreachable cells."""
    n = P.GRID
    dist = np.full((n, n), -1, dtype=np.int32)
    dist[src[0], src[1]] = 0
    dq = deque([(src[0], src[1])])
    while dq:
        y, x = dq.popleft()
        d = dist[y, x] + 1
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and dist[ny, nx] < 0 and not terrain[ny, nx] and not blocked[ny, nx]:
                dist[ny, nx] = d
                dq.append((ny, nx))
    return dist


def _path_actions(dist_to_target: np.ndarray, src) -> List[int]:
    """Greedy descent on a BFS distance map rooted at the target."""
    y, x = src
    out = []
    while dist_to_target[y, x] > 0:
        d = dist_to_target[y, x]
        for a in range(4):
            ny, nx = y + P.DR[a], x + P.DC[a]
            if 0 <= ny < P.GRID and 0 <= nx < P.GRID and dist_to_target[ny, nx] == d - 1:
                out.append(a)
                y, x = ny, nx
                break
    return out


def plan(sc: Scenario, max_nodes: int = 4000, init: Optional[dict] = None) -> PlanResult:
    """Search from the scenario start, or from a mid-episode `init` state
    {t, pos, energy, hydration, food_ready, water_ready} (used by the point-of-no-return probe)."""
    H = sc.horizon
    nf, nw = sc.nf, sc.nw
    blocked = sc.hazard.astype(bool)  # hazard tiles are never entered by the planner

    # per-step drains (cumulative), so any wait/move block is O(1)
    cold = sc.cold.astype(np.int64)
    cost = sc.cost_pct.astype(np.int64)
    e_wait = P.BASE_ENERGY_DRAIN + P.COLD_EXTRA_ENERGY * cold
    e_move = e_wait + P.MOVE_EXTRA_ENERGY * cost // 100
    cum_wait = np.concatenate([[0], np.cumsum(e_wait)])
    cum_move = np.concatenate([[0], np.cumsum(e_move)])

    # site table: (kind, index, positions (before, after relocation), delay, until)
    sites = []
    for i in range(nf):
        sites.append(dict(kind=0, pos=(tuple(sc.food_pos[i]),) * 2, delay=int(sc.food_delay[i]),
                          until=int(sc.food_until[i])))
    for i in range(nw):
        sites.append(dict(kind=1, pos=(tuple(sc.water_pos[i]), tuple(sc.water_pos2[i])),
                          delay=int(sc.water_delay[i]), until=int(sc.water_until[i])))
    S = len(sites)
    dist_cache: dict = {}

    def dist_map(pos):
        key = (int(pos[0]), int(pos[1]))
        if key not in dist_cache:
            dist_cache[key] = _bfs(sc.terrain, blocked, key)
        return dist_cache[key]

    reloc = sc.reloc_step

    def site_pos(s, t):
        return sites[s]["pos"][1 if (reloc >= 0 and t >= reloc and sites[s]["kind"] == 1) else 0]

    def idle_survival(t, e, h):
        """Steps survived (counted like the env) if the agent just waits from time t."""
        k = 0
        # binary search the first step at which a stat would hit zero
        lo, hi = 0, H - t
        def ok(m):
            return e - (cum_wait[t + m] - cum_wait[t]) > 0 and h - P.BASE_HYDRATION_DRAIN * m > 0
        if ok(hi):
            return hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    best = {"surv": -1, "t": 0, "macros": []}
    nodes = 0
    capped = False
    seen = set()

    def expand(t, pos, e, h, fr, wr):
        """Candidate macro moves ordered by urgency. Yields (site, travel, wait, new state...)."""
        td_e = e // 100  # rough steps of energy left at ~1 unit/step
        td_h = h // 60
        cands = []
        for s in range(S):
            sd = sites[s]
            if sd["kind"] == 0:
                ready = int(fr[s])
            else:
                ready = int(wr[s - nf])
            p = site_pos(s, t)
            dm = dist_map(p)
            d = int(dm[pos[0], pos[1]])
            if d < 0:
                continue
            arrive = t + d
            act = max(arrive, ready)
            if reloc >= 0 and t < reloc <= act:
                continue  # site would move under us; handled by waiting out the event instead
            if act >= H or act >= sd["until"]:
                continue
            urgency = td_e if sd["kind"] == 0 else td_h
            cands.append((urgency, act, s, d, ready, p))
        cands.sort(key=lambda x: (x[0], x[1]))
        return cands

    def dfs(t, pos, e, h, fr, wr, macros):
        nonlocal nodes, capped
        if best["surv"] >= H:
            return
        nodes += 1
        if nodes > max_nodes:
            capped = True
            return
        idle = idle_survival(t, e, h)
        if t + idle > best["surv"]:
            best["surv"], best["t"] = t + idle, t
            best["macros"] = list(macros)
            best["tail"] = idle
        if t + idle >= H:
            return
        key = (t, pos, e, h, fr, wr)
        if key in seen:
            return
        seen.add(key)
        for _, act, s, d, ready, p in expand(t, pos, e, h, fr, wr):
            sd = sites[s]
            # travel d move-steps
            e1 = e - (cum_move[t + d] - cum_move[t])
            h1 = h - P.BASE_HYDRATION_DRAIN * d
            if e1 <= 0 or h1 <= 0:
                continue  # idling from this node already covers any shorter survival
            arrive = t + d
            w = act - arrive
            e2 = e1 - (cum_wait[act] - cum_wait[arrive])
            h2 = h1 - P.BASE_HYDRATION_DRAIN * w
            if e2 <= 0 or h2 <= 0:
                continue
            # interact step at index `act`: restore, then drain (base + cold)
            if sd["kind"] == 0:
                e3 = min(P.STAT_MAX, e2 + P.FOOD_RESTORE) - int(e_wait[act])
                h3 = h2 - P.BASE_HYDRATION_DRAIN
                nfr = list(fr); nfr[s] = act + 1 + sd["delay"]
                nwr = wr
                nfr, nwr = tuple(nfr), nwr
            else:
                e3 = e2 - int(e_wait[act])
                h3 = min(P.STAT_MAX, h2 + P.WATER_RESTORE) - P.BASE_HYDRATION_DRAIN
                nwr = list(wr); nwr[s - nf] = act + 1 + sd["delay"]
                nfr, nwr = fr, tuple(nwr)
            if e3 <= 0 or h3 <= 0:
                continue
            macros.append((s, p, d, w))
            dfs(act + 1, (int(p[0]), int(p[1])), e3, h3, nfr, nwr, macros)
            macros.pop()
            if best["surv"] >= H or capped:
                return
            # wait-for-event option is implicit: relocation is handled by the `reloc` guard above
        # Relocation guard: if every candidate was rejected because of an imminent relocation,
        # allow idling until the relocation step and continue from there.
        if reloc >= 0 and t < reloc < H:
            gap = reloc - t
            e1 = e - (cum_wait[reloc] - cum_wait[t])
            h1 = h - P.BASE_HYDRATION_DRAIN * gap
            if e1 > 0 and h1 > 0:
                macros.append((-1, pos, 0, gap))
                dfs(reloc, pos, e1, h1, fr, wr, macros)
                macros.pop()

    if init is None:
        start = (int(sc.start[0]), int(sc.start[1]))
        dfs(0, start, sc.start_energy, sc.start_hydration,
            tuple(0 for _ in range(nf)), tuple(0 for _ in range(nw)), [])
        t_init = 0
    else:
        start = (int(init["pos"][0]), int(init["pos"][1]))
        t_init = int(init["t"])
        dfs(t_init, start, int(init["energy"]), int(init["hydration"]),
            tuple(int(x) for x in init["food_ready"]), tuple(int(x) for x in init["water_ready"]), [])

    # Expand the best macro list into primitive actions.
    actions: List[int] = []
    pos = start
    t = t_init
    for s, p, d, w in best["macros"]:
        if s >= 0:
            dm = dist_map(p)
            actions.extend(_path_actions(dm, pos))
            actions.extend([P.WAIT] * w)
            actions.append(P.INTERACT)
            pos = (int(p[0]), int(p[1]))
        else:
            actions.extend([P.WAIT] * w)
    tail = best.get("tail", 0)
    actions.extend([P.WAIT] * tail)
    actions = actions[:H - t_init]
    surv = min(best["surv"], H)
    return PlanResult(certified=surv >= H, survival=surv, actions=actions, nodes=nodes, capped=capped)


def verify_plan(sc: Scenario, actions: List[int]) -> int:
    """Replay actions in the reference environment and return the survival count."""
    env = SurvivalEnv()
    env.reset(sc)
    for a in actions:
        if env.done:
            break
        env.step(a)
    return env.survived
