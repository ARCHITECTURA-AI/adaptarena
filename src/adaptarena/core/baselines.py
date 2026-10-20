"""Scripted baselines for the policy ladder (rungs 1-3). They consume observations only.

Rung 1  RandomPolicy    uniform over the six actions.
Rung 2  ReactiveScript  the v1 rules, frozen before any test run: act on what is in view.
Rung 3  MemoryScript    dead-reckons its own position from successful moves, remembers
                        every resource it has seen relative to its start, and plans over
                        the map it has built. It never sees global coordinates or hidden
                        resources.

The rule constants below are part of the frozen definition; `SCRIPT_VERSION` is hashed into
the analysis plan.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from . import params as P

SCRIPT_VERSION = "scripts-v1"
REACTIVE_THRESHOLD = 0.40  # hydration/energy below this triggers a search (the "rule at 40")
REACTIVE_OPPORTUNISTIC = 0.70
MEMORY_THRESHOLD = 0.55
MEMORY_OPPORTUNISTIC = 0.75
MEMORY_EST_DELAY = 50  # assumed respawn delay for a site last seen depleted

C = P.VIEW_R  # centre index of the 9x9 view
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Policy:
    """Per-environment stateful policy over observations."""
    name = "policy"

    def reset(self, seed: int = 0) -> None:
        pass

    def act(self, obs: dict) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    name = "random"

    def reset(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng([0xAC71, int(seed)])

    def act(self, obs: dict) -> int:
        return int(self.rng.integers(0, P.N_ACTIONS))


def _view_bfs(floor: np.ndarray, targets: np.ndarray):
    """First action of a shortest path from the view centre to the nearest target cell."""
    n = floor.shape[0]
    if targets[C, C]:
        return P.INTERACT
    prev = {(C, C): None}
    dq = deque([(C, C)])
    while dq:
        y, x = dq.popleft()
        for a, (dy, dx) in enumerate(_DIRS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and (ny, nx) not in prev and floor[ny, nx]:
                prev[(ny, nx)] = ((y, x), a)
                if targets[ny, nx]:
                    cur = (ny, nx)
                    while prev[cur][0] != (C, C):
                        cur = prev[cur][0]
                    return prev[cur][1]
                dq.append((ny, nx))
    return None


class ReactiveScript(Policy):
    name = "reactive_script"

    def reset(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng([0xAC72, int(seed)])
        self.heading = int(self.rng.integers(0, 4))

    def act(self, obs: dict) -> int:
        g, s = obs["grid"], obs["stats"]
        energy, hyd = float(s[P.S_ENERGY]), float(s[P.S_HYDRATION])
        floor = g[P.CH_FLOOR] > 0
        on_food, on_water = g[P.CH_FOOD, C, C] > 0, g[P.CH_WATER, C, C] > 0
        if on_water and hyd < REACTIVE_OPPORTUNISTIC:
            return P.INTERACT
        if on_food and energy < REACTIVE_OPPORTUNISTIC:
            return P.INTERACT
        need_w, need_f = hyd < REACTIVE_THRESHOLD, energy < REACTIVE_THRESHOLD
        if not (need_w or need_f):
            return P.WAIT
        order = []
        if need_w and (not need_f or hyd <= energy):
            order = [P.CH_WATER, P.CH_FOOD] if need_f else [P.CH_WATER]
        elif need_f:
            order = [P.CH_FOOD, P.CH_WATER] if need_w else [P.CH_FOOD]
        for ch in order:
            tgt = g[ch] > 0
            if tgt.any():
                a = _view_bfs(floor, tgt)
                if a is not None:
                    return a
        # nothing useful in view: keep walking, turn when blocked
        prev_a = int(np.argmax(s[P.S_PREV_ACTION])) if s[P.S_PREV_ACTION].any() else None
        if prev_a is not None and prev_a < 4 and s[P.S_PREV_VALID] < 0.5:
            self.heading = int(self.rng.integers(0, 4))
        return self.heading


class MemoryScript(Policy):
    name = "memory_script"

    def reset(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng([0xAC73, int(seed)])
        self.x = self.y = 0  # dead-reckoned offset from the start cell
        self.t = 0
        self.floor: Dict[Tuple[int, int], bool] = {}  # known cells: True floor, False wall
        self.sites: Dict[Tuple[int, int], dict] = {}  # cell -> {kind, ready, seen_dep}
        self.heading = int(self.rng.integers(0, 4))

    # -- map maintenance ---------------------------------------------------------
    def _observe(self, g: np.ndarray) -> None:
        for dy in range(-C, C + 1):
            for dx in range(-C, C + 1):
                if g[P.CH_VISIBLE, C + dy, C + dx] == 0:
                    continue
                cell = (self.y + dy, self.x + dx)
                if g[P.CH_WALL, C + dy, C + dx] > 0:
                    self.floor[cell] = False
                    continue
                self.floor[cell] = True
                kind, state = None, None
                if g[P.CH_FOOD, C + dy, C + dx] > 0:
                    kind, state = "food", True
                elif g[P.CH_FOOD_DEPLETED, C + dy, C + dx] > 0:
                    kind, state = "food", False
                elif g[P.CH_WATER, C + dy, C + dx] > 0:
                    kind, state = "water", True
                elif g[P.CH_WATER_DEPLETED, C + dy, C + dx] > 0:
                    kind, state = "water", False
                if kind is not None:
                    rec = self.sites.get(cell)
                    if rec is None or rec["kind"] != kind:
                        rec = {"kind": kind, "ready": state, "seen_dep": self.t if not state else None}
                    else:
                        if state is False and rec["ready"]:
                            rec["seen_dep"] = self.t
                        rec["ready"] = state
                    self.sites[cell] = rec
                elif cell in self.sites:  # the site vanished (gone, relocated)
                    del self.sites[cell]
                if g[P.CH_HAZARD, C + dy, C + dx] > 0:
                    self.floor[cell] = False

    def _bfs(self, goal_fn):
        start = (self.y, self.x)
        prev = {start: None}
        dq = deque([(start, 0)])
        while dq:
            (cy, cx), d = dq.popleft()
            if d > 0 and goal_fn((cy, cx)):
                cur = (cy, cx)
                while prev[cur][0] != start:
                    cur = prev[cur][0]
                return prev[cur][1], d, (cy, cx)
            for a, (dy, dx) in enumerate(_DIRS):
                nb = (cy + dy, cx + dx)
                if nb not in prev and self.floor.get(nb, False):
                    prev[nb] = ((cy, cx), a)
                    dq.append((nb, d + 1))
        return None

    def _dist_to_all(self):
        start = (self.y, self.x)
        dist = {start: 0}
        first = {start: None}
        dq = deque([start])
        while dq:
            cur = dq.popleft()
            for a, (dy, dx) in enumerate(_DIRS):
                nb = (cur[0] + dy, cur[1] + dx)
                if nb not in dist and self.floor.get(nb, False):
                    dist[nb] = dist[cur] + 1
                    first[nb] = a if first[cur] is None else first[cur]
                    dq.append(nb)
        return dist, first

    # -- decision -----------------------------------------------------------------
    def act(self, obs: dict) -> int:
        g, s = obs["grid"], obs["stats"]
        if s[P.S_PREV_ACTION].any():
            pa = int(np.argmax(s[P.S_PREV_ACTION]))
            if pa < 4 and s[P.S_PREV_VALID] > 0.5:
                self.y += _DIRS[pa][0]
                self.x += _DIRS[pa][1]
        self._observe(g)
        a = self._decide(g, s)
        self.t += 1
        return a

    def _decide(self, g, s) -> int:
        energy, hyd = float(s[P.S_ENERGY]), float(s[P.S_HYDRATION])
        here = (self.y, self.x)
        rec = self.sites.get(here)
        if rec and rec["ready"]:
            frac = hyd if rec["kind"] == "water" else energy
            if frac < MEMORY_OPPORTUNISTIC:
                rec["ready"] = False
                rec["seen_dep"] = self.t
                return P.INTERACT
        need = None
        if min(energy, hyd) < MEMORY_THRESHOLD:
            need = "water" if hyd <= energy else "food"
        dist, first = self._dist_to_all()
        if need is not None:
            best, best_cost = None, 1e9
            for cell, rc in self.sites.items():
                if rc["kind"] != need or cell not in dist:
                    continue
                d = dist[cell]
                wait = 0 if rc["ready"] else max(0, (rc["seen_dep"] or self.t) + MEMORY_EST_DELAY - (self.t + d))
                cost = d + wait
                if cost < best_cost:
                    best, best_cost = cell, cost
            if best is not None:
                if best == here:
                    return P.WAIT  # waiting on a depleted site for it to respawn
                return first[best]
            # no known site of this kind: go and look
            return self._explore(dist, first)
        # nothing urgent: discover the map while resources are still scarce in memory
        kinds = [r["kind"] for r in self.sites.values()]
        if kinds.count("water") < 2 or kinds.count("food") < 2:
            return self._explore(dist, first, quiet=True)
        return P.WAIT

    def _explore(self, dist, first, quiet: bool = False) -> int:
        best, bd = None, 1e9
        for cell, d in dist.items():
            if d == 0:
                continue
            for dy, dx in _DIRS:
                nb = (cell[0] + dy, cell[1] + dx)
                if nb not in self.floor:
                    if d < bd:
                        best, bd = cell, d
                    break
        if best is not None:
            return first[best]
        if quiet:
            return P.WAIT
        return int(self.rng.integers(0, 4))


def make_policy(name: str) -> Policy:
    return {"random": RandomPolicy, "reactive_script": ReactiveScript, "memory_script": MemoryScript}[name]()
