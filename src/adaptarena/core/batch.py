"""Vectorised batch environment. Same dynamics as `SurvivalEnv`, N episodes at once.

Equivalence with the reference implementation is enforced by tests that compare the
SHA-256 state hash of every environment at every step.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

from . import params as P
from .env import EnvConfig, potential
from .scenario import Scenario, ScenarioSpec, build_scenario

_DR = np.array(P.DR, dtype=np.int64)
_DC = np.array(P.DC, dtype=np.int64)
PAD = P.VIEW_R


class BatchEnv:
    def __init__(self, n: int, config: Optional[EnvConfig] = None, max_horizon: int = P.HORIZON):
        self.n = n
        self.config = config or EnvConfig()
        self.H = max_horizon
        S = P.MAX_SITES
        i64, i32 = np.int64, np.int32
        self.wallpad = np.ones((n, P.GRID + 2 * PAD, P.GRID + 2 * PAD), dtype=bool)
        self.hazard = np.zeros((n, P.GRID, P.GRID), dtype=bool)
        self.hazard_start = np.full(n, P.NEVER, dtype=i64)
        self.horizon = np.full(n, P.HORIZON, dtype=i64)
        self.nf = np.zeros(n, dtype=i64)
        self.nw = np.zeros(n, dtype=i64)
        self.fpos = np.zeros((n, S, 2), dtype=i64)
        self.wpos = np.zeros((n, S, 2), dtype=i64)
        self.wpos2 = np.zeros((n, S, 2), dtype=i64)
        self.reloc = np.full(n, -1, dtype=i64)
        self.fdelay = np.zeros((n, S), dtype=i64)
        self.wdelay = np.zeros((n, S), dtype=i64)
        self.funtil = np.zeros((n, S), dtype=i64)  # 0 for absent sites => never active
        self.wuntil = np.zeros((n, S), dtype=i64)
        self.cold = np.zeros((n, max_horizon), dtype=i64)
        self.cost = np.full((n, max_horizon), 100, dtype=i64)
        self.t = np.zeros(n, dtype=i64)
        self.r = np.zeros(n, dtype=i64)
        self.c = np.zeros(n, dtype=i64)
        self.energy = np.zeros(n, dtype=i64)
        self.hyd = np.zeros(n, dtype=i64)
        self.prev_action = np.zeros(n, dtype=i64)
        self.has_prev = np.zeros(n, dtype=i64)
        self.prev_valid = np.zeros(n, dtype=i64)
        self.last_drain = np.zeros(n, dtype=i64)
        self.alive = np.ones(n, dtype=bool)
        self.flags = np.zeros(n, dtype=i64)
        self.fready = np.zeros((n, S), dtype=i64)
        self.wready = np.zeros((n, S), dtype=i64)
        self.survived = np.zeros(n, dtype=i64)
        self.prev_phi = np.zeros(n, dtype=np.float64)
        self.scenarios: list = [None] * n
        self._ar = np.arange(n)
        self._site_ar = np.arange(S)
        self._mask = self.config.vision_mask()
        self._win = np.arange(P.VIEW)

    # ------------------------------------------------------------------ reset
    def reset_env(self, i: int, scenario) -> None:
        if isinstance(scenario, ScenarioSpec):
            scenario = build_scenario(scenario)
        sc: Scenario = scenario
        self.scenarios[i] = sc
        self.wallpad[i] = True
        self.wallpad[i, PAD:PAD + P.GRID, PAD:PAD + P.GRID] = sc.terrain.astype(bool)
        self.hazard[i] = sc.hazard.astype(bool)
        self.hazard_start[i] = sc.hazard_start
        self.horizon[i] = sc.horizon
        self.nf[i], self.nw[i] = sc.nf, sc.nw
        for arr in (self.fpos, self.wpos, self.wpos2):
            arr[i] = 0
        self.fpos[i, :sc.nf] = sc.food_pos
        self.wpos[i, :sc.nw] = sc.water_pos
        self.wpos2[i, :sc.nw] = sc.water_pos2
        self.reloc[i] = sc.reloc_step
        for arr in (self.fdelay, self.wdelay, self.funtil, self.wuntil, self.fready, self.wready):
            arr[i] = 0
        self.fdelay[i, :sc.nf] = sc.food_delay
        self.wdelay[i, :sc.nw] = sc.water_delay
        self.funtil[i, :sc.nf] = sc.food_until
        self.wuntil[i, :sc.nw] = sc.water_until
        self.cold[i] = 0
        self.cold[i, :sc.horizon] = sc.cold
        self.cost[i] = 100
        self.cost[i, :sc.horizon] = sc.cost_pct
        self.t[i] = 0
        self.r[i], self.c[i] = sc.start
        self.energy[i], self.hyd[i] = sc.start_energy, sc.start_hydration
        self.prev_action[i] = self.has_prev[i] = self.prev_valid[i] = self.last_drain[i] = 0
        self.alive[i] = True
        self.flags[i] = 0
        self.survived[i] = 0
        self.prev_phi[i] = potential(sc.start_energy, sc.start_hydration, self.config.potential_scale)

    def reset_all(self, scenarios: Sequence) -> None:
        assert len(scenarios) == self.n
        for i, s in enumerate(scenarios):
            self.reset_env(i, s)

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray):
        """Advance every environment one step. Returns reward, terminated, truncated, info."""
        a = np.asarray(actions, dtype=np.int64)
        n, ar, t = self.n, self._ar, self.t
        is_move = a < 4
        tr = self.r + _DR[a]
        tc = self.c + _DC[a]
        blocked = self.wallpad[ar, tr + PAD, tc + PAD]
        move_ok = is_move & ~blocked
        nr = np.where(move_ok, tr, self.r)
        nc = np.where(move_ok, tc, self.c)

        site_ok = self._site_ar[None, :]
        tcol = t[:, None]
        # food
        f_match = ((self.fpos[:, :, 0] == self.r[:, None]) & (self.fpos[:, :, 1] == self.c[:, None])
                   & (site_ok < self.nf[:, None]) & (tcol < self.funtil) & (self.fready <= tcol))
        reloc_on = (self.reloc >= 0) & (t >= self.reloc)
        wp = np.where(reloc_on[:, None, None], self.wpos2, self.wpos)
        w_match = ((wp[:, :, 0] == self.r[:, None]) & (wp[:, :, 1] == self.c[:, None])
                   & (site_ok < self.nw[:, None]) & (tcol < self.wuntil) & (self.wready <= tcol))
        is_int = a == P.INTERACT
        f_hit = is_int & f_match.any(1)
        w_hit = is_int & w_match.any(1)
        valid = move_ok | f_hit | w_hit | (a == P.WAIT)

        fi = f_match.argmax(1)
        wi = w_match.argmax(1)
        self.energy = np.where(f_hit, np.minimum(P.STAT_MAX, self.energy + P.FOOD_RESTORE), self.energy)
        self.hyd = np.where(w_hit, np.minimum(P.STAT_MAX, self.hyd + P.WATER_RESTORE), self.hyd)
        fidx = ar[f_hit]
        self.fready[fidx, fi[f_hit]] = t[f_hit] + 1 + self.fdelay[fidx, fi[f_hit]]
        widx = ar[w_hit]
        self.wready[widx, wi[w_hit]] = t[w_hit] + 1 + self.wdelay[widx, wi[w_hit]]
        self.r, self.c = nr, nc

        tt = np.minimum(t, self.H - 1)
        cost = self.cost[ar, tt]
        drain_e = (P.BASE_ENERGY_DRAIN + np.where(is_move, P.MOVE_EXTRA_ENERGY * cost // 100, 0)
                   + P.COLD_EXTRA_ENERGY * self.cold[ar, tt])
        self.energy = self.energy - drain_e
        self.hyd = self.hyd - P.BASE_HYDRATION_DRAIN

        flags = (np.where((t >= self.hazard_start) & self.hazard[ar, self.r, self.c], P.CAUSE_HAZARD, 0)
                 | np.where(self.hyd <= 0, P.CAUSE_DEHYDRATION, 0)
                 | np.where(self.energy <= 0, P.CAUSE_STARVATION, 0))
        self.flags = flags.astype(np.int64)
        alive = flags == 0
        self.alive = alive
        self.prev_action = a
        self.has_prev = np.ones(n, dtype=np.int64)
        self.prev_valid = valid.astype(np.int64)
        self.last_drain = drain_e
        self.t = t + 1
        self.survived = self.survived + alive

        consumed = f_hit | w_hit
        mode = self.config.reward
        base = alive.astype(np.float32)
        if mode == "naive":
            reward = base + self.config.naive_bonus * consumed
        elif mode == "potential":
            phi = np.where(alive, potential(self.energy, self.hyd, self.config.potential_scale), 0.0)
            reward = base + self.config.gamma * phi - self.prev_phi
            self.prev_phi = phi
        else:
            reward = base
        terminated = ~alive
        truncated = alive & (self.t >= self.horizon)
        return reward.astype(np.float32), terminated, truncated

    @property
    def done(self) -> np.ndarray:
        return (~self.alive) | (self.t >= self.horizon)

    # ------------------------------------------------------------------ observation
    def observe(self) -> dict:
        n, ar = self.n, self._ar
        grid = np.zeros((n, P.N_CHANNELS, P.VIEW, P.VIEW), dtype=np.float32)
        rows = (self.r[:, None] + self._win[None, :])  # padded coordinates of the window
        cols = (self.c[:, None] + self._win[None, :])
        wall = self.wallpad[ar[:, None, None], rows[:, :, None], cols[:, None, :]]
        grid[:, P.CH_WALL] = wall
        grid[:, P.CH_FLOOR] = ~wall
        t = self.t
        tcol = t[:, None]
        site_ok = self._site_ar[None, :]
        reloc_on = (self.reloc >= 0) & (t >= self.reloc)
        wp = np.where(reloc_on[:, None, None], self.wpos2, self.wpos)

        def put(pos, valid, ready, ch_ready, ch_dep):
            dy = pos[:, :, 0] - self.r[:, None]
            dx = pos[:, :, 1] - self.c[:, None]
            vis = valid & (np.abs(dy) <= P.VIEW_R) & (np.abs(dx) <= P.VIEW_R)
            nn, ss = np.nonzero(vis)
            ch = np.where(ready[nn, ss], ch_ready, ch_dep)
            grid[nn, ch, dy[nn, ss] + P.VIEW_R, dx[nn, ss] + P.VIEW_R] = 1.0

        put(self.fpos, (site_ok < self.nf[:, None]) & (tcol < self.funtil), self.fready <= tcol,
            P.CH_FOOD, P.CH_FOOD_DEPLETED)
        put(wp, (site_ok < self.nw[:, None]) & (tcol < self.wuntil), self.wready <= tcol,
            P.CH_WATER, P.CH_WATER_DEPLETED)

        haz_on = t >= self.hazard_start
        if haz_on.any():
            hp = np.zeros((n, P.GRID + 2 * PAD, P.GRID + 2 * PAD), dtype=bool)
            hp[:, PAD:PAD + P.GRID, PAD:PAD + P.GRID] = self.hazard
            hz = hp[ar[:, None, None], rows[:, :, None], cols[:, None, :]]
            grid[:, P.CH_HAZARD] = hz & haz_on[:, None, None]
        grid *= self._mask
        grid[:, P.CH_VISIBLE] = self._mask

        stats = np.zeros((n, P.STATS_DIM), dtype=np.float32)
        stats[:, P.S_ENERGY] = self.energy / P.STAT_MAX
        stats[:, P.S_HYDRATION] = self.hyd / P.STAT_MAX
        stats[:, P.S_COLD] = self.cold[ar, np.minimum(t, self.H - 1)]
        if self.config.prev_action:
            has = self.has_prev.astype(bool)
            stats[ar[has], 3 + self.prev_action[has]] = 1.0
            stats[:, P.S_PREV_VALID] = self.prev_valid * self.has_prev
        if self.config.cost_stat:
            stats[:, P.S_LAST_DRAIN] = np.minimum(1.0, self.last_drain / P.DRAIN_NORM)
        return {"grid": grid, "stats": stats}

    def flat_obs(self) -> np.ndarray:
        o = self.observe()
        return np.concatenate([o["grid"].reshape(self.n, -1), o["stats"]], axis=1)

    # ------------------------------------------------------------------ hashing
    def state_matrix(self) -> np.ndarray:
        head = np.stack([self.t, self.r, self.c, self.energy, self.hyd, self.prev_action,
                         self.prev_valid, self.last_drain, self.alive.astype(np.int64), self.flags], axis=1)
        out = np.zeros((self.n, 10 + 2 * P.MAX_SITES), dtype="<i4")
        out[:, :10] = head
        out[:, 10:10 + P.MAX_SITES] = self.fready
        out[:, 10 + P.MAX_SITES:] = self.wready
        return out

    def state_hashes(self) -> list:
        m = self.state_matrix()
        return [hashlib.sha256(m[i].tobytes()).hexdigest() for i in range(self.n)]
