"""Reference (scalar, readable) implementation of AdaptArena-Survival-v1.

Step order, for action `a` taken at step index `t` (state is after `t` steps):

 1. Interpret the action. A move into a wall or off the map is invalid. Interact is
    valid only on a ready, still-existing site at the agent's cell. Wait is always valid.
 2. Apply it: update the position; on a valid interact restore the stat (capped at
    10000) and start that site's respawn timer (it is ready again at t + 1 + delay).
 3. Drain. energy -= base + (movement surcharge if the action was a move attempt,
    valid or not) + (cold surcharge if cold[t]); hydration -= base.
    Invalid interactions therefore cost only the base drain.
 4. Death flags: an active hazard tile at the agent's cell, hydration <= 0, energy <= 0.
    All that apply are stored; the primary reason is hazard > dehydration > starvation.
 5. Reward is 1 if the agent is alive after the step. t advances. The episode is
    truncated (never terminated) when it is still alive at the horizon.

Every quantity in the state is an integer, so the state hash is identical on every
machine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import params as P
from .scenario import Scenario, ScenarioSpec, build_scenario


@dataclass(frozen=True)
class EnvConfig:
    """Everything about the interface that an ablation may change."""
    vision: int = 9  # crop side: 9, 5 or 3
    prev_action: bool = True
    cost_stat: bool = True
    reward: str = "sparse"  # sparse | naive | potential
    gamma: float = 0.99  # used by potential-based shaping
    naive_bonus: float = 0.1
    potential_scale: float = 5.0

    @property
    def obs_version(self) -> str:
        parts = [f"v{self.vision}"]
        if not self.prev_action:
            parts.append("noprev")
        if not self.cost_stat:
            parts.append("nocost")
        return "obs-" + "-".join(parts)

    def vision_mask(self) -> np.ndarray:
        r = self.vision // 2
        m = np.zeros((P.VIEW, P.VIEW), dtype=np.float32)
        lo, hi = P.VIEW_R - r, P.VIEW_R + r + 1
        m[lo:hi, lo:hi] = 1.0
        return m


def potential(energy, hydration, scale: float):
    return scale * (energy + hydration) / (2.0 * P.STAT_MAX)


def state_vector(t, r, c, energy, hydration, prev_action, prev_valid, last_drain, alive, flags,
                 food_ready, water_ready) -> np.ndarray:
    head = np.array([t, r, c, energy, hydration, prev_action, prev_valid, last_drain, int(alive), flags],
                    dtype="<i4")
    fr = np.zeros(P.MAX_SITES, dtype="<i4")
    wr = np.zeros(P.MAX_SITES, dtype="<i4")
    fr[:len(food_ready)] = food_ready
    wr[:len(water_ready)] = water_ready
    return np.concatenate([head, fr, wr])


def hash_state_vector(v: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, dtype="<i4").tobytes()).hexdigest()


class SurvivalEnv:
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.scenario: Optional[Scenario] = None
        self._mask = self.config.vision_mask()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, scenario) -> dict:
        if isinstance(scenario, ScenarioSpec):
            scenario = build_scenario(scenario)
        self.scenario = sc = scenario
        self.t = 0
        self.r, self.c = sc.start
        self.energy = sc.start_energy
        self.hydration = sc.start_hydration
        self.prev_action = 0  # all-zero one-hot is encoded by prev_flag below
        self.has_prev = 0
        self.prev_valid = 0
        self.last_drain = 0
        self.alive = True
        self.flags = 0
        self.food_ready = np.zeros(sc.nf, dtype=np.int32)
        self.water_ready = np.zeros(sc.nw, dtype=np.int32)
        self.survived = 0
        self._prev_phi = potential(self.energy, self.hydration, self.config.potential_scale)
        return self.observation()

    @property
    def horizon(self) -> int:
        return self.scenario.horizon

    @property
    def done(self) -> bool:
        return (not self.alive) or self.t >= self.horizon

    # ------------------------------------------------------------------ observation
    def _site_cells(self):
        sc = self.scenario
        t = self.t
        for i in range(sc.nf):
            if t < sc.food_until[i]:
                yield P.CH_FOOD, P.CH_FOOD_DEPLETED, sc.food_pos[i], self.food_ready[i] <= t
        wp = sc.water_positions(t)
        for i in range(sc.nw):
            if t < sc.water_until[i]:
                yield P.CH_WATER, P.CH_WATER_DEPLETED, wp[i], self.water_ready[i] <= t

    def observation(self) -> dict:
        sc = self.scenario
        grid = np.zeros((P.N_CHANNELS, P.VIEW, P.VIEW), dtype=np.float32)
        for dy in range(-P.VIEW_R, P.VIEW_R + 1):
            for dx in range(-P.VIEW_R, P.VIEW_R + 1):
                y, x = self.r + dy, self.c + dx
                if 0 <= y < P.GRID and 0 <= x < P.GRID and not sc.terrain[y, x]:
                    grid[P.CH_FLOOR, dy + P.VIEW_R, dx + P.VIEW_R] = 1
                else:
                    grid[P.CH_WALL, dy + P.VIEW_R, dx + P.VIEW_R] = 1
        for ch_ready, ch_dep, pos, ready in self._site_cells():
            dy, dx = pos[0] - self.r, pos[1] - self.c
            if abs(dy) <= P.VIEW_R and abs(dx) <= P.VIEW_R:
                grid[ch_ready if ready else ch_dep, dy + P.VIEW_R, dx + P.VIEW_R] = 1
        if self.t >= sc.hazard_start:
            for dy in range(-P.VIEW_R, P.VIEW_R + 1):
                for dx in range(-P.VIEW_R, P.VIEW_R + 1):
                    y, x = self.r + dy, self.c + dx
                    if 0 <= y < P.GRID and 0 <= x < P.GRID and sc.hazard[y, x]:
                        grid[P.CH_HAZARD, dy + P.VIEW_R, dx + P.VIEW_R] = 1
        grid *= self._mask
        grid[P.CH_VISIBLE] = self._mask

        stats = np.zeros(P.STATS_DIM, dtype=np.float32)
        stats[P.S_ENERGY] = self.energy / P.STAT_MAX
        stats[P.S_HYDRATION] = self.hydration / P.STAT_MAX
        stats[P.S_COLD] = float(sc.cold[min(self.t, sc.horizon - 1)])
        if self.config.prev_action and self.has_prev:
            stats[3 + self.prev_action] = 1.0
            stats[P.S_PREV_VALID] = float(self.prev_valid)
        if self.config.cost_stat:
            stats[P.S_LAST_DRAIN] = min(1.0, self.last_drain / P.DRAIN_NORM)
        return {"grid": grid, "stats": stats}

    # ------------------------------------------------------------------ dynamics
    def step(self, action: int):
        assert self.alive and self.t < self.horizon, "step() on a finished episode"
        sc, t = self.scenario, self.t
        a = int(action)
        valid = True
        moved = False
        nr, nc = self.r, self.c
        food_hit = water_hit = -1

        if a < 4:
            moved = True
            tr, tc = self.r + P.DR[a], self.c + P.DC[a]
            if 0 <= tr < P.GRID and 0 <= tc < P.GRID and not sc.terrain[tr, tc]:
                nr, nc = tr, tc
            else:
                valid = False
        elif a == P.INTERACT:
            for i in range(sc.nf):
                if (sc.food_pos[i][0] == self.r and sc.food_pos[i][1] == self.c
                        and t < sc.food_until[i] and self.food_ready[i] <= t):
                    food_hit = i
            wp = sc.water_positions(t)
            for i in range(sc.nw):
                if (wp[i][0] == self.r and wp[i][1] == self.c
                        and t < sc.water_until[i] and self.water_ready[i] <= t):
                    water_hit = i
            if food_hit < 0 and water_hit < 0:
                valid = False

        self.r, self.c = nr, nc
        consumed = False
        if food_hit >= 0:
            self.energy = min(P.STAT_MAX, self.energy + P.FOOD_RESTORE)
            self.food_ready[food_hit] = t + 1 + sc.food_delay[food_hit]
            consumed = True
        if water_hit >= 0:
            self.hydration = min(P.STAT_MAX, self.hydration + P.WATER_RESTORE)
            self.water_ready[water_hit] = t + 1 + sc.water_delay[water_hit]
            consumed = True

        drain_e = P.BASE_ENERGY_DRAIN
        if moved:
            drain_e += P.MOVE_EXTRA_ENERGY * int(sc.cost_pct[t]) // 100
        if sc.cold[t]:
            drain_e += P.COLD_EXTRA_ENERGY
        self.energy -= drain_e
        self.hydration -= P.BASE_HYDRATION_DRAIN

        flags = 0
        if t >= sc.hazard_start and sc.hazard[self.r, self.c]:
            flags |= P.CAUSE_HAZARD
        if self.hydration <= 0:
            flags |= P.CAUSE_DEHYDRATION
        if self.energy <= 0:
            flags |= P.CAUSE_STARVATION
        self.flags = flags
        self.alive = flags == 0

        self.prev_action, self.has_prev, self.prev_valid, self.last_drain = a, 1, int(valid), drain_e
        self.t += 1
        if self.alive:
            self.survived += 1

        reward = self._reward(consumed)
        terminated = not self.alive
        truncated = self.alive and self.t >= self.horizon
        info = {"valid": valid, "drain": drain_e, "flags": flags, "survived": self.survived,
                "primary_cause": P.primary_cause(flags)}
        self._prev_phi = self._phi()
        return self.observation(), reward, terminated, truncated, info

    def _phi(self) -> float:
        if not self.alive:
            return 0.0
        return potential(self.energy, self.hydration, self.config.potential_scale)

    def _reward(self, consumed: bool) -> float:
        base = 1.0 if self.alive else 0.0
        mode = self.config.reward
        if mode == "naive":
            return base + (self.config.naive_bonus if consumed else 0.0)
        if mode == "potential":
            return base + self.config.gamma * self._phi() - self._prev_phi
        return base

    # ------------------------------------------------------------------ hashing
    def state_vector(self) -> np.ndarray:
        return state_vector(self.t, self.r, self.c, self.energy, self.hydration, self.prev_action,
                            self.prev_valid, self.last_drain, self.alive, self.flags,
                            self.food_ready, self.water_ready)

    def state_hash(self) -> str:
        return hash_state_vector(self.state_vector())

    def scenario_id(self) -> str:
        return self.scenario.sid


def flatten_obs(obs: dict) -> np.ndarray:
    return np.concatenate([obs["grid"].reshape(-1), obs["stats"]]).astype(np.float32)
