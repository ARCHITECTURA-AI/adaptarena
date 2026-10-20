"""Scenario specification and deterministic generation.

A `ScenarioSpec` is a small JSON-serialisable record of integers. `build_scenario`
turns it into arrays using three independent seed streams (layout, resource timing,
weather). No floating point enters the generated state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .params import GRID, HORIZON, MAX_SITES, NEVER, STAT_MAX

# Mid-episode interventions are precomputed schedule edits: (name, step, arg)
#   cost    step, pct      movement-cost percent changes from `step` on
#   cold    step, length   a forced cold spell
#   drought step, k        the first k water sites disappear at `step`
#   blight  step, k        the first k food sites disappear at `step`
#   reloc   step, 0        every water site moves to its alternate position
#   hazard  step, n        n hazard tiles become deadly at `step`
INTERVENTIONS = ("cost", "cold", "drought", "blight", "reloc", "hazard")


@dataclass(frozen=True)
class ScenarioSpec:
    layout_id: int = 0
    timing_seed: int = 0
    weather_seed: int = 0
    n_food: int = 4
    n_water: int = 4
    wall_pct: int = 10
    food_delay_min: int = 40
    food_delay_max: int = 60
    water_delay_min: int = 40
    water_delay_max: int = 60
    move_cost_pct: int = 100
    cold_pct: int = 0
    cold_len: int = 40
    horizon: int = HORIZON
    arena: int = GRID  # side of the open square; the rest is wall (tiny maps for tests)
    interventions: tuple = ()

    def to_json(self) -> dict:
        d = asdict(self)
        d["interventions"] = [list(i) for i in self.interventions]
        return d

    @staticmethod
    def from_json(d: dict) -> "ScenarioSpec":
        d = dict(d)
        d["interventions"] = tuple(tuple(int(x) if not isinstance(x, str) else x for x in i)
                                   for i in d.get("interventions", ()))
        return ScenarioSpec(**d)

    def with_(self, **kw) -> "ScenarioSpec":
        d = {**self.__dict__, **kw}
        return ScenarioSpec(**d)

    @property
    def sid(self) -> str:
        blob = json.dumps(self.to_json(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Scenario:
    spec: ScenarioSpec
    terrain: np.ndarray  # (15,15) uint8, 1 = wall
    start: tuple
    food_pos: np.ndarray  # (nf,2) int32
    water_pos: np.ndarray  # (nw,2)
    water_pos2: np.ndarray  # (nw,2) alternate positions used after relocation
    reloc_step: int  # -1 = never
    food_delay: np.ndarray  # (nf,)
    water_delay: np.ndarray  # (nw,)
    food_until: np.ndarray  # (nf,) step at which the site disappears
    water_until: np.ndarray  # (nw,)
    start_energy: int
    start_hydration: int
    cold: np.ndarray  # (horizon,) uint8
    cost_pct: np.ndarray  # (horizon,) int32
    hazard: np.ndarray  # (15,15) uint8
    hazard_start: int
    horizon: int

    @property
    def nf(self) -> int:
        return len(self.food_pos)

    @property
    def nw(self) -> int:
        return len(self.water_pos)

    @property
    def sid(self) -> str:
        return self.spec.sid

    def water_positions(self, t: int) -> np.ndarray:
        if self.reloc_step >= 0 and t >= self.reloc_step:
            return self.water_pos2
        return self.water_pos


def _largest_component(free: np.ndarray) -> np.ndarray:
    n = free.shape[0]
    seen = np.zeros_like(free, dtype=bool)
    best: list = []
    for r in range(n):
        for c in range(n):
            if free[r, c] and not seen[r, c]:
                comp = []
                stack = [(r, c)]
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < n and 0 <= nx < n and free[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if len(comp) > len(best):
                    best = comp
    out = np.zeros_like(free, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def _layout(spec: ScenarioSpec):
    rng = np.random.default_rng([0xA11A, int(spec.layout_id)])
    need = min(19, spec.n_food + spec.n_water + 1)
    for _ in range(50):
        draw = rng.integers(0, 100, size=(GRID, GRID))
        free = draw >= spec.wall_pct
        free[spec.arena:, :] = False
        free[:, spec.arena:] = False
        free = _largest_component(free)
        if free.sum() >= max(need, 12 if spec.arena >= 8 else need):
            break
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(free))]
    order = rng.permutation(len(cells))
    cells = [cells[i] for i in order]
    return (~free).astype(np.uint8), cells


def build_scenario(spec: ScenarioSpec) -> Scenario:
    terrain, cells = _layout(spec)
    nf, nw = spec.n_food, spec.n_water
    if len(cells) >= 1 + 2 * MAX_SITES:
        food_c = cells[1:1 + MAX_SITES][:nf]
        water_c = cells[1 + MAX_SITES:1 + 2 * MAX_SITES][:nw]
        extra = cells[1 + 2 * MAX_SITES:]
    else:  # tiny maps
        food_c = cells[1:1 + nf]
        water_c = cells[1 + nf:1 + nf + nw]
        extra = cells[1 + nf + nw:]
    assert len(food_c) == nf and len(water_c) == nw, "arena too small for the requested sites"
    start = cells[0]
    water2 = (extra[:nw] if len(extra) >= nw else water_c)
    hazard_cells = extra[nw:nw + 10]

    trng = np.random.default_rng([0x71AE, int(spec.timing_seed)])
    fd = trng.integers(spec.food_delay_min, spec.food_delay_max + 1, size=MAX_SITES)[:nf]
    wd = trng.integers(spec.water_delay_min, spec.water_delay_max + 1, size=MAX_SITES)[:nw]
    se = int(trng.integers(6000, 9001))
    sh = int(trng.integers(6000, 9001))

    H = spec.horizon
    wrng = np.random.default_rng([0x3EA7, int(spec.weather_seed)])
    cold = np.zeros(H, dtype=np.uint8)
    cold_draws = wrng.integers(0, 1_000_000, size=(64, 2))
    if spec.cold_pct > 0:
        t = int(cold_draws[0, 0] % 60)
        k = 1
        while t < H and k < 64:
            ln = max(1, spec.cold_len // 2 + int(cold_draws[k, 0] % max(1, spec.cold_len)))
            cold[t:t + ln] = 1
            gap = ln * (100 - spec.cold_pct) // spec.cold_pct
            gap = max(1, gap // 2 + int(cold_draws[k, 1] % max(1, gap)))
            t += ln + gap
            k += 1

    sc = Scenario(
        spec=spec, terrain=terrain, start=start,
        food_pos=np.array(food_c, dtype=np.int32).reshape(nf, 2),
        water_pos=np.array(water_c, dtype=np.int32).reshape(nw, 2),
        water_pos2=np.array(water2, dtype=np.int32).reshape(nw, 2),
        reloc_step=-1,
        food_delay=fd.astype(np.int32), water_delay=wd.astype(np.int32),
        food_until=np.full(nf, NEVER, dtype=np.int32), water_until=np.full(nw, NEVER, dtype=np.int32),
        start_energy=se, start_hydration=sh, cold=cold,
        cost_pct=np.full(H, spec.move_cost_pct, dtype=np.int32),
        hazard=np.zeros((GRID, GRID), dtype=np.uint8), hazard_start=NEVER, horizon=H,
    )
    for name, step, arg in spec.interventions:
        _apply(sc, name, int(step), int(arg), hazard_cells)
    return sc


def _apply(sc: Scenario, name: str, step: int, arg: int, hazard_cells: list) -> None:
    if name == "cost":
        sc.cost_pct[step:] = arg
    elif name == "cold":
        sc.cold[step:step + arg] = 1
    elif name == "drought":
        sc.water_until[:arg] = np.minimum(sc.water_until[:arg], step)
    elif name == "blight":
        sc.food_until[:arg] = np.minimum(sc.food_until[:arg], step)
    elif name == "reloc":
        sc.reloc_step = step
    elif name == "hazard":
        for (r, c) in hazard_cells[:arg]:
            sc.hazard[r, c] = 1
        sc.hazard_start = step
    else:
        raise ValueError(f"unknown intervention {name!r}")


def with_intervention(spec: ScenarioSpec, name: str, step: int, arg: int = 0) -> ScenarioSpec:
    if name not in INTERVENTIONS:
        raise ValueError(name)
    return spec.with_(interventions=tuple(spec.interventions) + ((name, int(step), int(arg)),))
