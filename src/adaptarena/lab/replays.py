"""Replay recording: the compact JSON the static site plays, with bit-exact verification data.

A replay holds the scenario (terrain, sites, schedules), the action list, and per-step frames. Frames are
re-derivable from the actions alone; the hash checkpoints let anyone verify that.
"""
from __future__ import annotations

import json
from typing import List, Optional, Sequence

import numpy as np

from ..core import params as P
from ..core.baselines import make_policy
from ..core.env import EnvConfig, SurvivalEnv
from ..core.scenario import Scenario, ScenarioSpec, build_scenario

HASH_EVERY = 25


def scenario_payload(sc: Scenario) -> dict:
    return {
        "spec": sc.spec.to_json(), "sid": sc.sid, "horizon": sc.horizon,
        "terrain": ["".join(str(int(x)) for x in row) for row in sc.terrain],
        "start": [int(sc.start[0]), int(sc.start[1])],
        "food": [[int(a), int(b)] for a, b in sc.food_pos], "water": [[int(a), int(b)] for a, b in sc.water_pos],
        "water2": [[int(a), int(b)] for a, b in sc.water_pos2], "reloc_step": int(sc.reloc_step),
        "food_until": [int(x) for x in sc.food_until], "water_until": [int(x) for x in sc.water_until],
        "hazard": [[int(r), int(c)] for r, c in zip(*np.nonzero(sc.hazard))], "hazard_start": int(sc.hazard_start),
        "cold": "".join(str(int(x)) for x in sc.cold),
    }


def snapshot_env(env: SurvivalEnv) -> dict:
    """Everything needed to restore the reference env mid-episode."""
    return {"t": env.t, "pos": (env.r, env.c), "energy": env.energy, "hydration": env.hydration,
            "food_ready": [int(x) for x in env.food_ready], "water_ready": [int(x) for x in env.water_ready]}


def state_at(sc: Scenario, actions: Sequence[int], k: int, env_config: Optional[EnvConfig] = None) -> SurvivalEnv:
    env = SurvivalEnv(env_config)
    env.reset(sc)
    for a in actions[:k]:
        if env.done:
            break
        env.step(a)
    return env


def record_replay(spec: ScenarioSpec, policy, action_seed: int = 0, env_config: Optional[EnvConfig] = None,
                  with_probs: bool = False, label: str = "") -> dict:
    """Run one episode with the scalar reference env and record everything the site needs.

    policy: script name or a torch model (recurrent models carry their hidden state).
    """
    import torch
    sc = build_scenario(spec)
    env = SurvivalEnv(env_config)
    obs = env.reset(sc)
    torch_mode = hasattr(policy, "sequence")
    if torch_mode:
        policy.eval()
        state, first = policy.initial_state(1), torch.ones(1)
    else:
        pol = make_policy(policy) if isinstance(policy, str) else policy
        pol.reset(action_seed)
    frames: List[list] = []  # per step BEFORE acting: [r, c, energy, hydration, cold, fmask, wmask]
    actions: List[int] = []
    valid: List[int] = []
    probs: List[list] = []
    hashes = []
    log = []

    def sites_mask():
        t = env.t
        f = sum((1 << i) for i in range(sc.nf) if t < sc.food_until[i] and env.food_ready[i] <= t)
        w = sum((1 << i) for i in range(sc.nw) if t < sc.water_until[i] and env.water_ready[i] <= t)
        return f, w

    while not env.done:
        f, w = sites_mask()
        frames.append([env.r, env.c, env.energy, env.hydration, int(sc.cold[env.t]), f, w])
        if env.t % HASH_EVERY == 0:
            hashes.append(env.state_hash())
        if torch_mode:
            flat = torch.from_numpy(np.concatenate([obs["grid"].reshape(-1), obs["stats"]])[None])
            with torch.no_grad():
                logits, _, state = policy.step(flat, state, first)
            a = int(logits.argmax(-1))
            if with_probs:
                probs.append([round(float(p), 3) for p in torch.softmax(logits, -1)[0]])
            first = torch.zeros(1)
        else:
            a = pol.act(obs)
        log.append({"t": env.t, "obs_food_in_view": bool(obs["grid"][P.CH_FOOD].any()),
                    "obs_water_in_view": bool(obs["grid"][P.CH_WATER].any()), "energy": env.energy,
                    "hydration": env.hydration, "action": a})
        obs, r, te, tr, info = env.step(a)
        actions.append(a)
        valid.append(int(info["valid"]))
    f, w = sites_mask()
    frames.append([env.r, env.c, env.energy, env.hydration, int(sc.cold[min(env.t, sc.horizon - 1)]), f, w])
    hashes.append(env.state_hash())
    return {"label": label, "scenario": scenario_payload(sc), "actions": actions, "valid": valid,
            "frames": frames, "survival": env.survived, "alive": bool(env.alive),
            "cause": P.primary_cause(env.flags), "final_hash": env.state_hash(), "hash_every": HASH_EVERY,
            "hashes": hashes, "probs": probs if with_probs else None, "action_seed": action_seed, "_log": log}


def verify_replay(rep: dict) -> bool:
    """Re-run the actions in the reference env and compare the hash checkpoints."""
    spec = ScenarioSpec.from_json(rep["scenario"]["spec"])
    env = SurvivalEnv()
    env.reset(spec)
    hashes = []
    for i, a in enumerate(rep["actions"]):
        if i % rep["hash_every"] == 0:
            hashes.append(env.state_hash())
        env.step(a)
    hashes.append(env.state_hash())
    return hashes == rep["hashes"] and env.state_hash() == rep["final_hash"]


def strip_private(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if not k.startswith("_")}


def replay_from_actions(spec: ScenarioSpec, actions: Sequence[int], label: str = "") -> dict:
    """Replay recorded actions in the reference env (used for human runs; needs only NumPy)."""
    sc = build_scenario(spec)
    env = SurvivalEnv()
    env.reset(sc)
    frames, valid, hashes = [], [], []

    def snap():
        t = env.t
        f = sum((1 << i) for i in range(sc.nf) if t < sc.food_until[i] and env.food_ready[i] <= t)
        w = sum((1 << i) for i in range(sc.nw) if t < sc.water_until[i] and env.water_ready[i] <= t)
        frames.append([env.r, env.c, env.energy, env.hydration, int(sc.cold[min(t, sc.horizon - 1)]), f, w])

    done_actions = []
    for a in actions:
        if env.done:
            break
        if env.t % HASH_EVERY == 0:
            hashes.append(env.state_hash())
        snap()
        _, _, _, _, info = env.step(int(a))
        done_actions.append(int(a))
        valid.append(int(info["valid"]))
    snap()
    hashes.append(env.state_hash())
    return {"label": label, "scenario": scenario_payload(sc), "actions": done_actions, "valid": valid,
            "frames": frames, "survival": env.survived, "alive": bool(env.alive),
            "cause": P.primary_cause(env.flags), "final_hash": env.state_hash(), "hash_every": HASH_EVERY,
            "hashes": hashes, "probs": None, "action_seed": 0}
