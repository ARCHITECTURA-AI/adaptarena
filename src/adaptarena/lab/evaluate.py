"""Evaluation harness. Evaluation never changes model weights.

`evaluate` runs any policy (scripted `Policy`, or a torch model) on a list of scenario specs
and returns one result per episode. Every result stores the evaluation mode, the action seed,
and the final state hash, so a published number can be re-derived bit-exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..core import params as P
from ..core.baselines import Policy, make_policy
from ..core.batch import BatchEnv
from ..core.env import EnvConfig
from ..core.planner import plan
from ..core.scenario import ScenarioSpec, build_scenario


@dataclass
class EpisodeResult:
    sid: str
    survival: int
    horizon: int
    cause: str
    mode: str
    action_seed: int
    final_hash: str
    energy: int
    hydration: int
    actions: Optional[list] = None

    def row(self) -> dict:
        d = dict(self.__dict__)
        d.pop("actions")
        return d


def evaluate(policy, specs: Sequence[ScenarioSpec], action_seeds: Optional[Sequence[int]] = None,
             env_config: Optional[EnvConfig] = None, deterministic: bool = True,
             record_actions: bool = False, max_batch: int = 256) -> List[EpisodeResult]:
    """policy: a policy name ("random", ...), a `Policy` instance factory, or a torch model."""
    results: List[EpisodeResult] = []
    seeds = list(action_seeds) if action_seeds is not None else list(range(len(specs)))
    for lo in range(0, len(specs), max_batch):
        results += _evaluate_chunk(policy, list(specs[lo:lo + max_batch]), seeds[lo:lo + max_batch],
                                   env_config, deterministic, record_actions)
    return results


def _is_torch(policy) -> bool:
    return hasattr(policy, "sequence") and hasattr(policy, "parameters")


def _evaluate_chunk(policy, specs, seeds, env_config, deterministic, record_actions):
    import torch
    n = len(specs)
    env = BatchEnv(n, env_config)
    scs = [build_scenario(s) for s in specs]
    env.reset_all(scs)
    live = np.ones(n, dtype=bool)
    out: List[Optional[EpisodeResult]] = [None] * n
    acts_log = [[] for _ in range(n)] if record_actions else None
    torch_mode = _is_torch(policy)
    if torch_mode:
        was_training = policy.training
        policy.eval()
        state = policy.initial_state(n)
        first = torch.ones(n)
        gen = torch.Generator().manual_seed(int(seeds[0]) * 7919 + len(seeds))  # one stream per evaluation batch
        mode = "deterministic" if deterministic else "sampled"
    else:
        import copy
        pols = [make_policy(policy) if isinstance(policy, str) else (copy.deepcopy(policy) if isinstance(policy, Policy) else policy())
                for _ in range(n)]
        for p, s in zip(pols, seeds):
            p.reset(int(s))
        mode = "scripted"
    hashes_needed = env.n
    while live.any():
        obs = env.observe()
        if torch_mode:
            flat = torch.from_numpy(np.concatenate([obs["grid"].reshape(n, -1), obs["stats"]], axis=1))
            with torch.no_grad():
                logits, _, state = policy.step(flat, state, first)
            if deterministic:
                a = logits.argmax(-1).numpy()
            else:
                probs = torch.softmax(logits, -1)
                a = torch.multinomial(probs, 1, generator=gen)[:, 0].numpy()
        else:
            a = np.zeros(n, dtype=np.int64)
            for i in np.nonzero(live)[0]:
                a[i] = pols[i].act({"grid": obs["grid"][i], "stats": obs["stats"][i]})
        a = np.where(live, a, P.WAIT)
        if record_actions:
            for i in np.nonzero(live)[0]:
                acts_log[i].append(int(a[i]))
        before_alive = live.copy()
        env.step(a)
        done = env.done & before_alive
        if done.any():
            hashes = env.state_hashes()
            for i in np.nonzero(done)[0]:
                out[i] = EpisodeResult(
                    sid=specs[i].sid, survival=int(env.survived[i]), horizon=int(env.horizon[i]),
                    cause=P.primary_cause(int(env.flags[i])), mode=mode, action_seed=int(seeds[i]),
                    final_hash=hashes[i], energy=int(env.energy[i]), hydration=int(env.hyd[i]),
                    actions=acts_log[i] if record_actions else None)
            live &= ~done
        if torch_mode:
            first = torch.from_numpy(env.done.astype(np.float32))
    if torch_mode and was_training:
        policy.train()
    return out


def survival_array(results: Sequence[EpisodeResult]) -> np.ndarray:
    return np.array([r.survival for r in results], dtype=np.float64)


# ------------------------------------------------------------------ ladder references
def planner_survival(specs: Sequence[ScenarioSpec], max_nodes: int = 4000) -> np.ndarray:
    return np.array([plan(build_scenario(s), max_nodes=max_nodes).survival for s in specs], dtype=np.float64)


def random_mean(specs: Sequence[ScenarioSpec], n_seeds: int = 10) -> np.ndarray:
    """Mean random-policy survival per scenario over recorded action seeds 0..n_seeds-1."""
    flat_specs, flat_seeds = [], []
    for s in specs:
        for k in range(n_seeds):
            flat_specs.append(s)
            flat_seeds.append(k)
    res = evaluate("random", flat_specs, flat_seeds)
    return survival_array(res).reshape(len(specs), n_seeds).mean(1)


def ladder_normalise(survival: np.ndarray, rand: np.ndarray, planner: np.ndarray):
    """(policy - random) / (planner - random) per scenario. Returns (scores, valid_mask).

    Scenarios whose denominator is zero are excluded (the caller counts them).
    """
    denom = planner - rand
    valid = denom > 1e-9
    score = np.full_like(survival, np.nan, dtype=np.float64)
    score[valid] = (survival[valid] - rand[valid]) / denom[valid]
    return score, valid
