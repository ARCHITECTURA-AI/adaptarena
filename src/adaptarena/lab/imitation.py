"""Phase 4.1: the imitation ceiling (diagnostic).

Roll out the memory script on the arm B training distribution, log (observation, script action), and fit a
feedforward net and an LSTM by cross-entropy. No reinforcement learning is involved, so what a clone cannot
reproduce is a limit of the architecture and the data, not of PPO. Everything is scored on the validation suites;
the fresh test suite is never opened here.

The script builds its map from the observation stream (previous action and its valid flag are in the stats), so
DAgger-lite is possible: `ShadowLabeler` runs the script's logic over the *learner's* observations. The one
action-coupled step, marking a site depleted when the script itself chooses INTERACT, is rolled back, because the
learner may not have interacted.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core import params as P
from ..core.baselines import MemoryScript
from ..core.batch import BatchEnv
from ..core.distributions import CONDITIONS, sample_train
from ..core.env import EnvConfig
from ..core.scenario import ScenarioSpec, build_scenario
from . import stats as S
from .experiments import load_tier, tier_dir, write_json
from .suites import Suite, per_seed_score

GRID_DIM = P.N_CHANNELS * P.VIEW * P.VIEW
STEP_BUDGET = 2048  # samples per optimiser step, for every architecture and window


class ShadowLabeler(MemoryScript):
    """The memory script as a labeller: returns the action it would take, without acting on its own bookkeeping."""
    name = "shadow_labeler"

    def label(self, obs: dict) -> int:
        g, s = obs["grid"], obs["stats"]
        if s[P.S_PREV_ACTION].any():
            pa = int(np.argmax(s[P.S_PREV_ACTION]))
            if pa < 4 and s[P.S_PREV_VALID] > 0.5:
                dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[pa]
                self.y += dy
                self.x += dx
        self._observe(g)
        here = self.sites.get((self.y, self.x))
        saved = None if here is None else (here["ready"], here["seen_dep"])
        a = self._decide(g, s)
        if here is not None:  # undo "I interacted, so it is now empty": the learner may have done something else
            here["ready"], here["seen_dep"] = saved
        self.t += 1
        return a


class Episodes:
    """Variable-length episodes: grid (float16, flat 648), stats (float32, 11), action (int8)."""

    def __init__(self):
        self.grid: List[np.ndarray] = []
        self.stats: List[np.ndarray] = []
        self.act: List[np.ndarray] = []

    def add(self, grid, stats, act):
        self.grid.append(np.asarray(grid, np.float16))
        self.stats.append(np.asarray(stats, np.float32))
        self.act.append(np.asarray(act, np.int8))

    def extend(self, other: "Episodes"):
        self.grid += other.grid
        self.stats += other.stats
        self.act += other.act

    def __len__(self):
        return len(self.act)

    @property
    def n_samples(self) -> int:
        return int(sum(len(a) for a in self.act))

    def flat(self, i: int) -> np.ndarray:
        return np.concatenate([self.grid[i].astype(np.float32), self.stats[i]], axis=1)


# --------------------------------------------------------------------------------------- rollouts
def _roll(specs: Sequence[ScenarioSpec], seeds: Sequence[int], learner=None, env_config=None) -> Episodes:
    """Run the script (learner=None) or `learner` (a torch policy, argmax) with script labels on every visited state."""
    import torch
    torch.set_num_threads(1)
    n = len(specs)
    env = BatchEnv(n, env_config or EnvConfig())
    env.reset_all([build_scenario(s) for s in specs])
    labelers = [ShadowLabeler() for _ in range(n)]
    for lab, s in zip(labelers, seeds):
        lab.reset(int(s))
    scripts = [MemoryScript() for _ in range(n)]
    for p, s in zip(scripts, seeds):
        p.reset(int(s))
    g_log: List[list] = [[] for _ in range(n)]
    s_log: List[list] = [[] for _ in range(n)]
    a_log: List[list] = [[] for _ in range(n)]
    live = np.ones(n, bool)
    state = learner.initial_state(n) if learner is not None else None
    first = torch.ones(n)
    while live.any():
        o = env.observe()
        flat = np.concatenate([o["grid"].reshape(n, -1), o["stats"]], axis=1)
        if learner is not None:
            with torch.no_grad():
                logits, _, state = learner.step(torch.from_numpy(flat), state, first)
            act = logits.argmax(-1).numpy()
        else:
            act = np.zeros(n, np.int64)
        for i in np.nonzero(live)[0]:
            ob = {"grid": o["grid"][i], "stats": o["stats"][i]}
            if learner is None:
                lab_a = scripts[i].act(ob)  # the real script, acting
                act[i] = lab_a
            else:
                lab_a = labelers[i].label(ob)  # the script's opinion of the learner's state
            g_log[i].append(o["grid"][i].reshape(-1))
            s_log[i].append(o["stats"][i])
            a_log[i].append(lab_a)
        act = np.where(live, act, P.WAIT)
        was = live.copy()
        env.step(act)
        live &= ~(env.done & was)
        first = torch.from_numpy(env.done.astype(np.float32))
    out = Episodes()
    for i in range(n):
        out.add(np.stack(g_log[i]), np.stack(s_log[i]), np.array(a_log[i]))
    return out


def _roll_job(args) -> Episodes:
    spec_json, seeds, model_path, kind = args
    specs = [ScenarioSpec.from_json(s) for s in spec_json]
    learner = None
    if model_path:
        import torch
        from .experiments import load_model
        learner = load_model(model_path, kind, which="clone")
    return _roll(specs, seeds, learner)


def collect(n_episodes: int, seed: int = 0, workers: int = 4, learner_dir: str = None, kind: str = None,
            chunk: int = 25) -> Episodes:
    """Episodes on the arm B training distribution (`sample_train`), labelled by the memory script."""
    rng = np.random.default_rng([0x1417A, seed])
    specs = [sample_train(rng) for _ in range(n_episodes)]
    jobs = []
    for lo in range(0, n_episodes, chunk):
        sl = specs[lo:lo + chunk]
        jobs.append(([s.to_json() for s in sl], list(range(lo, lo + len(sl))), learner_dir, kind))
    out = Episodes()
    if workers <= 1:
        for j in jobs:
            out.extend(_roll_job(j))
    else:
        with ProcessPoolExecutor(workers) as ex:
            for ep in ex.map(_roll_job, jobs):
                out.extend(ep)
    return out


# --------------------------------------------------------------------------------------- training
def _batch_tensors(ep: Episodes, idx: Sequence[int], lo: int, hi: int):
    """(T, B, D) float32 observations, (T, B) actions, (T, B) mask and 'first' flags for steps [lo, hi)."""
    import torch
    B = len(idx)
    T = hi - lo
    D = GRID_DIM + P.STATS_DIM
    x = np.zeros((T, B, D), np.float32)
    a = np.zeros((T, B), np.int64)
    m = np.zeros((T, B), np.float32)
    f = np.zeros((T, B), np.float32)
    for j, i in enumerate(idx):
        L = len(ep.act[i])
        h = min(hi, L)
        if h <= lo:
            continue
        x[:h - lo, j, :GRID_DIM] = ep.grid[i][lo:h]
        x[:h - lo, j, GRID_DIM:] = ep.stats[i][lo:h]
        a[:h - lo, j] = ep.act[i][lo:h]
        m[:h - lo, j] = 1.0
        if lo == 0:
            f[0, j] = 1.0
    return torch.from_numpy(x), torch.from_numpy(a), torch.from_numpy(m), torch.from_numpy(f)


def train_clone(kind: str, ep: Episodes, epochs: int, seed: int, window: int = 64, lr: float = 1e-3,
                model=None, log=None):
    """Cross-entropy cloning. FF: shuffled minibatches. LSTM: whole episodes in order, truncated BPTT over `window`
    steps with the carried state detached between windows (the same scheme PPO uses). Samples per optimiser step
    are STEP_BUDGET for every architecture and window, so only the backpropagation length differs."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .models import make_model
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng([0xC10E, seed])
    model = model or make_model(kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    n_ep = len(ep)
    total_steps = max(1, epochs * int(np.ceil(ep.n_samples / STEP_BUDGET)))
    done = 0
    t0 = time.time()
    model.train()
    if kind == "ff":
        grid_all, stats_all = np.concatenate(ep.grid), np.concatenate(ep.stats)
        acts = np.concatenate(ep.act).astype(np.int64)
    for e in range(epochs):
        if kind == "ff":
            perm = rng.permutation(len(acts))
            for lo in range(0, len(perm), STEP_BUDGET):
                idx = np.sort(perm[lo:lo + STEP_BUDGET])
                xb = torch.from_numpy(np.concatenate([grid_all[idx].astype(np.float32), stats_all[idx]], 1))
                for g in opt.param_groups:
                    g["lr"] = lr * (1 - done / total_steps)
                logits, _, _ = model.sequence(xb, None, None)
                loss = F.cross_entropy(logits, torch.from_numpy(acts[idx]))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                done += 1
        else:
            B = max(1, STEP_BUDGET // window)
            order = rng.permutation(n_ep)
            for lo in range(0, n_ep, B):
                idx = order[lo:lo + B]
                Lmax = max(len(ep.act[i]) for i in idx)
                state = model.initial_state(len(idx))
                for w0 in range(0, Lmax, window):
                    x, a, m, f = _batch_tensors(ep, idx, w0, min(Lmax, w0 + window))
                    for g in opt.param_groups:
                        g["lr"] = lr * max(0.0, 1 - done / total_steps)
                    logits, _, st = model.sequence(x, state, f)
                    ce = F.cross_entropy(logits.reshape(-1, P.N_ACTIONS), a.reshape(-1), reduction="none")
                    loss = (ce * m.reshape(-1)).sum() / m.sum().clamp(min=1.0)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    state = tuple(s.detach() for s in st)
                    done += 1
        if log:
            log(f"    {kind} w{window} seed {seed} epoch {e + 1}/{epochs} loss {float(loss.detach()):.4f} ({time.time() - t0:.0f}s)")
    model.eval()
    return model


def action_agreement(model, kind: str, ep: Episodes, max_eps: int = 100) -> float:
    """Fraction of held-out steps where the clone's argmax equals the script's label (diagnostic only)."""
    import torch
    n = min(len(ep), max_eps)
    hit = tot = 0
    with torch.no_grad():
        for lo in range(0, n, 25):
            idx = list(range(lo, min(n, lo + 25)))
            Lmax = max(len(ep.act[i]) for i in idx)
            x, a, m, f = _batch_tensors(ep, idx, 0, Lmax)
            logits, _, _ = model.sequence(x, model.initial_state(len(idx)), f) if model.recurrent else \
                model.sequence(x.reshape(-1, x.shape[-1]), None, None)
            pred = logits.reshape(-1, P.N_ACTIONS).argmax(-1)
            ok = (pred == a.reshape(-1)).float() * m.reshape(-1)
            hit += float(ok.sum())
            tot += float(m.sum())
    return hit / max(tot, 1.0)


# --------------------------------------------------------------------------------------- scoring
def aggregate(ev: dict) -> float:
    """Mean over conditions of the mean ladder-normalised score (the unit used for a single model)."""
    return float(np.mean([per_seed_score(ev[c]) for c in CONDITIONS]))


def score_policy(policy, suites: Dict[str, Suite]) -> dict:
    out = {}
    for name, suite in suites.items():
        ev = suite.evaluate(policy)
        out[name] = {"aggregate": aggregate(ev), "per_condition": {c: per_seed_score(ev[c]) for c in CONDITIONS}}
    return out


def script_reference(suites: Dict[str, Suite]) -> dict:
    out = {}
    for name, suite in suites.items():
        sc = suite.script_scores("memory_script")
        per = {}
        for c in CONDITIONS:
            v = sc[c]["valid"]
            per[c] = float(np.mean(sc[c]["score"][v])) if v.any() else float("nan")
        out[name] = {"aggregate": float(np.nanmean(list(per.values()))), "per_condition": per}
    return out


def save_episodes(ep: Episodes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, grid=np.concatenate(ep.grid), stats=np.concatenate(ep.stats), act=np.concatenate(ep.act),
             lens=np.array([len(a) for a in ep.act]))


def load_episodes(path: str) -> Episodes:
    z = np.load(path)
    out, lo = Episodes(), 0
    for L in z["lens"]:
        out.add(z["grid"][lo:lo + L], z["stats"][lo:lo + L], z["act"][lo:lo + L])
        lo += int(L)
    return out


def _suites(tier_name: str) -> Dict[str, Suite]:
    tdir = tier_dir(tier_name)
    return {"suite_val": Suite.load(os.path.join(tdir, "suite_val.json")),
            "suite_val_s1": Suite.load(os.path.join(tdir, "suite_val_s1.json"))}


def _clone_job(args) -> dict:
    tier_name, key, kind, window, sd, epochs = args
    from .v2_1 import v21_dir
    odir = v21_dir(tier_name, "diagnose")
    data = load_episodes(os.path.join(odir, "data", "script_train.npz"))
    held = load_episodes(os.path.join(odir, "data", "script_heldout.npz"))
    m = train_clone(kind, data, epochs, sd, window=window)
    _save_clone(m, os.path.join(odir, "clones", key, f"seed_{sd}"))
    return {"key": key, "seed": sd, "agreement_heldout": action_agreement(m, kind, held),
            "scores": score_policy(m, _suites(tier_name))}


def run(tier_name: str, workers: int = 4, dagger_rounds: int = 3, log=print) -> dict:
    """Phase 4.1 end to end. Writes results/<tier>/v2_1/diagnose/imitation.json (diagnostic)."""
    from .v2_1 import v21_dir
    tier = load_tier(tier_name)
    odir = v21_dir(tier_name, "diagnose")
    n_ep, epochs, seeds = tier["imitation_episodes"], tier["imitation_epochs"], tier["imitation_seeds"]
    suites = _suites(tier_name)
    t0 = time.time()
    data = collect(n_ep, seed=0, workers=workers)
    held = collect(max(10, n_ep // 20), seed=1, workers=workers)
    save_episodes(data, os.path.join(odir, "data", "script_train.npz"))
    save_episodes(held, os.path.join(odir, "data", "script_heldout.npz"))
    log(f"  collected {len(data)} episodes, {data.n_samples} samples in {time.time() - t0:.0f}s")
    out = {"tier_label": "diagnostic", "episodes": len(data), "samples": data.n_samples, "epochs": epochs,
           "step_budget": STEP_BUDGET, "script": script_reference(suites), "clones": {}, "dagger": {}}
    configs = [("ff", "ff", 64), ("lstm_w64", "lstm", 64), ("lstm_w256", "lstm", 256)]
    jobs = [(tier_name, key, kind, w, sd, epochs) for key, kind, w in configs for sd in range(seeds)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_clone_job, jobs))
    for key, kind, w in configs:
        rs = sorted([r for r in rows if r["key"] == key], key=lambda r: r["seed"])
        out["clones"][key] = {"kind": kind, "window": w, "seeds": rs,
                              "aggregate_mean": {s: float(np.mean([r["scores"][s]["aggregate"] for r in rs])) for s in suites}}
        log(f"  {key}: " + ", ".join(f"{s} {v:.3f}" for s, v in out["clones"][key]["aggregate_mean"].items()))
    write_json(os.path.join(odir, "imitation.json"), out)
    out["dagger"] = dagger(tier_name, dagger_rounds, odir, log)
    write_json(os.path.join(odir, "imitation.json"), out)
    return out


def _save_clone(model, d: str):
    import torch
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "model_clone.pt"))


def _dagger_job(args) -> tuple:
    tier_name, kind, window, rounds, odir = args
    tier = load_tier(tier_name)
    suites = _suites(tier_name)
    epochs = max(1, tier["imitation_epochs"] // 2)
    n_new = max(10, tier["imitation_episodes"] // 4)
    key = f"{kind}_w{window}" if kind == "lstm" else "ff"
    agg = Episodes()
    agg.extend(load_episodes(os.path.join(odir, "data", "script_train.npz")))
    model = train_clone(kind, agg, tier["imitation_epochs"], 0, window=window)
    rounds_out = [{"round": 0, "samples": agg.n_samples, "scores": score_policy(model, suites)}]
    for r in range(1, rounds + 1):
        d = os.path.join(odir, "dagger", key, f"round_{r}")
        _save_clone(model, d)
        agg.extend(collect(n_new, seed=100 + r, workers=1, learner_dir=d, kind=kind))
        model = train_clone(kind, agg, epochs, r, window=window, model=model)
        rounds_out.append({"round": r, "samples": agg.n_samples, "scores": score_policy(model, suites)})
    return key, rounds_out


def dagger(tier_name: str, rounds: int, odir: str, log=print) -> dict:
    """DAgger-lite: roll out the current clone, label its visited states with the script, aggregate, retrain."""
    jobs = [(tier_name, "ff", 64, rounds, odir), (tier_name, "lstm", 256, rounds, odir)]
    with ProcessPoolExecutor(2) as ex:
        out = dict(ex.map(_dagger_job, jobs))
    for key, rs in out.items():
        log(f"  dagger {key}: " + " -> ".join(f"{r['scores']['suite_val']['aggregate']:.3f}" for r in rs))
    return out
