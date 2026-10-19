"""What-it-learned probes. Each probe states what it does NOT show.

  hydration threshold   constructed states (a real observation with the hydration stat overwritten):
                        behaviour in states that may be off-distribution, not in natural play.
  distilled tree        a depth-3 tree that agrees with the policy 80% of the time approximates it;
                        it is not the policy's logic.
  occlusion             input sensitivity only, never reasoning.
  training time machine when behaviours appear, not why.
  point of no return    valid for the planner's search power, not a proof that no plan exists.
"""
from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core import params as P
from ..core.baselines import Policy
from ..core.env import SurvivalEnv
from ..core.scenario import ScenarioSpec, build_scenario
from .evaluate import evaluate, survival_array
from .experiments import get_val_suite, load_model, load_tier, tier_dir, write_json
from .replays import record_replay, strip_private

C = P.VIEW_R
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
FEATURES = ["hydration", "energy", "cold", "water_dist", "water_dy", "water_dx", "food_dist", "food_dy",
            "food_dx", "on_water", "on_food", "wall_up", "wall_down", "wall_left", "wall_right", "prev_valid"]


def _flat(obs: dict) -> np.ndarray:
    return np.concatenate([obs["grid"].reshape(-1), obs["stats"]]).astype(np.float32)


def collect_states(model, specs: Sequence[ScenarioSpec], max_per_episode: int = 400) -> List[dict]:
    """Roll the policy out (deterministic) and keep (obs, hidden state, action, episode id) per step."""
    import torch
    model.eval()
    out = []
    for ep, spec in enumerate(specs):
        env = SurvivalEnv()
        obs = env.reset(spec)
        state = model.initial_state(1)
        first = torch.ones(1)
        k = 0
        while not env.done and k < max_per_episode:
            flat = torch.from_numpy(_flat(obs)[None])
            before = None if state is None else tuple(s.clone() for s in state)
            with torch.no_grad():
                logits, _, state = model.step(flat, state, first)
            a = int(logits.argmax(-1))
            out.append({"obs": obs, "state": before, "first": float(first[0]), "action": a, "episode": ep,
                        "t": env.t, "probs": torch.softmax(logits, -1)[0].numpy()})
            obs, *_ = env.step(a)
            first = torch.zeros(1)
            k += 1
    return out


def _probs(model, obs_list: List[dict], states: list, firsts: list):
    import torch
    flat = torch.from_numpy(np.stack([_flat(o) for o in obs_list]))
    if states[0] is None:
        st = None
    else:
        st = model.cat_states(states)
    with torch.no_grad():
        logits, _, _ = model.step(flat, st, torch.tensor(firsts, dtype=torch.float32))
    return torch.softmax(logits, -1).numpy()


# ----------------------------------------------------------------------------------- features
def _view_dist_map(grid: np.ndarray, ch: int):
    """BFS distance from the nearest ready resource of channel `ch` to every view cell (over floor)."""
    floor = grid[P.CH_FLOOR] > 0
    tgt = grid[ch] > 0
    dist = np.full(floor.shape, 99, dtype=np.int32)
    dq = deque()
    for y, x in zip(*np.nonzero(tgt)):
        dist[y, x] = 0
        dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in _DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < P.VIEW and 0 <= nx < P.VIEW and floor[ny, nx] and dist[ny, nx] > dist[y, x] + 1:
                dist[ny, nx] = dist[y, x] + 1
                dq.append((ny, nx))
    return dist, bool(tgt.any())


def toward_actions(grid: np.ndarray, ch: int) -> List[int]:
    """Actions that make progress to the nearest in-view ready resource (interact if standing on it)."""
    dist, any_t = _view_dist_map(grid, ch)
    if not any_t or dist[C, C] >= 99:
        return []
    if dist[C, C] == 0:
        return [P.INTERACT]
    return [a for a, (dy, dx) in enumerate(_DIRS) if dist[C + dy, C + dx] < dist[C, C]]


def features(obs: dict) -> np.ndarray:
    g, s = obs["grid"], obs["stats"]
    f = np.zeros(len(FEATURES), dtype=np.float32)
    f[0], f[1], f[2] = s[P.S_HYDRATION] * 100, s[P.S_ENERGY] * 100, s[P.S_COLD]
    for base, ch in ((3, P.CH_WATER), (6, P.CH_FOOD)):
        ys, xs = np.nonzero(g[ch])
        if len(ys):
            d = np.abs(ys - C) + np.abs(xs - C)
            i = int(np.argmin(d))
            f[base], f[base + 1], f[base + 2] = d[i], np.sign(ys[i] - C), np.sign(xs[i] - C)
        else:
            f[base], f[base + 1], f[base + 2] = 12, 0, 0
    f[9], f[10] = g[P.CH_WATER, C, C], g[P.CH_FOOD, C, C]
    f[11], f[12], f[13], f[14] = g[P.CH_WALL, C - 1, C], g[P.CH_WALL, C + 1, C], g[P.CH_WALL, C, C - 1], g[P.CH_WALL, C, C + 1]
    f[15] = s[P.S_PREV_VALID]
    return f


# ----------------------------------------------------------------------------------- probes
def hydration_threshold(model, states: List[dict], levels=range(0, 101, 5), max_states: int = 300) -> dict:
    """P(heads for visible ready water) as hydration is swept 0..100; energy held at its original value."""
    cand = [s for s in states if s["obs"]["grid"][P.CH_WATER].any() and toward_actions(s["obs"]["grid"], P.CH_WATER)
            and np.abs(np.nonzero(s["obs"]["grid"][P.CH_WATER])[0] - C).min() >= 0]
    rng = np.random.default_rng(0)
    if len(cand) > max_states:
        cand = [cand[i] for i in rng.choice(len(cand), max_states, replace=False)]
    curve = []
    for lv in levels:
        obs_l = []
        for s in cand:
            o = {"grid": s["obs"]["grid"], "stats": s["obs"]["stats"].copy()}
            o["stats"][P.S_HYDRATION] = lv / 100.0
            obs_l.append(o)
        p = _probs(model, obs_l, [s["state"] for s in cand], [s["first"] for s in cand])
        head = [p[i, toward_actions(cand[i]["obs"]["grid"], P.CH_WATER)].sum() for i in range(len(cand))]
        curve.append(float(np.mean(head)))
    return {"levels": list(levels), "p_head_for_water": curve, "n_states": len(cand),
            "script_rule_hydration": 40, "reading": "constructed states (hydration overwritten) may be off-distribution"}


class TreePolicy(Policy):
    name = "distilled_tree"

    def __init__(self, tree):
        self.tree = tree

    def act(self, obs: dict) -> int:
        return int(self.tree.predict(features(obs)[None])[0])


def distilled_tree(model, train_specs, test_specs, eval_specs) -> dict:
    from sklearn.tree import DecisionTreeClassifier, export_text
    tr = collect_states(model, train_specs)
    te = collect_states(model, test_specs)
    X = np.stack([features(s["obs"]) for s in tr])
    y = np.array([s["action"] for s in tr])
    Xt = np.stack([features(s["obs"]) for s in te])
    yt = np.array([s["action"] for s in te])
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    agree = float((tree.predict(Xt) == yt).mean())
    tree_surv = float(survival_array(evaluate(TreePolicy(tree), eval_specs)).mean())
    pol_surv = float(survival_array(evaluate(model, eval_specs)).mean())
    t = tree.tree_
    nodes = [{"feature": FEATURES[t.feature[i]] if t.feature[i] >= 0 else None, "threshold": float(t.threshold[i]),
              "left": int(t.children_left[i]), "right": int(t.children_right[i]),
              "class": int(np.argmax(t.value[i])) if t.children_left[i] < 0 else None,
              "n": int(t.n_node_samples[i])} for i in range(t.node_count)]
    return {"agreement_heldout": agree, "tree_survival": tree_surv, "policy_survival": pol_surv,
            "text": export_text(tree, feature_names=FEATURES), "nodes": nodes, "action_names": list(P.ACTION_NAMES),
            "reading": "a tree that agrees often approximates the policy; it is not the policy's logic"}


def occlusion(model, states: List[dict], max_states: int = 150) -> dict:
    rng = np.random.default_rng(1)
    if len(states) > max_states:
        states = [states[i] for i in rng.choice(len(states), max_states, replace=False)]
    heat = np.zeros((P.VIEW, P.VIEW))
    for y in range(P.VIEW):
        for x in range(P.VIEW):
            obs_l = []
            for s in states:
                g = s["obs"]["grid"].copy()
                g[:, y, x] = 0.0
                obs_l.append({"grid": g, "stats": s["obs"]["stats"]})
            p_occ = _probs(model, obs_l, [s["state"] for s in states], [s["first"] for s in states])
            p0 = np.stack([s["probs"] for s in states])
            heat[y, x] = 0.5 * np.abs(p_occ - p0).sum(1).mean()  # total variation distance
    return {"heatmap": heat.tolist(), "n_states": len(states), "reading": "input sensitivity only, never reasoning"}


def time_machine(run_dir: str, kind: str, spec: ScenarioSpec) -> dict:
    out = {"scenario": spec.to_json(), "checkpoints": []}
    for pct in (0, 10, 50, 100):
        path = os.path.join(run_dir, f"model_pct{pct:03d}.pt")
        if not os.path.exists(path):
            continue
        import torch
        from .models import make_model
        m = make_model(kind)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        rep = record_replay(spec, m, label=f"{pct}% of training", with_probs=True)
        out["checkpoints"].append({"pct": pct, "survival": rep["survival"], "cause": rep["cause"],
                                   "replay": strip_private(rep)})
    out["reading"] = "shows when behaviours appear, not why"
    return out


def run(tier_name: str) -> dict:
    tier = load_tier(tier_name)
    tdir = tier_dir(tier_name)
    suite = get_val_suite(tdir, tier, 0)
    n_ep = tier["probe_episodes"]
    pool = suite.all_specs()
    rng = np.random.default_rng(2)
    order = rng.permutation(len(pool))
    train_specs = [pool[i] for i in order[: max(4, n_ep // 2)]]
    test_specs = [pool[i] for i in order[max(4, n_ep // 2): max(4, n_ep // 2) + max(4, n_ep // 2)]]
    eval_specs = suite.specs["unseen_layout"]
    out: dict = {"policies": {}}
    for arm in ("B", "D"):
        d = os.path.join(tdir, "headline", f"arm_{arm}", "seed_0")
        if not os.path.exists(os.path.join(d, "model_best.pt")):
            continue
        m = load_model(d, "ff")
        states = collect_states(m, test_specs)
        entry = {"hydration_threshold": hydration_threshold(m, states),
                 "distilled_tree": distilled_tree(m, train_specs, test_specs, eval_specs),
                 "occlusion": occlusion(m, states),
                 "time_machine": time_machine(d, "ff", suite.specs["familiar"][0])}
        out["policies"][f"arm_{arm}"] = entry
    # the reactive script's own rule, for the overlay
    out["script_overlay"] = {"reactive_threshold": 40, "opportunistic_threshold": 70}
    write_json(os.path.join(tdir, "probes", "probes.json"), out)
    return out
