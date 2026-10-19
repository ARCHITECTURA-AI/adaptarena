"""PPO written from scratch, CleanRL style: one readable file.

Contents: GAE with termination-vs-truncation handling, the clipped surrogate objective with
entropy bonus, value loss, advantage normalisation, linear learning-rate decay and gradient
clipping, vectorised rollouts over `BatchEnv`, and a recurrent variant that trains on
sequence minibatches with hidden-state resets at episode boundaries.

Termination vs truncation: when an episode is cut by the horizon the target bootstraps from
the value of the final observation (reward += gamma * V(s_T)); when the agent dies there is
no bootstrap. `bootstrap_timeout=False` flips the truncation case to the "time-out as death"
ablation.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..core import params as P
from ..core.batch import BatchEnv
from ..core.env import EnvConfig
from ..core.scenario import ScenarioSpec, build_scenario
from .evaluate import evaluate, survival_array


@dataclass
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 8
    n_steps: int = 256
    epochs: int = 10
    n_minibatches: int = 4
    seq_len: int = 64  # recurrent models train on chunks of this length (truncated BPTT)
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    bootstrap_timeout: bool = True
    reward_scale: float = 0.01  # returns reach ~100 with gamma .99; scaling keeps the value loss from swamping clipped gradients
    seed: int = 0
    eval_every: int = 25  # updates between validation evaluations
    save_fracs: tuple = (0.0, 0.1, 0.5, 1.0)


# --------------------------------------------------------------------------- core maths
def compute_gae(rewards, values, done, next_value, gamma, lam):
    """GAE over (T, n) arrays. done[t] = the episode ended at step t (reward[t] is its last).

    No bootstrap through a done step, so a death contributes no future value. A truncated step
    already carries gamma*V(final obs) inside rewards[t] and is also marked done.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        nv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def ppo_loss(logits, values, actions, old_logp, adv, returns, cfg: PPOConfig, normalise=True):
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(actions)
    log_ratio = logp - old_logp
    ratio = log_ratio.exp()
    if normalise:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)
    pg_loss = torch.max(pg1, pg2).mean()
    v_loss = 0.5 * ((values - returns) ** 2).mean()
    ent = dist.entropy().mean()
    loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfrac = ((ratio - 1.0).abs() > cfg.clip).float().mean()
    return loss, dict(policy_loss=pg_loss.item(), value_loss=v_loss.item(), entropy=ent.item(),
                      approx_kl=approx_kl.item(), clipfrac=clipfrac.item(), ratio_mean=ratio.mean().item())


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    var = np.var(target)
    return float("nan") if var == 0 else float(1 - np.var(target - pred) / var)


# --------------------------------------------------------------------------- scenario sources
class ScenarioSource:
    """Decides which scenario each environment plays next."""
    needs_pvl = False

    def sample(self, env_i: int):
        raise NotImplementedError

    def episode_end(self, meta, survival: int, pvl: Optional[float]) -> None:
        pass

    def stats(self) -> dict:
        return {}


class SamplerSource(ScenarioSource):
    """Arms A and B: draw from a fixed distribution, no state."""

    def __init__(self, sampler: Callable, seed: int):
        self.sampler = sampler
        self.rng = np.random.default_rng([0x50A1, seed])

    def sample(self, env_i):
        return self.sampler(self.rng), None


# --------------------------------------------------------------------------- training
@dataclass
class TrainResult:
    out_dir: str
    best_val: float
    best_update: int
    final_val: float
    steps: int
    wall_seconds: float
    sps: float
    extra: dict = field(default_factory=dict)


def train(model: nn.Module, cfg: PPOConfig, source: ScenarioSource, env_config: EnvConfig,
          val_specs: Sequence[ScenarioSpec], out_dir: str, log: Optional[Callable] = None) -> TrainResult:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    n, T = cfg.n_envs, cfg.n_steps
    batch_size = n * T
    seq = min(cfg.seq_len, T)
    n_updates = max(1, cfg.total_steps // batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    env = BatchEnv(n, env_config)
    metas = []
    for i in range(n):
        spec, meta = source.sample(i)
        env.reset_env(i, spec)
        metas.append(meta)
    pvl_acc = np.zeros(n)
    pvl_cnt = np.zeros(n)

    obs_buf = torch.zeros(T, n, P.OBS_DIM)
    act_buf = torch.zeros(T, n, dtype=torch.long)
    logp_buf = torch.zeros(T, n)
    rew_buf = torch.zeros(T, n)
    done_buf = torch.zeros(T, n)
    first_buf = torch.zeros(T, n)
    val_buf = torch.zeros(T, n)

    state = model.initial_state(n)
    first = torch.ones(n)
    obs = torch.from_numpy(env.flat_obs())

    rows_f = open(os.path.join(out_dir, "train_log.csv"), "w", newline="")
    val_f = open(os.path.join(out_dir, "val_log.csv"), "w", newline="")
    train_w = val_w = None
    val_seeds = list(range(len(val_specs)))
    best_val, best_update, last_val = -1.0, -1, float("nan")
    steps = 0
    t_start = time.time()
    save_marks = {int(round(f * n_updates)): f for f in cfg.save_fracs}

    def validate(update):
        nonlocal best_val, best_update, last_val, val_w
        res = evaluate(model, val_specs, val_seeds, env_config, deterministic=True)
        last_val = float(survival_array(res).mean())
        if val_w is None:
            val_w = csv.writer(val_f)
            val_w.writerow(["update", "steps", "val_survival"])
        val_w.writerow([update, steps, last_val])
        val_f.flush()
        if last_val > best_val:  # strict: on a tie the earlier checkpoint stays
            best_val, best_update = last_val, update
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

    if 0 in save_marks:
        torch.save(model.state_dict(), os.path.join(out_dir, "model_pct000.pt"))
    validate(0)

    for update in range(1, n_updates + 1):
        if cfg.anneal_lr:
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)
        chunk_states = []
        episode_events = []  # (t, env, meta, survival)
        for t in range(T):
            if model.recurrent and t % seq == 0:
                chunk_states.append(tuple(s.clone() for s in state))
            obs_buf[t], first_buf[t] = obs, first
            with torch.no_grad():
                logits, value, state = model.step(obs, state, first)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            val_buf[t], act_buf[t], logp_buf[t] = value, action, logp
            reward, term, trunc = env.step(action.numpy())
            reward = torch.from_numpy(reward) * cfg.reward_scale
            done_np = term | trunc
            if trunc.any() and cfg.bootstrap_timeout:
                with torch.no_grad():
                    fobs = torch.from_numpy(env.flat_obs())
                    _, vfin, _ = model.step(fobs, state, torch.zeros(n))
                reward = reward + cfg.gamma * vfin * torch.from_numpy(trunc.astype(np.float32))
            rew_buf[t] = reward
            done_buf[t] = torch.from_numpy(done_np.astype(np.float32))
            steps += n
            for i in np.nonzero(done_np)[0]:
                episode_events.append((t, int(i), metas[i], int(env.survived[i])))
                spec, meta = source.sample(int(i))
                env.reset_env(int(i), spec)
                metas[i] = meta
            obs = torch.from_numpy(env.flat_obs())
            first = torch.from_numpy(done_np.astype(np.float32))

        with torch.no_grad():
            _, next_value, _ = model.step(obs, state, first)
        adv, ret = compute_gae(rew_buf, val_buf, done_buf, next_value, cfg.gamma, cfg.gae_lambda)

        # -- optimisation
        stats_acc = {}
        if model.recurrent:
            C = seq
            n_chunks = T // C
            assert T % C == 0
            def chunk(x):  # (T, n, ...) -> (C, n_chunks * n, ...), batch index = chunk * n + env
                x = x.reshape(n_chunks, C, n, *x.shape[2:]).transpose(0, 1)
                return x.reshape(C, n_chunks * n, *x.shape[3:])
            c_obs, c_act, c_logp, c_adv, c_ret, c_first = map(
                chunk, (obs_buf, act_buf, logp_buf, adv, ret, first_buf))
            c_state = model.cat_states(chunk_states)
            B = n_chunks * n
            per_mb = max(1, B // cfg.n_minibatches)
            for _ in range(cfg.epochs):
                perm = np.random.permutation(B)
                for lo in range(0, B, per_mb):
                    idx = torch.as_tensor(perm[lo:lo + per_mb])
                    logits, values, _ = model.sequence(c_obs[:, idx], model.select_state(c_state, idx),
                                                       c_first[:, idx])
                    loss, st = ppo_loss(logits.reshape(-1, P.N_ACTIONS), values.reshape(-1),
                                        c_act[:, idx].reshape(-1), c_logp[:, idx].reshape(-1),
                                        c_adv[:, idx].reshape(-1), c_ret[:, idx].reshape(-1), cfg)
                    opt.zero_grad()
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    st["grad_norm"] = float(gn)
                    for k, v in st.items():
                        stats_acc.setdefault(k, []).append(v)
        else:
            flat = lambda x: x.reshape(batch_size, *x.shape[2:])
            b_obs, b_act, b_logp, b_adv, b_ret = map(flat, (obs_buf, act_buf, logp_buf, adv, ret))
            mb = batch_size // cfg.n_minibatches
            for _ in range(cfg.epochs):
                perm = np.random.permutation(batch_size)
                for lo in range(0, batch_size, mb):
                    idx = torch.as_tensor(perm[lo:lo + mb])
                    logits, values, _ = model.sequence(b_obs[idx], None, None)
                    loss, st = ppo_loss(logits, values, b_act[idx], b_logp[idx], b_adv[idx], b_ret[idx], cfg)
                    opt.zero_grad()
                    loss.backward()
                    gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    st["grad_norm"] = float(gn)
                    for k, v in st.items():
                        stats_acc.setdefault(k, []).append(v)

        # -- scenario-source bookkeeping (positive value loss per finished episode, then scores)
        if source.needs_pvl:
            adv_np = adv.clamp(min=0).numpy()
            ev_by_env = {}
            for (t, i, meta, surv) in episode_events:
                ev_by_env.setdefault(i, []).append((t, meta, surv))
            for i in range(n):
                prev = 0
                for (t, meta, surv) in ev_by_env.get(i, []):
                    seg = adv_np[prev:t + 1, i]
                    tot = pvl_acc[i] + seg.sum()
                    cnt = pvl_cnt[i] + len(seg)
                    source.episode_end(meta, surv, float(tot / max(cnt, 1)))
                    pvl_acc[i] = pvl_cnt[i] = 0
                    prev = t + 1
                pvl_acc[i] += adv_np[prev:, i].sum()
                pvl_cnt[i] += T - prev
        else:
            for (t, i, meta, surv) in episode_events:
                source.episode_end(meta, surv, None)

        # -- logging
        elapsed = time.time() - t_start
        m = {k: float(np.mean(v)) for k, v in stats_acc.items()}
        m.update(update=update, steps=steps, lr=opt.param_groups[0]["lr"], sps=steps / max(elapsed, 1e-9),
                 explained_variance=explained_variance(val_buf.numpy().ravel(), ret.numpy().ravel()),
                 adv_mean=float(adv.mean()), adv_std=float(adv.std()),
                 mean_reward_per_step=float(rew_buf.mean()),
                 episodes=len(episode_events),
                 ep_survival=float(np.mean([e[3] for e in episode_events])) if episode_events else float("nan"))
        if train_w is None:
            train_w = csv.DictWriter(rows_f, fieldnames=list(m.keys()))
            train_w.writeheader()
        train_w.writerow(m)
        if update % 10 == 0:
            rows_f.flush()
        if update % cfg.eval_every == 0 or update == n_updates:
            validate(update)
            if log:
                log(f"[{update}/{n_updates}] steps={steps} val={last_val:.1f} best={best_val:.1f} sps={m['sps']:.0f}")
        if update in save_marks and save_marks[update] > 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_pct{int(round(save_marks[update] * 100)):03d}.pt"))

    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    rows_f.close()
    val_f.close()
    wall = time.time() - t_start
    result = TrainResult(out_dir=out_dir, best_val=best_val, best_update=best_update, final_val=last_val,
                         steps=steps, wall_seconds=wall, sps=steps / wall, extra=source.stats())
    with open(os.path.join(out_dir, "train_result.json"), "w") as f:
        json.dump({**asdict(result), "config": asdict(cfg), "model": model.name,
                   "n_params": sum(p.numel() for p in model.parameters())}, f, indent=1, default=str)
    return result
