"""Policy networks for ladder rungs 5, 7 and 8.

All three share one interface so the same PPO loop trains them:

    initial_state(n)                      -> state (None for the feedforward net)
    step(obs, state, first)               -> logits, value, new_state     one time step
    sequence(obs, state, first)           -> logits, value, new_state     (T, n, ...) at once

`first[t]` is 1 where obs[t] starts a new episode: the recurrent state is reset to
zero *before* that observation is processed, which is the hidden-state reset at
episode boundaries the PPO tests check.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import params as P

OBS_DIM = P.OBS_DIM
N_ACTIONS = P.N_ACTIONS


def _init(layer: nn.Linear, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class FFPolicy(nn.Module):
    """Rung 5: separate actor and critic MLPs over the flattened observation, no memory."""
    recurrent = False
    name = "ppo_ff"

    def __init__(self, hidden=(256, 128), obs_dim=OBS_DIM):
        super().__init__()
        def mlp():
            return nn.Sequential(_init(nn.Linear(obs_dim, hidden[0])), nn.Tanh(),
                                 _init(nn.Linear(hidden[0], hidden[1])), nn.Tanh())
        self.actor, self.critic = mlp(), mlp()
        self.pi = _init(nn.Linear(hidden[1], N_ACTIONS), 0.01)
        self.v = _init(nn.Linear(hidden[1], 1), 1.0)

    def initial_state(self, n: int):
        return None

    def select_state(self, state, idx):
        return None

    def cat_states(self, states):
        return None

    def sequence(self, obs, state, first):
        return self.pi(self.actor(obs)), self.v(self.critic(obs)).squeeze(-1), None

    def step(self, obs, state, first):
        return self.sequence(obs, state, first)


class LSTMPolicy(nn.Module):
    """Rung 7: shared encoder, one LSTM layer, actor and critic heads on the hidden state."""
    recurrent = True
    name = "ppo_lstm"

    def __init__(self, enc=192, hidden=192, obs_dim=OBS_DIM):
        super().__init__()
        self.hidden = hidden
        self.enc = _init(nn.Linear(obs_dim, enc))
        self.cell = nn.LSTMCell(enc, hidden)
        for n_, p in self.cell.named_parameters():
            if "bias" in n_:
                nn.init.constant_(p, 0.0)
            else:
                nn.init.orthogonal_(p, 1.0)
        self.pi = _init(nn.Linear(hidden, N_ACTIONS), 0.01)
        self.v = _init(nn.Linear(hidden, 1), 1.0)

    def initial_state(self, n: int):
        return (torch.zeros(n, self.hidden), torch.zeros(n, self.hidden))

    def select_state(self, state, idx):
        return tuple(s[idx] for s in state)

    def cat_states(self, states):
        return tuple(torch.cat(parts, 0) for parts in zip(*states))

    def sequence(self, obs, state, first):
        # obs: (T, n, D) or (n, D) for a single step
        single = obs.dim() == 2
        if single:
            obs, first = obs.unsqueeze(0), first.unsqueeze(0)
        x = torch.tanh(self.enc(obs))
        h, c = state
        outs = []
        for t in range(x.shape[0]):
            keep = (1.0 - first[t]).unsqueeze(-1)
            h, c = self.cell(x[t], (h * keep, c * keep))
            outs.append(h)
        out = torch.stack(outs, 0)
        logits, value = self.pi(out), self.v(out).squeeze(-1)
        if single:
            logits, value = logits.squeeze(0), value.squeeze(0)
        return logits, value, (h, c)

    def step(self, obs, state, first):
        return self.sequence(obs, state, first)


class SSMPolicy(nn.Module):
    """Rung 8: a small Mamba-style selective state-space block.

    Input-dependent step size, B and C (the "selective" part), a short causal conv, a gated
    output, and a residual connection. The scan is a plain Python loop over time, which is
    slow but simple enough to verify against the step-by-step pass.
    """
    recurrent = True
    name = "ppo_ssm"

    def __init__(self, d_model=192, d_state=16, expand=2, d_conv=4, obs_dim=OBS_DIM):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = d_model * expand
        self.dt_rank = max(4, d_model // 16)
        self.enc = _init(nn.Linear(obs_dim, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv_w = nn.Parameter(torch.randn(self.d_inner, d_conv) * 0.2)
        self.conv_b = nn.Parameter(torch.zeros(self.d_inner))
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        with torch.no_grad():  # dt bias so softplus(dt) starts in [1e-3, 1e-1]
            dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.pi = _init(nn.Linear(d_model, N_ACTIONS), 0.01)
        self.v = _init(nn.Linear(d_model, 1), 1.0)

    def initial_state(self, n: int):
        return (torch.zeros(n, self.d_inner, self.d_state), torch.zeros(n, self.d_inner, self.d_conv - 1))

    def select_state(self, state, idx):
        return tuple(s[idx] for s in state)

    def cat_states(self, states):
        return tuple(torch.cat(parts, 0) for parts in zip(*states))

    def _cell(self, x, h, conv):
        """One time step. x: (n, d_model); h: (n, d_inner, d_state); conv: (n, d_inner, d_conv-1)."""
        xz = self.in_proj(self.norm(x))
        u, z = xz.chunk(2, dim=-1)
        window = torch.cat([conv, u.unsqueeze(-1)], dim=-1)  # (n, d_inner, d_conv)
        u = F.silu((window * self.conv_w).sum(-1) + self.conv_b)
        conv = window[..., 1:]
        dbc = self.x_proj(u)
        dt, B, C = dbc.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (n, d_inner)
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (n, d_inner, d_state)
        dBu = dt.unsqueeze(-1) * B.unsqueeze(1) * u.unsqueeze(-1)
        h = dA * h + dBu
        y = (h * C.unsqueeze(1)).sum(-1) + self.D * u
        y = y * F.silu(z)
        return x + self.out_proj(y), h, conv

    def sequence(self, obs, state, first):
        single = obs.dim() == 2
        if single:
            obs, first = obs.unsqueeze(0), first.unsqueeze(0)
        x = torch.tanh(self.enc(obs))
        h, conv = state
        outs = []
        for t in range(x.shape[0]):
            keep = (1.0 - first[t])
            h, conv = h * keep.view(-1, 1, 1), conv * keep.view(-1, 1, 1)
            o, h, conv = self._cell(x[t], h, conv)
            outs.append(o)
        out = torch.stack(outs, 0)
        logits, value = self.pi(out), self.v(out).squeeze(-1)
        if single:
            logits, value = logits.squeeze(0), value.squeeze(0)
        return logits, value, (h, conv)

    def step(self, obs, state, first):
        return self.sequence(obs, state, first)


def make_model(kind: str, obs_dim: int = OBS_DIM) -> nn.Module:
    return {"ff": FFPolicy, "lstm": LSTMPolicy, "ssm": SSMPolicy}[kind](obs_dim=obs_dim)


def clone_state(state):
    if state is None:
        return None
    return tuple(s.clone() for s in state)
