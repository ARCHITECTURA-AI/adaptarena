import numpy as np
import pytest
import torch

import adaptarena.core.params as P
from adaptarena.core import BatchEnv, EnvConfig, ScenarioSpec, build_scenario
from adaptarena.lab.models import FFPolicy, LSTMPolicy, SSMPolicy, count_params
from adaptarena.lab.ppo import PPOConfig, SamplerSource, compute_gae, ppo_loss, train
from adaptarena.core.distributions import sample_train


def test_gae_hand_computed_with_death_and_truncation():
    # one env, 4 steps: reward 1,1,1(death at t=2),1 ; values chosen by hand
    gamma, lam = 0.9, 0.8
    r = torch.tensor([[1.0], [1.0], [0.0], [1.0]])  # death step has reward 0
    v = torch.tensor([[2.0], [3.0], [1.0], [4.0]])
    d = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    nv = torch.tensor([5.0])
    adv, ret = compute_gae(r, v, d, nv, gamma, lam)
    # t=3: delta = 1 + .9*5 - 4 = 1.5
    a3 = 1.5
    # t=2 (death): delta = 0 + 0 - 1 = -1, no bootstrap, no carry from t=3
    a2 = -1.0
    # t=1: delta = 1 + .9*1 - 3 = -1.1 ; a1 = delta + .9*.8*a2
    a1 = -1.1 + 0.72 * a2
    a0 = (1 + 0.9 * 3 - 2) + 0.72 * a1
    assert torch.allclose(adv.squeeze(), torch.tensor([a0, a1, a2, a3]), atol=1e-6)
    assert torch.allclose(ret, adv + v)


def test_truncation_value_target_differs_from_death():
    """A truncated step carries gamma*V(final) in its reward; a death step carries nothing."""
    gamma = 0.99
    v_final = 50.0
    r_trunc = torch.tensor([[1.0 + gamma * v_final]])
    r_death = torch.tensor([[0.0]])
    d = torch.tensor([[1.0]])
    v = torch.tensor([[10.0]])
    nv = torch.tensor([123.0])  # must be ignored: the episode ended
    _, ret_t = compute_gae(r_trunc, v, d, nv, gamma, 0.95)
    _, ret_d = compute_gae(r_death, v, d, nv, gamma, 0.95)
    assert ret_t.item() == pytest.approx(1 + gamma * v_final)
    assert ret_d.item() == pytest.approx(0.0)


def test_ratio_one_and_clipfrac_zero_on_first_epoch():
    torch.manual_seed(0)
    m = FFPolicy()
    obs = torch.rand(64, P.OBS_DIM)
    logits, values, _ = m.sequence(obs, None, None)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    old = dist.log_prob(a).detach()
    loss, st = ppo_loss(logits, values, a, old, torch.randn(64), torch.randn(64), PPOConfig())
    assert st["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert st["clipfrac"] == 0.0 and abs(st["approx_kl"]) < 1e-6


def test_overfit_one_fixed_batch():
    torch.manual_seed(0)
    m = FFPolicy()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    obs = torch.rand(128, P.OBS_DIM)
    a = torch.randint(0, 6, (128,))
    ret = torch.randn(128)
    adv = torch.randn(128)
    with torch.no_grad():
        logits, _, _ = m.sequence(obs, None, None)
        old = torch.distributions.Categorical(logits=logits).log_prob(a)
    cfg = PPOConfig(clip=10.0, ent_coef=0.0)  # no clipping: pure fit of value head and policy
    first = None
    for i in range(400):
        logits, values, _ = m.sequence(obs, None, None)
        loss, st = ppo_loss(logits, values, a, old, adv, ret, cfg, normalise=False)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else st["value_loss"]
    assert st["value_loss"] < 0.01 * first


@pytest.mark.parametrize("cls", [LSTMPolicy, SSMPolicy])
def test_sequence_pass_equals_step_by_step(cls):
    torch.manual_seed(0)
    m = cls()
    T, n = 12, 3
    obs = torch.rand(T, n, P.OBS_DIM)
    first = torch.zeros(T, n)
    first[0] = 1
    first[5, 1] = 1  # an episode boundary in the middle of env 1
    first[8, 2] = 1
    state = m.initial_state(n)
    lo, vo, _ = m.sequence(obs, state, first)
    s = m.initial_state(n)
    for t in range(T):
        l, v, s = m.step(obs[t], s, first[t])
        assert torch.allclose(l, lo[t], atol=1e-5) and torch.allclose(v, vo[t], atol=1e-5)


@pytest.mark.parametrize("cls", [LSTMPolicy, SSMPolicy])
def test_hidden_state_resets_at_episode_boundary(cls):
    torch.manual_seed(1)
    m = cls()
    o = torch.rand(1, P.OBS_DIM)
    s0 = m.initial_state(1)
    l_fresh, _, _ = m.step(o, s0, torch.ones(1))
    # run some history, then a "first" flag must erase it
    s = m.initial_state(1)
    for _ in range(5):
        _, _, s = m.step(torch.rand(1, P.OBS_DIM), s, torch.zeros(1))
    l_reset, _, _ = m.step(o, s, torch.ones(1))
    l_carry, _, _ = m.step(o, s, torch.zeros(1))
    assert torch.allclose(l_fresh, l_reset, atol=1e-6)
    assert (l_fresh - l_carry).abs().max() > 1e-6


def test_param_counts_comparable():
    ff, lstm, ssm = count_params(FFPolicy()), count_params(LSTMPolicy()), count_params(SSMPolicy())
    assert 0.8 < lstm / ff < 1.25 and 0.7 < ssm / ff < 1.25, (ff, lstm, ssm)
    assert abs(ssm / lstm - 1) <= 0.10, (lstm, ssm)  # v2.1 plan 5b.1: SSM within 10% of the LSTM


@pytest.mark.parametrize("cls", [LSTMPolicy, SSMPolicy])
def test_chunked_scan_carrying_state_matches_step_by_step(cls):
    """Training runs in chunks with the state carried across chunk boundaries; it must equal one step-by-step pass."""
    torch.manual_seed(3)
    m = cls()
    T, n, C = 20, 3, 7
    obs = torch.rand(T, n, P.OBS_DIM)
    first = torch.zeros(T, n)
    first[0] = 1
    first[11, 1] = 1  # a boundary inside a chunk
    first[14, 2] = 1  # a boundary exactly at a chunk start
    s = m.initial_state(n)
    ref_l, ref_v = [], []
    for t in range(T):
        l, v, s = m.step(obs[t], s, first[t])
        ref_l.append(l)
        ref_v.append(v)
    ref_l, ref_v = torch.stack(ref_l), torch.stack(ref_v)
    s = m.initial_state(n)
    ls, vs = [], []
    for lo in range(0, T, C):
        l, v, s = m.sequence(obs[lo:lo + C], s, first[lo:lo + C])
        ls.append(l)
        vs.append(v)
    assert torch.allclose(torch.cat(ls), ref_l, atol=1e-5) and torch.allclose(torch.cat(vs), ref_v, atol=1e-5)


def test_running_mean_std_matches_numpy():
    from adaptarena.lab.ppo import RunningMeanStd
    rng = np.random.default_rng(0)
    rms, chunks = RunningMeanStd(), [rng.normal(3.0, 2.0, size=int(k)) for k in rng.integers(1, 40, size=30)]
    for c in chunks:
        rms.update(c)
    allv = np.concatenate(chunks)
    assert abs(rms.mean - allv.mean()) < 1e-3 and abs(rms.var - allv.var()) < 1e-3  # 1e-4 initial pseudo-count only


def test_return_normalizer_resets_at_episode_ends_and_clips():
    from adaptarena.lab.ppo import ReturnNormalizer
    g = 0.9
    nz = ReturnNormalizer(2, g, clip=5.0)
    # env 0 runs three steps of reward 1 and ends on the third; env 1 never ends
    refs = [np.zeros(2)]
    acc = np.zeros(2)
    for step in range(5):
        r = np.array([1.0, 1.0])
        done = np.array([1.0 if step == 2 else 0.0, 0.0])
        nz(r, done)
        acc = acc * g + r                    # what the accumulator held when the variance was updated
        refs.append(acc.copy())
        acc = acc * (1 - done)               # then env 0 is reset at the episode boundary
        assert np.allclose(nz.acc, acc)
    assert nz.acc[0] == pytest.approx(1.0 + g, abs=1e-12)  # restarted from zero after step 2, then two more steps
    assert nz.acc[1] == pytest.approx(sum(g ** k for k in range(5)), abs=1e-12)  # env 1 never reset
    # clip bound holds for a huge reward
    assert np.abs(nz(np.array([1e9, -1e9]), np.zeros(2))).max() <= 5.0 + 1e-9
    # the running variance equals NumPy's over every accumulator value it was shown
    nz2 = ReturnNormalizer(1, g)
    seen, a = [], 0.0
    for r in np.random.default_rng(1).normal(size=200):
        nz2(np.array([r]), np.zeros(1))
        a = a * g + r
        seen.append(a)
    assert abs(nz2.rms.var - np.var(seen)) < 1e-3


def test_normalize_returns_is_off_by_default_and_trains(tmp_path):
    assert PPOConfig().normalize_returns is False
    cfg = PPOConfig(total_steps=4096, seed=0, normalize_returns=True, eval_every=1000)
    spec = sample_train(np.random.default_rng(0))
    res = train(FFPolicy(), cfg, SamplerSource(lambda rng: spec, 0), EnvConfig(), [spec], str(tmp_path))
    assert np.isfinite(res.best_val)


def _tiny_source(spec_fn):
    class S(SamplerSource):
        pass
    return S(spec_fn, 0)


def test_tiny_solvable_environment_learns_the_right_answer(tmp_path):
    """One fixed scenario with a known optimum: the agent starts next to water-less food... the
    optimal return is the full horizon, and a random policy dies far earlier."""
    spec = ScenarioSpec(layout_id=2, wall_pct=0, arena=5, n_food=1, n_water=1, food_delay_min=8, food_delay_max=8,
                        water_delay_min=8, water_delay_max=8, horizon=200)
    from adaptarena.core import plan, verify_plan
    assert plan(build_scenario(spec)).certified
    cfg = PPOConfig(total_steps=8 * 256 * 40, eval_every=10, save_fracs=(), seed=0)
    m = FFPolicy()
    res = train(m, cfg, SamplerSource(lambda rng: spec, 0), EnvConfig(), [spec], str(tmp_path))
    from adaptarena.lab.evaluate import evaluate
    rand = np.mean([r.survival for r in evaluate("random", [spec] * 10, list(range(10)))])
    learned = evaluate(m, [spec], [0])[0].survival
    assert learned > rand + 20, (learned, rand)


def test_same_seed_same_curve(tmp_path):
    def run(d):
        cfg = PPOConfig(total_steps=4 * 64 * 3, n_envs=4, n_steps=64, eval_every=100, save_fracs=(), seed=3, n_minibatches=2)
        torch.manual_seed(3)
        m = FFPolicy()
        specs = [sample_train(np.random.default_rng(i)) for i in range(4)]
        train(m, cfg, SamplerSource(sample_train, 3), EnvConfig(), specs, str(d))
        return open(d / "train_log.csv").read().splitlines()
    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    key = lambda L: [",".join(l.split(",")[:12]) for l in L]  # drop wall-clock columns
    cols = a[0].split(",")
    keep = [i for i, c in enumerate(cols) if c not in ("sps",)]
    pick = lambda L: [[x.split(",")[i] for i in keep] for x in L]
    assert pick(a) == pick(b)


@pytest.mark.parametrize("cls", [LSTMPolicy, SSMPolicy])
def test_recurrent_training_runs(cls, tmp_path):
    cfg = PPOConfig(total_steps=4 * 32 * 2, n_envs=4, n_steps=32, eval_every=100, save_fracs=(), n_minibatches=2, epochs=2)
    m = cls()
    specs = [sample_train(np.random.default_rng(i)) for i in range(3)]
    res = train(m, cfg, SamplerSource(sample_train, 0), EnvConfig(), specs, str(tmp_path))
    assert res.steps == 4 * 32 * 2
