import numpy as np
import pytest

import adaptarena.core.params as P
from adaptarena.core import BatchEnv, EnvConfig, ScenarioSpec, SurvivalEnv, build_scenario, plan, verify_plan
from adaptarena.core.distributions import CONDITIONS, sample_condition, sample_train
from adaptarena.core.baselines import MemoryScript, ReactiveScript, RandomPolicy
from adaptarena.core.scenario import with_intervention


def _specs():
    rng = np.random.default_rng(7)
    out = [sample_train(rng) for _ in range(4)]
    base = out[0]
    out += [
        with_intervention(base, "cost", 50, 220),
        with_intervention(base, "cold", 30, 60),
        with_intervention(base, "drought", 60, 2),
        with_intervention(base, "blight", 60, 2),
        with_intervention(base, "reloc", 80),
        with_intervention(with_intervention(base, "hazard", 40, 10), "reloc", 100),
        base.with_(horizon=120),
        base.with_(n_food=1, n_water=1),
    ]
    out += [sample_condition(c, rng, "val") for c in CONDITIONS]
    return out


def test_scenario_is_deterministic_and_serialisable():
    s = _specs()[5]
    a, b = build_scenario(s), build_scenario(ScenarioSpec.from_json(s.to_json()))
    assert s.sid == ScenarioSpec.from_json(s.to_json()).sid
    for f in ("terrain", "food_pos", "water_pos", "water_pos2", "food_delay", "cold", "cost_pct", "hazard"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


def test_layout_is_connected_and_sites_on_floor():
    for s in _specs():
        sc = build_scenario(s)
        free = sc.terrain == 0
        for p in list(sc.food_pos) + list(sc.water_pos) + list(sc.water_pos2):
            assert free[p[0], p[1]]
        assert free[sc.start[0], sc.start[1]]
        assert len({tuple(p) for p in list(sc.food_pos) + list(sc.water_pos)}) == sc.nf + sc.nw


def test_golden_state_hash():
    """Pins the bit-exact state across machines and Python/NumPy versions."""
    env = SurvivalEnv()
    env.reset(ScenarioSpec(layout_id=3, timing_seed=11, weather_seed=5, cold_pct=10))
    rng = np.random.default_rng(0)
    for _ in range(60):
        env.step(int(rng.integers(0, 6)))
        if env.done:
            break
    assert env.state_hash() == GOLDEN


GOLDEN = "5c15678334efac1242c6af53533b133582bdea5e2558f0d71177151d3450c058"


@pytest.mark.parametrize("reward", ["sparse", "naive", "potential"])
@pytest.mark.parametrize("vision", [9, 5, 3])
def test_batch_matches_reference(reward, vision):
    cfg = EnvConfig(vision=vision, reward=reward, prev_action=vision != 5, cost_stat=vision != 3)
    specs = _specs()
    n = len(specs)
    refs = [SurvivalEnv(cfg) for _ in range(n)]
    batch = BatchEnv(n, cfg)
    for i, s in enumerate(specs):
        refs[i].reset(s)
        batch.reset_env(i, s)
    rng = np.random.default_rng(3)
    for step in range(400):
        obs_b = batch.observe()
        for i, e in enumerate(refs):
            if e.done:
                continue
            o = e.observation()
            assert np.array_equal(o["grid"], obs_b["grid"][i]), (step, i, specs[i])
            assert np.allclose(o["stats"], obs_b["stats"][i]), (step, i)
        acts = rng.integers(0, 6, size=n)
        live = np.array([not e.done for e in refs])
        if not live.any():
            break
        # only advance live envs: freeze finished ones by resetting their batch row
        r_b, term_b, trunc_b = batch.step(np.where(live, acts, P.WAIT))
        for i, e in enumerate(refs):
            if not live[i]:
                continue
            _, r, te, tr, _ = e.step(int(acts[i]))
            assert abs(r - r_b[i]) < 1e-5, (step, i, r, r_b[i])
            assert te == term_b[i] and tr == trunc_b[i]
            assert e.state_hash() == batch.state_hashes()[i], (step, i)


def test_edge_cases():
    spec = ScenarioSpec(layout_id=1, wall_pct=0, n_food=1, n_water=1)
    env = SurvivalEnv()
    env.reset(spec)
    sc = env.scenario
    e0, h0 = env.energy, env.hydration
    # invalid interact: base drain only
    _, _, _, _, info = env.step(P.INTERACT)
    assert not info["valid"] and info["drain"] == P.BASE_ENERGY_DRAIN
    # a move attempt into the map edge: invalid, but pays the movement surcharge
    env2 = SurvivalEnv(); env2.reset(spec.with_(arena=15))
    env2.r, env2.c = 0, 0
    _, _, _, _, info = env2.step(P.UP)
    assert not info["valid"] and info["drain"] == P.BASE_ENERGY_DRAIN + P.MOVE_EXTRA_ENERGY
    assert env2.r == 0
    # stats vector on reset: indices 3..10 are zero
    o = SurvivalEnv().reset(spec)
    assert np.all(o["stats"][3:] == 0)


def test_death_flags_and_primary_order():
    env = SurvivalEnv()
    env.reset(ScenarioSpec(layout_id=2, wall_pct=0))
    env.energy, env.hydration = 10, 10
    _, r, term, trunc, info = env.step(P.WAIT)
    assert term and r == 0.0 and info["flags"] == P.CAUSE_DEHYDRATION | P.CAUSE_STARVATION
    assert info["primary_cause"] == "dehydration"


def test_integer_state():
    env = SurvivalEnv()
    env.reset(sample_train(np.random.default_rng(1)))
    for _ in range(50):
        env.step(2)
        if env.done:
            break
    for v in (env.energy, env.hydration, env.t, env.last_drain):
        assert isinstance(v, (int, np.integer))


def test_observation_has_no_information_leak():
    """Hidden quantities (respawn delays, future weather, unseen sites) must not change the observation."""
    base = ScenarioSpec(layout_id=5, timing_seed=1, weather_seed=1, n_food=4, n_water=4, wall_pct=15)
    a, b = build_scenario(base), build_scenario(base)
    b.food_delay[:] = b.food_delay + 17
    b.water_delay[:] = b.water_delay + 9
    horizon_cut = 100
    b.cold[horizon_cut:] = 1 - b.cold[horizon_cut:]
    ea, eb = SurvivalEnv(), SurvivalEnv()
    ea.reset(a); eb.reset(b)
    rng = np.random.default_rng(2)
    for t in range(horizon_cut - 1):
        act = int(rng.integers(0, 4))  # moving only: no consumption, so respawn delays stay unobserved
        oa, *_ = ea.step(act)
        ob, *_ = eb.step(act)
        if ea.done:
            break
        assert np.array_equal(oa["grid"], ob["grid"])
        assert np.allclose(oa["stats"], ob["stats"])


def test_unseen_sites_are_not_in_the_observation():
    sc = build_scenario(ScenarioSpec(layout_id=5, wall_pct=10, n_food=5, n_water=5))
    env = SurvivalEnv()
    o = env.reset(sc)
    seen_food = int(o["grid"][P.CH_FOOD].sum() + o["grid"][P.CH_FOOD_DEPLETED].sum())
    in_view = sum(1 for p in sc.food_pos if abs(p[0] - env.r) <= 4 and abs(p[1] - env.c) <= 4)
    assert seen_food == in_view


def test_reduced_vision_has_explicit_visibility_channel():
    env = SurvivalEnv(EnvConfig(vision=3))
    o = env.reset(ScenarioSpec(layout_id=4))
    assert o["grid"][P.CH_VISIBLE].sum() == 9
    assert o["grid"][:7][:, o["grid"][P.CH_VISIBLE] == 0].sum() == 0


def test_potential_shaping_telescopes():
    cfg = EnvConfig(reward="potential", gamma=1.0)
    env = SurvivalEnv(cfg)
    env.reset(ScenarioSpec(layout_id=6))
    total, n = 0.0, 0
    while not env.done and n < 200:
        _, r, *_ = env.step(P.WAIT)
        total += r; n += 1
    # with gamma=1 the shaping sums to phi(end)-phi(start), bounded by the potential scale
    assert abs(total - env.survived) <= cfg.potential_scale + 1e-6


# ---------------------------------------------------------------------------- planner
def test_planner_certificates_verify_by_replay():
    rng = np.random.default_rng(11)
    certified = 0
    for _ in range(12):
        sc = build_scenario(sample_train(rng))
        res = plan(sc)
        assert verify_plan(sc, res.actions) == res.survival
        certified += res.certified
    assert certified >= 8


def _brute_force(sc):
    """Exact best survival by exhaustive search over primitive actions on a tiny scenario."""
    import functools
    H = sc.horizon

    @functools.lru_cache(maxsize=None)
    def go(t, r, c, e, h, fr, wr):
        if t >= H:
            return 0
        best = 0
        for a in range(6):
            env = SurvivalEnv()
            env.reset(sc)
            env.t, env.r, env.c, env.energy, env.hydration = t, r, c, e, h
            env.food_ready = np.array(fr, dtype=np.int32)
            env.water_ready = np.array(wr, dtype=np.int32)
            env.alive = True
            env.survived = 0
            env.step(a)
            if not env.alive:
                continue
            best = max(best, 1 + go(env.t, env.r, env.c, env.energy, env.hydration,
                                    tuple(int(x) for x in env.food_ready), tuple(int(x) for x in env.water_ready)))
            if best >= H - t:
                break
        return best

    return go(0, sc.start[0], sc.start[1], sc.start_energy, sc.start_hydration,
              tuple([0] * sc.nf), tuple([0] * sc.nw))


@pytest.mark.slow
def test_planner_matches_brute_force_on_tiny_maps(harsh_physics):
    checked = solvable = 0
    for seed in range(8):
        spec = ScenarioSpec(layout_id=seed, timing_seed=seed, weather_seed=seed, arena=4, wall_pct=10,
                            n_food=2, n_water=2, food_delay_min=5, food_delay_max=9,
                            water_delay_min=5, water_delay_max=9, cold_pct=25, cold_len=6, horizon=24)
        sc = build_scenario(spec)
        res = plan(sc, max_nodes=100000)
        assert not res.capped
        assert verify_plan(sc, res.actions) == res.survival
        bf = _brute_force(sc)
        assert res.survival == bf, (seed, res.survival, bf)
        checked += 1
        solvable += res.certified
    assert checked == 8 and 0 < solvable < 8  # the sample contains both solvable and unsolvable maps


def test_baselines_run_and_are_ordered():
    rng = np.random.default_rng(5)
    specs = [sample_condition("unseen_layout", rng, "val") for _ in range(6)]
    mean = {}
    for pol in (RandomPolicy(), ReactiveScript(), MemoryScript()):
        scores = []
        for i, s in enumerate(specs):
            env = SurvivalEnv(); obs = env.reset(s); pol.reset(i)
            while not env.done:
                obs, *_ = env.step(pol.act(obs))
            scores.append(env.survived)
        mean[pol.name] = np.mean(scores)
    assert mean["random"] < mean["reactive_script"] < mean["memory_script"] + 1


def test_gymnasium_checker_and_make():
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    import adaptarena  # noqa: F401
    env = gym.make("AdaptArena-Survival-v1")
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs["grid"].shape == (8, 9, 9) and obs["stats"].shape == (11,)
    obs2, _ = env.reset(seed=0)
    assert np.array_equal(obs["grid"], obs2["grid"])
