import json
import os

import numpy as np
import pytest

import adaptarena.core.params as P
from adaptarena.core import ScenarioSpec, SurvivalEnv, build_scenario, plan
from adaptarena.core.distributions import build_suite, sample_condition
from adaptarena.lab import plan as planmod
from adaptarena.lab.evaluate import evaluate, ladder_normalise, survival_array
from adaptarena.lab.replays import record_replay, replay_from_actions, verify_replay
from adaptarena.lab.redteam import mutate, sample_space


def test_replay_roundtrip_and_hash_verification():
    spec = sample_condition("unseen_layout", np.random.default_rng(0), "val")
    rep = record_replay(spec, "memory_script")
    assert verify_replay(rep)
    again = replay_from_actions(spec, rep["actions"])
    assert again["final_hash"] == rep["final_hash"] and again["frames"] == rep["frames"]
    rep["actions"][5] = (rep["actions"][5] + 1) % 6  # tampering must be detected
    assert not verify_replay(rep)


def test_evaluate_is_deterministic_and_records_mode():
    specs = [sample_condition("familiar", np.random.default_rng(i), "val") for i in range(3)]
    a = evaluate("reactive_script", specs, [0, 1, 2])
    b = evaluate("reactive_script", specs, [0, 1, 2])
    assert [r.final_hash for r in a] == [r.final_hash for r in b]
    assert all(r.mode == "scripted" for r in a)


def test_batch_evaluation_matches_scalar_reference_for_scripts():
    spec = sample_condition("cold_snap", np.random.default_rng(3), "val")
    batch = evaluate("memory_script", [spec], [0])[0]
    rep = record_replay(spec, "memory_script", action_seed=0)
    assert batch.survival == rep["survival"] and batch.final_hash == rep["final_hash"]


def test_ladder_normalisation_excludes_zero_denominator():
    score, valid = ladder_normalise(np.array([100.0, 300.0, 500.0]), np.array([100.0, 100.0, 500.0]),
                                    np.array([500.0, 500.0, 500.0]))
    assert list(valid) == [True, True, False]
    assert score[0] == 0.0 and score[1] == 0.5


def test_suites_are_certified_and_disjoint_from_training():
    specs, meta = build_suite("val", "combined_shift", 3, seed=5)
    assert len(specs) == 3 and meta["n"] == 3
    for s in specs:
        assert plan(build_scenario(s)).certified
        assert s.layout_id >= 200_000  # never a training layout


def test_plan_lock_refuses_changed_plan(tmp_path):
    p, h = tmp_path / "plan.yaml", tmp_path / "plan.sha256"
    p.write_text("a: 1\n")
    planmod.lock_plan(str(p), str(h))
    assert planmod.verify_plan(str(p), str(h), require_git=False)["plan_hash"] == planmod.file_hash(str(p))
    p.write_text("a: 2\n")
    with pytest.raises(planmod.PlanError):
        planmod.verify_plan(str(p), str(h), require_git=False)
    with pytest.raises(planmod.PlanError):
        planmod.verify_plan(str(p), str(tmp_path / "missing"), require_git=False)


def test_repo_plan_hash_matches_file():
    assert planmod.file_hash(planmod.PLAN) == open(planmod.HASH).read().strip()


def test_redteam_space_and_mutations_stay_valid():
    rng = np.random.default_rng(0)
    s = sample_space(rng)
    for _ in range(30):
        s = mutate(s, rng)
        assert 1 <= s.n_food <= 5 and 1 <= s.n_water <= 5 and 100 <= s.move_cost_pct <= 300 and 5 <= s.wall_pct <= 30
        build_scenario(s)


def test_point_of_no_return_on_a_doomed_run():
    from adaptarena.lab.museum import point_of_no_return
    spec = sample_condition("familiar", np.random.default_rng(1), "val")
    sc = build_scenario(spec)
    actions = [P.WAIT] * 400  # waiting forever dies; the planner could still save the run early on
    env = SurvivalEnv(); env.reset(sc)
    n = 0
    while not env.done and n < len(actions):
        env.step(actions[n]); n += 1
    pnr = point_of_no_return(sc, actions[:n])
    assert pnr["step"] is not None and 0 <= pnr["step"] < n


def test_readme_relative_links_resolve():
    import re
    text = open(os.path.join(planmod.REPO, "README.md")).read()
    for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)\s]+)\)", text):
        path = target.split("#")[0]
        assert os.path.exists(os.path.join(planmod.REPO, path)), f"README link does not resolve: {target}"
    for needed in ("NOTICE", "LICENSE", "analysis_plan.yaml", "analysis_plan.sha256"):
        assert os.path.exists(os.path.join(planmod.REPO, needed))


def test_harness_refuses_suite_under_wrong_plan_hash(tmp_path):
    """The locked-suite gate: a plan that no longer matches its hash can neither open nor generate a suite."""
    from adaptarena.lab.suites import open_locked_test_suite
    plan_file = tmp_path / "analysis_plan_v2_1.yaml"
    plan_file.write_text("frozen_components: {}\n")
    h = planmod.lock_plan(str(plan_file))
    assert (tmp_path / "analysis_plan_v2_1.sha256").read_text().strip() == h
    (tmp_path / "suite_test_v2_1.json").write_text("{}")  # a suite already exists on disk
    plan_file.write_text("frozen_components: {}\nedited: true\n")
    with pytest.raises(planmod.PlanError):
        open_locked_test_suite(str(tmp_path), "test", plan_path=str(plan_file), name="suite_test_v2_1.json")
    assert not (tmp_path / "test_access_log.jsonl").exists()  # a refused open is not logged as an access


def test_cli_lock_plan_accepts_a_path(tmp_path):
    p = tmp_path / "analysis_plan_v2_1.yaml"
    p.write_text("a: 1\n")
    assert planmod.hash_path_for(str(p)).endswith("analysis_plan_v2_1.sha256")


def test_repo_v2_1_plan_hash_matches_file():
    assert planmod.file_hash(planmod.PLAN_V2_1) == open(planmod.HASH_V2_1).read().strip()


# ---------------------------------------------------------------------------------------- v2.1 Phase 4
def _script_and_tracker_agree(spec, seed=0):
    """Run the memory script; feed the same observations to a MapTracker; compare maps after every step."""
    from adaptarena.core.baselines import MemoryScript
    from adaptarena.lab.memory import MapTracker
    sc = build_scenario(spec)
    env = SurvivalEnv()
    obs = env.reset(sc)
    script, trk = MemoryScript(), MapTracker()
    script.reset(seed)
    trk.reset(seed)
    while not env.done:
        trk.update(obs)
        a = script.act(obs)
        assert (script.y, script.x) == (trk.y, trk.x)
        assert script.floor == trk.floor
        assert {c: r["kind"] for c, r in script.sites.items()} == {c: r["kind"] for c, r in trk.sites.items()}
        for c, r in script.sites.items():  # ready flags may differ only at the cell where the script just chose INTERACT
            if c != (script.y, script.x) or a != P.INTERACT:
                assert r["ready"] == trk.sites[c]["ready"], (c, r, trk.sites[c])
        obs = env.step(a)[0]


def test_map_tracker_reproduces_memory_script_map_on_recorded_rollouts():
    rng = np.random.default_rng(3)
    for cond in ("unseen_layout", "scarce_food", "higher_move_cost"):
        _script_and_tracker_agree(sample_condition(cond, rng, "val"), seed=1)


def test_map_batch_env_appends_channels_and_is_idempotent():
    from adaptarena.core import BatchEnv
    from adaptarena.lab.memory import MAP_DIM, MapBatchEnv
    rng = np.random.default_rng(0)
    specs = [sample_condition("unseen_layout", rng, "val") for _ in range(3)]
    plain, mapped = BatchEnv(3), MapBatchEnv(3)
    plain.reset_all(specs)
    mapped.reset_all(specs)
    for step in range(6):
        a = plain.flat_obs()
        b = mapped.flat_obs()
        assert b.shape == (3, P.OBS_DIM + MAP_DIM)
        assert np.array_equal(a, b[:, :P.OBS_DIM])
        assert np.array_equal(b, mapped.flat_obs())  # second call on the same state changes nothing
        acts = rng.integers(0, 6, 3)
        plain.step(acts)
        mapped.step(acts)
    assert mapped.flat_obs()[:, P.OBS_DIM + 2 * 81:].sum() > 0  # known floor is filled in


def test_models_accept_wider_observation_and_load_model_infers_width(tmp_path):
    import torch
    from adaptarena.lab.experiments import load_model
    from adaptarena.lab.models import make_model
    for kind in ("ff", "lstm", "ssm"):
        m = make_model(kind, obs_dim=902)
        d = tmp_path / kind
        d.mkdir()
        torch.save(m.state_dict(), d / "model_best.pt")
        assert load_model(str(d), kind).state_dict().keys() == m.state_dict().keys()


def test_shadow_labeler_agrees_with_the_script_on_its_own_rollouts():
    """On the script's own observation stream the shadow labeller must say what the script did (to within the
    one-step seen_dep lag), otherwise DAgger labels would not be the script's."""
    from adaptarena.lab import imitation as I
    rng = np.random.default_rng(5)
    specs = [sample_condition("unseen_layout", rng, "val") for _ in range(3)]
    ep = I._roll(specs, [0, 1, 2])
    match = total = 0
    for i in range(len(ep)):
        lab = I.ShadowLabeler()
        lab.reset(i)
        for t in range(len(ep.act[i])):
            obs = {"grid": ep.grid[i][t].astype(np.float32).reshape(P.N_CHANNELS, P.VIEW, P.VIEW), "stats": ep.stats[i][t]}
            match += int(lab.label(obs) == ep.act[i][t])
            total += 1
    assert match / total > 0.98, match / total


def test_imitation_smoke_collect_train_score(tmp_path):
    from adaptarena.lab import imitation as I
    ep = I.collect(4, seed=0, workers=1)
    assert len(ep) == 4 and ep.n_samples > 0
    I.save_episodes(ep, str(tmp_path / "d.npz"))
    back = I.load_episodes(str(tmp_path / "d.npz"))
    assert back.n_samples == ep.n_samples and np.array_equal(back.act[0], ep.act[0])
    for kind, w in (("ff", 64), ("lstm", 64)):
        m = I.train_clone(kind, ep, 1, 0, window=w)
        assert 0.0 <= I.action_agreement(m, kind, ep) <= 1.0
