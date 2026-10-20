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
