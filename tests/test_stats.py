import numpy as np
from adaptarena.lab import stats as S


def test_iqm_ignores_outliers():
    x = [0.0, 1, 1, 1, 1, 1, 1, 1, 1, 100]
    assert S.iqm(x) == 1.0


def test_bootstrap_ci_covers_truth_and_scales_with_n():
    rng = np.random.default_rng(0)
    cover = 0
    for rep in range(60):
        sc = {c: rng.normal(0.5, 0.1, size=10) for c in "abc"}
        r = S.iqm_with_ci(sc, seed=rep, B=400)
        cover += r["lo"] <= 0.5 <= r["hi"]
    assert cover >= 48  # nominal 95%, small-sample IQM bootstrap is slightly liberal
    small = S.iqm_with_ci({"a": rng.normal(0.5, 0.1, 5)}, B=1000)
    big = S.iqm_with_ci({"a": rng.normal(0.5, 0.1, 50)}, B=1000)
    assert (big["hi"] - big["lo"]) < (small["hi"] - small["lo"])


def test_probability_of_improvement():
    x, y = np.array([1.0, 2, 3]), np.array([0.0, 0.5, 4])
    assert abs(S.prob_improvement(x, y) - 6 / 9) < 1e-9
    assert S.prob_improvement(x, x) == 0.5


def test_difference_ci_excludes_zero_for_clear_gap():
    rng = np.random.default_rng(1)
    a = {c: rng.normal(0.8, 0.02, 10) for c in "ab"}
    b = {c: rng.normal(0.4, 0.02, 10) for c in "ab"}
    d = S.iqm_difference_with_ci(a, b, B=2000)
    assert d["lo"] > 0
    p = S.prob_improvement_with_ci(a, b, B=2000)
    assert p["lo"] > 0.5


def test_kaplan_meier_matches_empirical_without_early_censoring():
    times = np.array([10, 20, 20, 500, 500])
    died = times < 500
    t, Sv = S.kaplan_meier(times, died)
    assert Sv[10] == 1.0 and abs(Sv[11] - 0.8) < 1e-9 and abs(Sv[21] - 0.4) < 1e-9 and abs(Sv[500] - 0.4) < 1e-9
    assert S.censored_fraction(times) == 0.4


def test_breaking_point():
    assert S.breaking_point([1, 2, 3, 4], [0.9, 0.7, 0.4, 0.1], 0.5) == 3
    assert S.breaking_point([1, 2], [0.9, 0.8], 0.5) is None
