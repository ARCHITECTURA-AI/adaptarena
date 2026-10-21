# Changelog

## 2.1.0

Plan: `analysis_plan_v2_1.yaml` (own SHA-256; the v2 plan, its amendments, its locked suite and its access log are unchanged; the simulator hashes are unchanged). Every v2.1 result carries a tier label: confirmatory, exploratory or diagnostic.

- **Fresh locked test suite** `suite_test_v2_1.json`, generated through the access harness after the plan commit was pushed, with no scenario id or layout shared with the v2 suites. Existing arms and the LSTM were re-scored on it (yardstick only).
- **Audit**: the specialist's pooled 0.010 is a finding (0.81 on its own layout); ablation tables were scored on the suite that chose the checkpoints, and two findings change on `suite_val_s1.json`.
- **Training speed**: profile committed; no change reached the plan's 1.2x threshold (best 1.17x), so none was adopted. `fast_rollout`, `normalize_returns` and `profile` are PPO flags that default to v2 behaviour.
- **Memory diagnostics**: imitation ceiling with DAgger-lite, `obs-v9-map` (hand-fed map) wrapper, LSTM retune (two stages plus equal-effort feedforward), predictions P1 partial, P2 miss, P3 hit.
- **Mutation arms E, E-prime, F** and a budget sweep (B and D at 8M, and 20M if present in `results/full/v2_1/scale/sweep.json`). Pre-declared outcome for E: see the README.
- **Gamma re-run** with return normalization; **int8** post-training quantization for the feedforward and LSTM policies. **SSM** (rung 8) was cut for compute and is not claimed.
- Reporting: report Part II, a README results block generated from `results/full/`, ledger columns `plan` and `tier`, site additions (imitation bars, ledger columns), new releases with model cards.
- Packaging: `requirements-lock-cpu.txt`, NOTICE, CI matrix for Python 3.10 to 3.13.

## 2.0.0

The original release: environment, planner, own PPO, headline experiment, statistics, site, report, video, CI.
