# Model card: ppo_ff_armEprime_seed0

- **Policy**: ff (404,615 parameters), own PPO implementation, rung 5 of the ladder. Added in v2.1.
- **Training**: mutation arm Eprime (the best of E, E-prime, F by aggregate IQM), seed 0, 2,000,000 transitions
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `b0de8695a60d441c16dba4ad2c6b1909e10b6c312e1015a38b5156f9682ce70c`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Result**: fresh-suite ladder-normalised aggregate IQM for this arm 0.309 [0.292, 0.327] (10 seeds, 7 conditions); E minus B -0.090: not supported (reversed)
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; probes show behaviour, not reasoning; it did not beat uniform randomization (arm B), so it is released as a record, not as a better policy
- **Analysis plan**: `analysis_plan_v2_1.yaml` (SHA-256 `877878f227b51908`); tier label: confirmatory (E vs B), exploratory otherwise.
