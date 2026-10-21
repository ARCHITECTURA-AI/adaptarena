# Model card: ppo_lstm_seed0

- **Policy**: ppo_lstm (424,519 parameters), own PPO implementation, rung 7 of the ladder.
- **Training**: arm B (Uniform randomization), seed 0, 1,998,848 transitions, checkpoint = highest mean validation survival (update 800).
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `530ea298173c22dac7767f31aa653c1cbf4a83d1a617c0baba5b5127a529083c`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Probes**: they show behavior and input sensitivity, not reasoning.
- **Intended use**: reproducing the published benchmark and exploring the site's probes. Frozen policy: it does not learn at inference time.
- **Result on the locked test suite**: ladder-normalised aggregate IQM for this arm 0.316 [0.252, 0.372] (5 seeds, 7 conditions, locked test suite). IQMs are arm-level (over all seeds of the arm), not this seed's score.
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; behaviour outside the stated distributions is not characterised; probes show behaviour, not reasoning.
- **Config hash**: `1ad272242fd6c220`; analysis plan `936d3e3909573fa9`.
