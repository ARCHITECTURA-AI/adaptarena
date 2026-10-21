# Model card: ppo_ff_armA_seed0

- **Policy**: ppo_ff (404,615 parameters), own PPO implementation, rung 5 of the ladder.
- **Training**: arm A (Specialist), seed 0, 1,998,848 transitions, checkpoint = highest mean validation survival (update 625).
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `e780c166709132a05335a883e6e87b5c51eeed9989225a79f4c02adfb6b6e31e`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Probes**: they show behavior and input sensitivity, not reasoning.
- **Intended use**: reproducing the published benchmark and exploring the site's probes. Frozen policy: it does not learn at inference time.
- **Result on the locked test suite**: ladder-normalised aggregate IQM for this arm 0.010 [0.003, 0.019]. IQMs are arm-level (over all seeds of the arm), not this seed's score.
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; behaviour outside the stated distributions is not characterised; probes show behaviour, not reasoning.
- **Config hash**: `ce77e6b640217c28`; analysis plan `936d3e3909573fa9`.
