# Model card: ppo_ff_armD_seed0

- **Policy**: ppo_ff (404,615 parameters), own PPO implementation, rung 5 of the ladder.
- **Training**: arm D (Replay by planner regret), seed 0, 1,998,848 transitions, checkpoint = highest mean validation survival (update 725).
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `d6b78a4212042134028540bca15988280bb692eb4ef971fcfc8b4411d382e6d3`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Probes**: they show behavior and input sensitivity, not reasoning.
- **Intended use**: reproducing the published benchmark and exploring the site's probes. Frozen policy: it does not learn at inference time.
- **Result on the locked test suite**: ladder-normalised aggregate IQM for this arm 0.315 [0.299, 0.331]. IQMs are arm-level (over all seeds of the arm), not this seed's score.
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; behaviour outside the stated distributions is not characterised; probes show behaviour, not reasoning.
- **Config hash**: `ce77e6b640217c28`; analysis plan `936d3e3909573fa9`.
