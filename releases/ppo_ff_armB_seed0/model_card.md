# Model card: ppo_ff_armB_seed0

- **Policy**: ppo_ff (404,615 parameters), own PPO implementation, rung 5 of the ladder.
- **Training**: arm B (Uniform randomization), seed 0, 1,998,848 transitions, checkpoint = highest mean validation survival (update 750).
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `6c47da4ff48ae580b4ae96a1f8f7a54fc47a0bcf376c117bae38d1bc8e97f817`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Intended use**: reproducing the published benchmark and exploring the site's probes. Frozen policy: it does not learn at inference time.
- **Result on the locked test suite**: ladder-normalised aggregate IQM for this arm 0.387 [0.374, 0.400].
- **Limitations**: trained and evaluated on one small custom environment; behaviour outside the stated distributions is not characterised; probes show behaviour, not reasoning.
- **Config hash**: `ce77e6b640217c28`; analysis plan `936d3e3909573fa9`.
