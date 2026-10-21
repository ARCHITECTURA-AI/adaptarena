# Model card: ppo_lstm_retuned_seed0

- **Policy**: lstm (424,519 parameters), own PPO implementation, rung 7 of the ladder. Added in v2.1.
- **Training**: retuned LSTM config {'lr': 0.002, 'ent_coef': 0.01, 'seq_len': 256}, seed 0, 2,000,000 transitions (4 seeds trained)
- **Weights**: `model.pt` (PyTorch `state_dict`), SHA-256 `31e1e109c956fdd6cbadfa8636888f1d79dbe2b441a15a172304f79eb7062750`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Result**: validation-suite (not used for selection) aggregate IQM 0.121 [0.088, 0.165] vs feedforward arm B 0.351
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; probes show behaviour, not reasoning; it stays far below the feedforward policy and near random level
- **Analysis plan**: `analysis_plan_v2_1.yaml` (SHA-256 `877878f227b51908`); tier label: exploratory.
