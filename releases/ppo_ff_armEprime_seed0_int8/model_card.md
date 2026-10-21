# Model card: ppo_ff_armEprime_seed0_int8

- **Policy**: ff (dynamic int8) (404,615 parameters), own PPO implementation, rung 5 of the ladder. Added in v2.1.
- **Training**: post-training dynamic int8 quantization of `ppo_ff_armEprime_seed0` (Linear and LSTMCell weights int8; activations and recurrent state float32)
- **Weights**: `model_int8.pt` (PyTorch `state_dict`, quantized), SHA-256 `32366683c4fa849e6a6f7fe429b7a70538b367bf10e57d56f6c698e58b173671`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Result**: retention over float32 on suite_val_s1: 0.951 [0.906, 1.001] (float32 0.351, int8 0.334); action agreement 0.979; size 1623 kB to 415 kB
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; probes show behaviour, not reasoning; measured on Intel(R) Xeon(R) Processor @ 2.80GHz only; no edge-device or energy claim
- **Analysis plan**: `analysis_plan_v2_1.yaml` (SHA-256 `877878f227b51908`); tier label: exploratory.
