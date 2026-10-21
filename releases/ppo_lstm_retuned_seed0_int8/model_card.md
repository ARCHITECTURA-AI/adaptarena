# Model card: ppo_lstm_retuned_seed0_int8

- **Policy**: lstm (dynamic int8) (424,519 parameters), own PPO implementation, rung 7 of the ladder. Added in v2.1.
- **Training**: post-training dynamic int8 quantization of `ppo_lstm_retuned_seed0` (Linear and LSTMCell weights int8; activations and recurrent state float32)
- **Weights**: `model_int8.pt` (PyTorch `state_dict`, quantized), SHA-256 `a70a4c71dc2a9d0974d52b2387f74b27dfe9247621b1f89f9399d062b1faf706`.
- **Environment**: `AdaptArena-Survival-v1`, observation obs-v9.
- **Result**: retention over float32 on suite_val_s1: 1.119 [0.949, 1.305] (float32 0.121, int8 0.136); action agreement 0.933; size 1702 kB to 436 kB
- **Seed**: seed 0 was fixed in advance as the released checkpoint, not picked for its score.
- **Limitations**: trained and evaluated on one small custom environment; probes show behaviour, not reasoning; measured on Intel(R) Xeon(R) Processor @ 2.80GHz only; no edge-device or energy claim
- **Analysis plan**: `analysis_plan_v2_1.yaml` (SHA-256 `877878f227b51908`); tier label: exploratory.
