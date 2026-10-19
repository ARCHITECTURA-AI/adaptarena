# Methodology

Environment `AdaptArena-Survival-v1`: integer-state survival world (15x15, 6 actions, reward 1 per living step, 500-step horizon).
Seed streams: layout (`layout_id`), resource timing (`timing_seed`), weather (`weather_seed`), action sampling (per-episode action seed), training (`seed`).

## Training
- Algorithm: PPO written from scratch (`src/adaptarena/lab/ppo.py`): GAE (gamma 0.99, lambda 0.95), clip 0.2, entropy 0.003, value coefficient 0.5, gradient clipping 0.5,
  linear LR decay from 0.0003, 8 environments x 256 steps per update, 10 epochs, 4 minibatches, advantage normalisation, reward scaled by 0.01 inside the learner.
- Termination vs truncation: bootstrap from V(final observation) on horizon truncation, never on death.
- Transition budget per run: 20,480, counted across all environments. Seeds per arm: 1.
- Checkpoint rule: highest mean validation survival, earlier checkpoint wins ties.
- Recurrent policies: sequence chunks of 64 steps with hidden-state resets at episode boundaries; 2 sequence minibatches.
- Tuning: one sweep per method (3 learning rates x 3 entropy coefficients x 1 seeds, 10,240 transitions) scored on validation only.

## Evaluation partitions
- training layouts 0..9999; validation layouts 200000+; locked test layouts 100000+; stress 300000+; red team 400000+. Timing and weather seeds are in disjoint ranges.
- Locked test suite: 7 conditions x 4 planner-certified scenarios, deterministic action selection (primary), sampled actions secondary and never mixed in one aggregate.
- The harness opens the locked suite only if the SHA-256 of `analysis_plan.yaml` matches the committed `analysis_plan.sha256`; every access is logged.

## Interventions
Precomputed schedule edits `(name, step, argument)`: `cost` (movement-cost percent from a step), `cold` (forced cold spell), `drought` / `blight` (water / food sites vanish),
`reloc` (water sites move to alternate cells), `hazard` (tiles become deadly). Hazards are off in every benchmark suite and used only in the hazard-spill demo.

## Metrics
- Survival: steps survived within the 500-step horizon (right-censored). Ladder-normalised survival: (policy - random mean) / (planner - random mean) per scenario, random mean over action seeds 0..9.
- Aggregates: IQM over training seeds (episodes averaged inside a seed first), stratified bootstrap CIs (10,000 resamples), optimality gap, probability of improvement, performance profiles.

## Known limitations
See the technical report, section 6.
