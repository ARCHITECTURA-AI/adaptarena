# AdaptArena v2

> **Does training where the policy fails most produce agents that survive unseen shifts better than uniform randomization, at equal compute?**
> A hypothesis, tested in a small survival world where a reference planner makes regret cheap. If it loses, the report says so.

![Headline figure](docs/headline.png)

<!-- RESULTS:START -->
## Result (full tier: 10 seeds per arm, 2M transitions each, locked test suite of 7 conditions x 50 certified scenarios)

**Planner-regret replay did not beat uniform randomization; it lost.** The pre-declared outcome rule fired as `D_loses_to_B`.

| Arm | Aggregate ladder-normalised IQM [95% CI] |
| --- | --- |
| A: specialist | 0.010 [0.003, 0.019] |
| B: uniform randomization | 0.387 [0.374, 0.400] |
| C: replay by value loss | 0.337 [0.324, 0.351] |
| D: replay by planner regret | 0.315 [0.299, 0.331] |

- D vs B (pre-declared): IQM difference -0.072 [-0.093, -0.051], P(improvement) 0.193 [0.121, 0.271]: **not supported**.
- D vs C (pre-declared): -0.022 [-0.043, -0.001]: **not supported**.
- Why (exploratory diagnosis in the report): every training scenario is planner-solvable, so regret collapses to the policy's shortfall; ~48% of replay-arm episodes are repeats (half as many distinct scenarios at equal budget); scenario parameters explain almost none of what is hard (CV R² 0.02); the score is noisy (rank correlation 0.60 across sampled rollouts).
- Parity gate with Stable-Baselines3 passed: own PPO IQM survival 216.5, SB3 215.6, curves overlap at 100% of points.
- Memory track: 1 of 5 locked predictions hit; the LSTM did not beat the feedforward policy where memory was predicted to help (5 seeds vs 10, hyper-parameters borrowed from the feedforward sweep by a dated amendment).
- Ablations (exploratory): gamma 0.9, 3x3 vision and removing the previous action hurt; the energy-cost stat, naive/potential shaping, gamma 0.999 and time-out-as-death had no measurable effect.
- Scripted baselines are strong: the memory script reaches 0.7 to 1.0 of the planner on every condition, at or above every learned arm. The baseline ladder is on the site and in the report.

Limits: one small custom environment; only this benchmark; a stated hypothesis that lost. Nothing here says regret-guided training is worse in general. The SSM policy was not run (cut order: stretch lane). Hardware numbers are from one named 4-core Xeon.
<!-- RESULTS:END -->

AdaptArena v2 exists to show that its builder can design an RL environment, implement the learning algorithm, and report results that hold up under scrutiny.
**The report is the product** (`report/report.pdf`, also on the [site](site/)); the site, the video and the package exist to get you to it and let you trust it.

## What a reviewer finds

- **An environment with integer state, bit-exact replays and a reference planner** that certifies every scenario solvable (`src/adaptarena/core/`). Energy and hydration are integer hundredths, cost multipliers are integer percents, and the state hash is SHA-256 over a fixed byte layout; a golden hash is checked on Linux, macOS and Windows in CI. The planner's plans are verified by replay and cross-checked against exhaustive search on tiny maps.
- **PPO written from scratch, unit tested and matched against Stable-Baselines3** (`src/adaptarena/lab/ppo.py`, `tests/test_ppo.py`, the parity gate in the report).
- **A baseline ladder from random to planner** (uniform random, reactive script, memory script, human, own PPO, SB3 PPO, PPO+LSTM, small SSM, reference planner), so every number has context.
- **The headline experiment**: four arms (specialist, uniform randomization, replay by value loss, replay by planner regret), 10 training seeds each, IQM with stratified bootstrap CIs, a locked test suite, and an analysis plan that was hashed and committed before that suite existed (`analysis_plan.yaml`, `analysis_plan.sha256`).
- **Probes that show what each policy learned**, each paired with what it does not show; stress curves with Kaplan-Meier survival and breaking points; a red-team search that finds verified failures.
- **A static site** where anyone can play the same fog-of-war game (the Python core runs in the browser through Pyodide) and rank themselves against the agents.

## Install and use

```bash
pip install adaptarena            # environment, planner, baselines (NumPy + Gymnasium only)
pip install "adaptarena[lab]"     # + PyTorch, training code, statistics
```

```python
import gymnasium as gym, adaptarena
env = gym.make("AdaptArena-Survival-v1")
obs, info = env.reset(seed=0)        # obs = {"grid": (8, 9, 9), "stats": (11,)}
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

```python
from adaptarena import ScenarioSpec, build_scenario, plan, verify_plan
sc = build_scenario(ScenarioSpec(layout_id=7, timing_seed=3, n_water=2))
res = plan(sc)                                    # full-information search
print(res.certified, verify_plan(sc, res.actions))  # certificate, replayed in the environment
```

## Reproduce

```bash
make reproduce TIER=smoke   # minutes on a laptop: 1 seed, a few scenarios; runs in CI on every commit
make reproduce TIER=dev     # 3 seeds, 20 scenarios per condition
make reproduce TIER=full    # the published benchmark: 10 seeds, 50 scenarios per condition
```

Every stage is resumable and writes `results/<tier>/` (manifest, episode CSVs, aggregate CSVs, training metrics, replays, checksums). `adaptarena export --tier full` produces the experiment export package; `requirements-lock.txt` pins the software; `releases/` holds the released checkpoints with model cards.
Training reproducibility is stated as same machine and same software; evaluating a fixed checkpoint is bit-exact.

| Command | What it does |
| --- | --- |
| `adaptarena calibrate` | baselines and planner on validation seeds (difficulty calibration) |
| `adaptarena tune` | one small sweep per learned method, validation only |
| `adaptarena parity` | own PPO vs Stable-Baselines3 gate (blocks the headline run) |
| `adaptarena headline` | four arms, locked test suite, statistics, claim ledger |
| `adaptarena ablations` / `memory` | exploratory ablation ladder; locked memory-track predictions |
| `adaptarena stress` / `redteam` / `probes` | stress curves, verified failures, what-it-learned probes |
| `adaptarena site` / `report` / `export` | static site, technical report, export package |

## Live site

The static showcase is deployed to GitHub Pages from `site/published` by `.github/workflows/pages.yml` (repository Settings > Pages > Source: GitHub Actions). Rebuild it with `make publish` after re-running the experiments.

## Layout

```
src/adaptarena/core/   environment, batch env, planner, baselines, distributions   (NumPy only)
src/adaptarena/lab/    PPO, models (FF/LSTM/SSM), PLR buffer, statistics, probes, stress, red team, site/report builders
site/                  TypeScript + Canvas 2D showcase (builds to site/dist, plain static files)
analysis_plan.yaml     pre-registration; its SHA-256 is committed before the locked suite is generated
configs/tiers.yaml     smoke / dev / full sizes
tests/                 core, PPO, statistics and lab tests
```

Separation rules: the core never imports lab or frontend code; rendering never determines physics; policies consume observations, never the world object; evaluation never changes model weights; training runs headless.

## Claims this project will not make

That the agents developed intelligence or understand survival; that regret-guided training is better in general (only under this benchmark, with its limits stated); that the planner is optimal; that probes reveal reasoning; that domain randomization always improves robustness; that the environment models real ecology or that this is a new RL algorithm; any hardware speed or energy claim for the SSM or quantized policies that was not measured on named hardware.

## Related work (checked and cited, no novelty implied)

Minigrid (Chevalier-Boisvert et al., NeurIPS D&B 2023), Procgen (Cobbe et al., ICML 2020) and Craftax (Matthews et al., ICML 2024) are established benchmarks; Prioritized Level Replay (Jiang et al., ICML 2021), Replay-Guided Adversarial Environment Design (Jiang et al., NeurIPS 2021) and PAIRED (Dennis et al., NeurIPS 2020) are the replay and regret ideas used here. What this repository adds is narrow: a reference planner that gives solvability certificates and exact regret in a small world, bit-exact integer state, controlled interventions with stress curves, and a locked statistical protocol (Agarwal et al., NeurIPS 2021).

Apache-2.0 licensed.
