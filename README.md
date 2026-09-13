# 43008 Reinforcement Learning — Assignment 3

## Train Schedule Optimisation under Disruptions

Mini-project (40%) for 43008 Reinforcement Learning (Spring 2026).

**Team:** RailTech

| Student Name | Student ID |
|---|---|
| Daniel James Martirosov | 24948933 |
| Laurean Lim | 25049863 |
| Ziheng Deng | 25595315 |

## Overview

We train multi-agent reinforcement learning policies to optimise train schedules
under disruptions. The environment is [Flatland](https://flatland.aicrowd.com/intro.html),
a gridworld toolkit for multi-agent RL developed with Swiss Federal Railways (SBB).

- **Primary scenario:** Central–Strathfield (Main Suburban corridor).
- **Secondary scenario:** Granville–Harris Park (Main Western junction).
- **Disruptions:** stochastic malfunctions (track faults, maintenance, weather holds).
- **Algorithms:** PPO (primary), A2C, DQN — shared policy across agents.
- **Interface:** Flatland's built-in pygame renderer (headless PIL for figures).

## Repository layout

```
src/
  envs/        # Flatland environment: generator, tree observation, rewards
  train/       # PPO training loop + evaluation callback
  eval/        # curve plotting
results/       # generated: figures and training curves (gitignored)
notebooks/     # experiments and demos
```

## Setup

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.
Flatland (`flatland-rl`) requires Python 3.10+.

```bash
# full environment (Flatland + Stable-Baselines3 + torch + PettingZoo/SuperSuit)
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate
```

## Training

Train PPO with Stable-Baselines3 (a shared policy across agents via
PettingZoo/SuperSuit). The environment mirrors the paper's `small_v0` (25×25,
5 agents, 4 cities, sparse generator, malfunctions, tree observation depth 2,
rail/schedule regenerated each episode), and uses the paper's reward (−1 per
step, 0 on arrival; `--paper-reward`) with `VecNormalize` reward normalization.

```bash
# install full env (flatland-rl, sb3, torch, pettingzoo, supersuit)
uv pip install --python .venv -r requirements.txt

# short smoke test
python -m src.train.train_ppo --total-timesteps 20000 --n-envs 2 \
    --n-steps 128 --batch-size 128 --eval-freq 10000 --n-eval-episodes 3 \
    --out-dir results/smoke --run-name smoke

# full run (15M steps)
python -m src.train.train_ppo --seed 0 --total-timesteps 15000000 \
    --n-envs 8 --n-cpus 6 --eval-freq 100000 --n-eval-episodes 50 \
    --out-dir results/ppo --run-name ppo_seed0

# plot our training curves
python -m src.eval.plot_curves --train-glob 'results/ppo/*/eval.csv'
```

Notes:
- `--n-cpus 0` runs envs in-process. `--n-cpus > 0` uses SuperSuit
  multiprocessing; the training script forces the `fork` start method, which
  fixes a macOS `spawn` deadlock.
- Flatland already ships Cython-accelerated hot paths, so there is no extra
  build step. Threads do not help (the env step holds the GIL); use processes.
  Measured: pure env stepping scales ~4x at 8 workers, but full PPO training
  gains ~1.7x (6 workers) because policy forward/update becomes the bottleneck.
- Our evaluation logs `step, normalized_score, completion_rate` to `eval.csv`
  for plotting.

## Status

- [x] Environment setup (Flatland demo, training env)
- [x] PPO training pipeline (SB3 + PettingZoo/SuperSuit) + eval/plot
- [ ] Custom scenarios (Central–Strathfield, Granville–Harris Park)
- [ ] Implement PPO / A2C / DQN
- [ ] Training + hyperparameter tuning
- [ ] Evaluation on hold-out scenarios (normal vs delayed)
- [ ] GUI / dashboard and demo
