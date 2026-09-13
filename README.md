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
  baselines/   # paper W&B curve extraction + baseline plotting
data/          # generated: fetched W&B baseline CSVs (gitignored)
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

The baseline reproduction scripts only need the lightweight dependencies in
`requirements-baselines.txt` (no Flatland/RLlib):

```bash
uv pip install --python .venv -r requirements-baselines.txt
```

## Baselines

We reproduce the published learning curves of the original Flatland paper
([Mohanty et al., 2020, arXiv:2012.05893](https://arxiv.org/abs/2012.05893),
Figure 6) and use them as a reference baseline.

The paper's training code lives in
`flatland-association/flatland-baselines` / `AIcrowd/neurips2020-flatland-baselines`
(the `flatland-paper-baselines` branch), built on `ray[rllib]==0.8.5` and
`tensorflow==2.1`. That stack is x86-only and is not required to reproduce the
curves: the runs were logged to Weights & Biases, and we re-download the logged
history through the public, unauthenticated W&B GraphQL API.

Standard PPO is identified in the public `masterscrat/flatland` project by the
`ppo` tag with the stock tree observation on the 25×25 / 5-agent small sparse
grid and the full ~15M-step budget (9 runs). The paper averaged 3 seeds; all
identically-configured public runs are shown as fine lines with the mean in bold.

```bash
source .venv/bin/activate
python -m src.baselines.fetch_wandb --discover   # list PPO runs
python -m src.baselines.fetch_wandb --fetch      # download histories to CSV
python -m src.baselines.plot_baseline            # -> results/baselines/ppo_baseline_curves.png
```

- `src/baselines/fetch_wandb.py` — W&B GraphQL client, no API key required.
- `data/baselines/flatland_paper_ppo/` — per-run CSVs + `manifest.json`.
- `results/baselines/ppo_baseline_curves.png` — normalized score and completion
  rate versus training steps (Figure 6 style).

## Training (approximate RL reproduction)

We also train our own PPO baseline with the modern stack (current `flatland-rl`
+ PettingZoo/SuperSuit + Stable-Baselines3), as an approximation of the paper's
setup. The environment mirrors `small_v0` (25×25, 5 agents, 4 cities, sparse
generator, malfunctions, tree observation depth 2, rail/schedule regenerated
each episode). By default we use the paper's reward (−1 per step, 0 on arrival;
`--paper-reward`) and `VecNormalize` reward normalization (`--norm-reward`),
because Flatland's modern default rewards award partial credit for intermediate
stops and decouple the score from the completion rate.

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

# overlay our curve on the paper's W&B baseline
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
- Our evaluation logs `step, normalized_score, completion_rate` to `eval.csv`,
  matching the paper baseline schema.

## Status

- [x] Environment setup (Flatland demo, baseline reproduction)
- [x] Baseline curves reproduced from the paper's public W&B runs (standard PPO)
- [x] Modern PPO training pipeline (SB3 + PettingZoo/SuperSuit) + eval/plot
- [ ] Custom scenarios (Central–Strathfield, Granville–Harris Park)
- [ ] Implement PPO / A2C / DQN
- [ ] Training + hyperparameter tuning
- [ ] Evaluation on hold-out scenarios (normal vs delayed)
- [ ] GUI / dashboard and demo
