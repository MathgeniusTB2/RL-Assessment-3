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
src/         # environment, training, and evaluation code
notebooks/   # experiments and demos
```

## Setup

Flatland (`flatland-rl`) currently requires Python 3.10 or 3.11. Create an
environment with a supported interpreter, then install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

- [ ] Environment setup (Flatland demo, baseline reproduction)
- [ ] Custom scenarios (Central–Strathfield, Granville–Harris Park)
- [ ] Implement PPO / A2C / DQN
- [ ] Training + hyperparameter tuning
- [ ] Evaluation on hold-out scenarios (normal vs delayed)
- [ ] GUI / dashboard and demo
