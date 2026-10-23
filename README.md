# 43008 Reinforcement Learning: Assignment 3

## Train Schedule Optimisation under Disruptions

Mini-project (40%) for 43008 Reinforcement Learning (Spring 2026).

**Team:** RailTech

| Student Name | Student ID |
|---|---|
| Daniel James Martirosov | 24948933 |
| Laurean Lim | 25049863 |
| Ziheng Deng | 25595315 |

## What this is

`notebooks/train_schedule_optimisation.ipynb` is the deliverable. It builds two
Flatland multi-agent scenarios from real TfNSW GTFS data, trains policies that
share one network across all trains, and evaluates them under a deterministic
disruption regime:

| Scenario | Level | Services | Result |
|---|---|---|---|
| `network-simple-compact` | 87x8, four adjacent through tracks | 52 | **PPO beats the forward-only baseline by +0.109** (30 held-out seeds) |
| `granville-harris-park` | 20x20 Y-link junction | 126 movements | learned policy fails; diagnosed under **The junction** |

- **Algorithms:** PPO (joint `MultiDiscrete` over the fleet), A2C and DQN (per-agent).
- **Reward:** the assessment metric itself, paid densely as a potential,
  `r_t = R(t) - R(t-1)`, which reproduces `DefaultRewards` exactly.
- **Disruptions:** three fixed 450-step failures per episode, placed in the band
  between a stop's escape crossover and its platform, direction-balanced.

## Results

Corridor, 30 held-out seeds (3000-3029), checkpoints selected on unseen seeds:

| policy | score | completion |
|---|---|---|
| forward-only baseline | -0.1859 | 0.881 |
| **joint PPO (3M steps)** | **-0.0772** | **0.968** |

Paired delta **+0.1087** (median +0.1099), **28/30 wins**, sign test **p < 0.0001**.
Rows: `results/pair_compact_e.csv`.

Junction: route follower 126/126 (+0.0000) vs PPO 71/126 (-0.199) under no delays —
the timetable's junction turns are not recoverable from a local observation.

## Repository layout

```
notebooks/
  train_schedule_optimisation.ipynb   the deliverable (46 cells)
  Scenario2/Scenario2.ipynb           Scenario 2 assembly, executed by scripts/s2_env.py
data/levels/                          prebuilt Flatland envs (fixed inputs)
data/schedules/                       timetables extracted from GTFS
scripts/                              environment/regime/reward/training modules the notebook imports
results/                              the evidence behind the reported tables
```

Ten scripts are imported by the notebook at run time (`regimes`, `disruptions`,
`metric_rewards`, `train_ppo` and its dependencies, `s2_env`); `train_algo`,
`eval_pair` and `diag_action_collapse` reproduce the reported runs from the command
line. The notebook, not the scripts, is the graded artifact.

Not shipped: the GTFS feed (~1.4 GB), model checkpoints (~9 MB each, ~35 min to
retrain) and any rendered figures.

## Setup

Flatland needs Python 3.10+ (developed on 3.11).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate

# from the repository root, so data/ resolves
jupyter lab notebooks/train_schedule_optimisation.ipynb
```

`FULL_RUN = False` (in **Imports and configuration**) prints the recorded results instead of retraining, so the
notebook executes end to end in a few minutes. Set it to `True` to retrain inside
the notebook.

## Reproducing the reported runs

```bash
# corridor: 3M joint PPO, ~35 min on one GPU
python scripts/train_ppo.py --scenario network-simple-compact --disruption few_long \
    --incidents 3 --reward metric --shaping --priority --observation-radius 50 \
    --arrival-bonus 50 --stuck-penalty 1 --timesteps 3000000 \
    --eval-every 250000 --out runs/compact_station_e

# the paired test on held-out seeds
python scripts/eval_pair.py --scenario network-simple-compact --baseline forward \
    --disruption few_long --incidents 3 --priority --observation-radius 50 \
    --model runs/compact_station_e/best_model.zip \
    --seeds 3000,...,3029 --out results/pair_compact_e.csv

# A2C / DQN and the collapse diagnostic
python scripts/train_algo.py --algo a2c --out runs/a2c_v2
python scripts/diag_action_collapse.py
```

`scripts/nb_harness.py` executes the notebook's cells directly, so training and
evaluation cannot drift from the code the marker reads.
