# 43008 Reinforcement Learning: Assignment 3

## Train Schedule Optimisation under Disruptions

Mini-project (40%) for 43008 Reinforcement Learning (Spring 2026).

**Team:** RailTech

| Student Name | Student ID |
|---|---|
| Daniel James Martirosov | 24948933 |
| Laurean Lim | 25049863 |
| Ziheng Deng | 25595315 |

## Overview

Multi-agent reinforcement learning policies that optimise train schedules under
disruptions, on [Flatland](https://flatland.aicrowd.com/intro.html) (a gridworld
toolkit developed with Swiss Federal Railways) populated from real TfNSW GTFS data.

| Scenario | Level | Services | Role |
|---|---|---|---|
| `network-simple-compact` | 87x8, four adjacent through tracks | 52 | **Primary.** Every reported result is on this map. |
| `central-strathfield` | 92x37 turnback level | 74 | Full-size corridor; the source the compact map is trimmed from. |
| `granville-harris-park` | 20x20 Y-link junction | 126 movements | Negative result: per-agent RL cannot arbitrate the junction. |

- **Disruptions:** timetable failures placed in the band between a stop's escape
  crossover and its platform (never on the crossover, never at a station),
  direction-balanced, three per episode with fixed 450-step holds.
- **Algorithms:** PPO (joint `MultiDiscrete` action over the whole fleet), plus A2C
  and DQN as per-agent baselines: one shared policy across agents.
- **Reward:** a dense *metric potential* (`r_t = sum_a [R_a(t) - R_a(t-1)]`) that
  reproduces Flatland's `DefaultRewards` exactly, so the shaped reward optimises the
  same objective the score is computed from.
- **Interface:** Flatland's built-in renderer, headless PILSVG.

## Deliverable

`notebooks/train_schedule_optimisation.ipynb` is the code deliverable: the
tree-observation builder, the environment factory (with the env-enforced stop
reflex), the disruption regime, the metric reward, the greedy/route-follower
baselines, the PettingZoo/SuperSuit wrappers, PPO/A2C/DQN, the evaluation harness
and every figure and table in the report. With `FULL_RUN = False` (the default) it
executes end to end in minutes and prints the recorded numbers; set it to `True` to
retrain inside the notebook.

## Results

**Corridor, 30 held-out seeds (3000-3029)** — checkpoints selected on seeds
9001/9002, so these seeds are unseen:

| policy | score | completion |
|---|---|---|
| forward-only baseline (never changes track) | -0.1859 | 0.881 |
| **joint PPO, 3M steps** | **-0.0772** | **0.968** |

Paired delta **+0.1087** (median +0.1099), **28/30 wins**, sign test **p = 0.0000**.
Rows: `results/pair_compact_e.csv`.

**Junction (Granville-Harris Park)** — the learned policy does not work here:

| policy | normal | delayed |
|---|---|---|
| scripted route follower | **126/126, +0.0000** | 76/126, -0.177 |
| joint PPO | 71/126, -0.199 | 60/126, -0.259 |

PPO loses 55 trains with no delays at all: the timetable encodes junction turns as
an explicit `route` dict, and a local tree observation does not expose them.

**A2C and DQN** run end to end but collapse to a constant action (A2C: `MOVE_LEFT`
100%, DQN: `STOP_MOVING` 100%), so the evaluator distinguishes four policies with
four different scores while the policies themselves stop discriminating.

## Repository layout

```
notebooks/                  # the deliverable
  train_schedule_optimisation.ipynb
  Scenario2/Scenario2.ipynb # Scenario 2 assembly (junior notebook)
data/
  levels/                   # prebuilt Flatland envs (loaded by the notebook)
  schedules/                # timetables extracted from GTFS
  gtfs/                     # TfNSW GTFS feed (NOT committed, ~1.4 GB)
scripts/                    # scenario builders + the training/eval entry points
docs/                       # report figures (Part-D renders, comparison GIF)
results/                    # the evidence behind the reported tables
requirements.txt
```

Shipped evidence is deliberately small: `results/SUMMARY.md`,
`results/pair_compact_e.csv` (the 30-seed paired test),
`results/curves/*.csv` (training curves for the two final arms) and
`results/station_access_compact.json` (the crossover probe `scripts/regimes.py`
reads at runtime). Model checkpoints (~9 MB each) and `runs/` are **not**
committed; the commands below regenerate them.

## Setup

Flatland needs Python 3.10+ (developed on 3.11).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate
```

Run Jupyter from the repository root: the notebook resolves `data/` upwards from the
working directory.

```bash
jupyter lab notebooks/train_schedule_optimisation.ipynb
```

## Reproducing the results

```bash
# corridor: the reported result (3M joint PPO, ~35 min on one GPU)
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

Scenario construction (only needed if the maps or timetables are rebuilt):

```bash
python scripts/build_scenario1_env.py        # central-strathfield env from its level + round trips
python scripts/build_simple_map.py --compact # the 87x8 corridor used by every result
```

`scripts/nb_harness.py` execs notebook cells directly, so the notebook stays the
single source of truth for the environment, reward and evaluation code.

## Notebook map

| Section | Contents |
|---|---|
| 1 | Setup, paths, runtime budget (`FULL_RUN`) |
| 2 | Scenario registry, environment factory, tree observation, greedy baseline |
| 3 | The round-trip schedule and the map render |
| 4 | PettingZoo to SuperSuit wrappers |
| 5 | PPO (joint), A2C, DQN |
| 6 | Evaluation: score, completion, delay, on-time, stops served |
| 7 | Greedy baseline and training-metric comparison |
| 8 | Save / load |
| 9 | Scenario 2 (Granville-Harris Park) and why it fails |
| 10 | Notes |
| 11 | The disruption regime (what "delayed" means here) |
| 12 | Results and limitations |
| 13 | Reproducibility |

## Two metric properties worth knowing

- **`normal` is a ceiling by construction** on the calibrated scenario:
  `calibrate_deadlines()` sets each deadline to the reference policy's own arrival
  plus one slack, so greedy scores exactly 0.0000. Headroom exists only under
  `delayed`, and every claim in this project is delayed-only.
- **"Never depart" is the value-optimal policy for a weak learner**: a cancelled
  train costs about -100 while one dispatched into a jammed junction costs about
  -600, so the metric itself punishes dispatch until the policy can avoid
  deadlocks. That basin explains the A2C/DQN collapses and the repeated PPO
  collapses during development.

## Status

- [x] Environment setup, scenario registry and observation
- [x] Central-Strathfield from the real TfNSW round-trip schedule
- [x] Granville-Harris Park Y-link junction (hand-painted level, 126 movements)
- [x] Condensed 87x8 corridor used for all results
- [x] Env-enforced stop reflex and multi-target reset fix
- [x] Disruption regime: band placement, 450-step holds, direction balance
- [x] Metric-potential dense reward (verified exact per term against `DefaultRewards`)
- [x] PPO, A2C, DQN implementations
- [x] 30-seed paired evaluation with sign test
- [x] Part-D interface figures and reports
- [x] Final report, presentation and peer reviews
