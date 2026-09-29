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

We train multi-agent reinforcement learning policies to optimise train schedules
under disruptions. The environment is [Flatland](https://flatland.aicrowd.com/intro.html),
a gridworld toolkit for multi-agent RL developed with Swiss Federal Railways (SBB).

- **Primary scenario:** Central-Strathfield (Main Suburban corridor), the 92x37
  turnback level, run on the real round-trip schedule.
- **Secondary scenario:** Granville-Harris Park (Main Western junction), a
  20x20 level with a merge, a flat crossing and a branch to Merrylands.
- **Disruptions:** stochastic malfunctions (track faults, maintenance, weather holds).
- **Algorithms:** PPO (primary), A2C, DQN, one shared policy across agents.
- **Reward:** Flatland's shaped `DefaultRewards` (punctuality + schedule
  adherence).
- **Interface:** Flatland's built-in renderer (headless PILSVG for figures).

## Deliverable

`notebooks/train_schedule_optimisation.ipynb` is the primary code deliverable (the
assignment prefers a Colab / IPython notebook). It holds the tree-observation
builder, the environment factory (with the env-enforced stop reflex), the greedy
baseline, the PettingZoo/SuperSuit wrappers and the evaluation/plotting code,
with no `src/` imports.

Each scenario is a **prebuilt Flatland env** (rail + fleet + timetable) plus its
source schedule, both read from `data/`, so run the notebook from the repository
root:

- `data/levels/central_strathfield.pkl`: the prebuilt scenario-1 env loaded by the notebook.
- `data/schedules/central_strathfield_roundtrips.json`: the scenario-1 source timetable.
- `data/levels/central_strathfield_rail.mpk`: the rail grid used to build the scenario-1 env.
- `data/levels/granville_harris_park.pkl`, `data/schedules/granville_harris_park.json`
  and `data/levels/granville_harris_park_rail.mpk`: the same three files for scenario 2.

Build (or rebuild) the envs before running the notebook:

```bash
python scripts/build_scenario1_env.py
python scripts/build_scenario2_env.py
```

## Repository layout

```
notebooks/      # the deliverable (observation + env + wrappers + algorithms)
data/levels/    # rail grid (source) + prebuilt env (loaded by the notebook)
data/schedules/ # source schedule(s)
scripts/        # scenario/env build + GTFS schedule extraction + interface render
docs/           # report figures (interface render, comparison GIFs)
results/        # generated: figures and training curves (gitignored)
```

`scripts/` builds the `data/` artifacts and is kept for provenance. The schedule
is assembled once into the prebuilt env, so the notebook stays small and readable.

## Setup

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.
Flatland (`flatland-rl`) requires Python 3.10+.

```bash
# full environment (Flatland + Stable-Baselines3 + torch + PettingZoo/SuperSuit)
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
source .venv/bin/activate

# register the venv as a Jupyter kernel for the notebook
.venv/bin/python -m ipykernel install --user \
    --name rl-assessment-3 --display-name "Python 3.11 (RL Assessment 3)"

# launch from the repo root (the notebook reads data/)
uv run jupyter lab notebooks/train_schedule_optimisation.ipynb
```

The notebook's `kernelspec` is pinned to `rl-assessment-3`, so it uses `.venv`
(where Flatland and SB3 are installed) rather than the system Python. It resolves
`data/` from the repository root, so start Jupyter from there.

## Training

The algorithm implementations are **TODO**: see
`notebooks/train_schedule_optimisation.ipynb` (section 5) for the PPO / A2C / DQN
stubs. The environment, vectorisation (PettingZoo/SuperSuit), reward normalisation
(`VecNormalize`) and evaluation infrastructure (score, completion rate, timetable
delay) are ready to train against.

Each algorithm should share one policy across agents and log
`step, normalized_score, completion_rate` to `results/<algo>/<run>/eval.csv`; the
notebook's section 7 comparison plots every run automatically.

The notebook follows the official Flatland SB3 baseline
(`flatland/ml/pettingzoo/examples/flatland_pettingzoo_stable_baselines.py`):
tree observation depth **3**, shortest-path predictor depth **50**, PPO
`MlpPolicy` with **lr 1e-3, batch 256, 8 envs, no `VecNormalize`**. The
observation is Flatland's flattened normalized tree (1020-d at depth 3).

Scenario 1 is dense: the round-trip schedule yields 74 agents on the 92x37 grid,
so the greedy baseline completes about 0.66 of them under `normal` and 0.57 under
`delayed`. That leaves real headroom for the algorithms to beat the baseline.

```bash
# full environment (Flatland + Stable-Baselines3 + torch + PettingZoo/SuperSuit)
uv pip install --python .venv -r requirements.txt
```

### Deployment (AWS SageMaker)

Per the Part-B plan, full-scale training runs on **AWS SageMaker (GPU)** with
PyTorch via a scripted SageMaker training job that the examiner can re-run.
Trained policies are exported and demonstrated locally in
`notebooks/train_schedule_optimisation.ipynb` using Flatland's renderer, with a
recorded video prepared for the presentation. The algorithm entry points in the
notebook (section 5) are currently TODO.

Notes:
- Use SuperSuit's multiprocessing for data collection (`num_cpus > 0`); on macOS
  force the `fork` start method to avoid a `spawn` deadlock.
- Flatland ships Cython-accelerated hot paths, so there is no extra build step.
  Threads do not help (the env step holds the GIL); use processes.
- `notebooks/train_schedule_optimisation.ipynb` (section 6) provides a
  framework-agnostic `evaluate(policy_fn=None)` (greedy by default) reporting
  score, completion, mean/total timetable delay, on-time fraction, **stops
  served**, cumulative return and a collision/deadlock proxy. `evaluate_seeds`
  runs it once per seed in `SEEDS` (section 1) and reports mean +/- std; section
  6.1 compares the greedy baseline under `normal` and `delayed` across the seeds.
- Flatland only counts an intermediate stop as served if the train is `STOPPED`
  at the waypoint. The notebook's **force-stop reflex** (default on) does this
  automatically; with `force_stop=False` a policy must learn to stop itself or
  lose the `intermediate_not_served` penalty.

## Environments

The notebook exposes one factory over a **scenario registry** and **operating
conditions**:

```python
FORCE_STOP # env-enforce scheduled stops (default True; see section 2.4)
SCENARIOS  # "central-strathfield" | "granville-harris-park" (section 2.3)
CONDITIONS # "normal" (deterministic) | "delayed" (Poisson malfunctions, 1/540)

env = make_env(SCENARIO, CONDITION, seed=SEED)     # RailEnv: render / step
make_parallel_env(SCENARIO, CONDITION, seed=...)   # PettingZoo: evaluation
build_vec_env(SCENARIO, CONDITION, ...)            # SuperSuit VecEnv: training
evaluate(policy_fn=None, SCENARIO, CONDITION)      # score / delay / return
```

`delayed` injects Poisson malfunctions via the env's effects generator;
`normal` is deterministic. A train changes track only via `MOVE_LEFT` /
`MOVE_RIGHT` at switch cells, not via the line.

`FORCE_STOP` (default on) env-enforces scheduled stops: a train at a stop cell is
forced to `STOP_MOVING` and held for the scheduled dwell, so the policy never has
to learn to stop. Pass `force_stop=False` to leave stopping to the policy.

`SCENARIO` and `CONDITION` are set once in section 1 (defaults
`"central-strathfield"` / `"normal"`), so every constructor picks up the same
setting. Every factory also takes `scenario=`, e.g.
`build_vec_env(scenario="granville-harris-park")` trains on scenario 2.

## Scenario 1 (Central-Strathfield, round-trip schedule)

Scenario 1 uses the 92x37 Central-Strathfield turnback level and the real
round-trip schedule: every heavy-rail service (T1, T2, T3, T9, CCN, BMT) that
crosses the corridor in the 16:00-19:00 window, paired into physical round trips
(out to Central, reverse at a dead-end platform, return to Strathfield). The
schedule is extracted from the local Greater Sydney GTFS bundle and then assembled
into the prebuilt env:

```bash
# 1. rebuild the round-trip schedule (defaults: normal Thursday 2026-09-24, 16:00-19:00)
python scripts/build_scenario1_roundtrips.py --date 2026-09-24 --window 16:00-19:00

# 2. assemble the prebuilt env (rail grid + schedule -> agents + timetable)
python scripts/build_scenario1_env.py

# 3. run the deliverable from the repo root
jupyter notebook notebooks/train_schedule_optimisation.ipynb
```

Timing model: 1 Flatland step = 20 s; express speed 1.0, all-stops 0.5; real
departure spacing with reachable per-stop arrival deadlines. Evaluation reports
normalized score, completion rate, mean/total timetable delay, on-time fraction,
stops served, cumulative return and a collision/deadlock proxy under both
`normal` and `delayed` conditions.

The notebook's section 7 comparison renders the scenario-1 greedy baseline under
both conditions (`docs/comparison/scenario1_greedy_normal.gif` and
`scenario1_greedy_delayed.gif`) and overlays the training metrics of each
algorithm against the normal baseline. Section 7.3 puts the scenario-2 greedy
baseline next to it.

## Scenario 2 (Granville-Harris Park, flat junction)

Scenario 2 is a 20x20 level on the Main Western line between Harris Park (west)
and Granville (east): four parallel tracks (Up/Down Main, Up/Down Suburban), five
crossovers, and a branch to Merrylands that joins the Up Suburban at a merge,
leaves the Down Suburban at a junction and crosses it on the flat. It is a
junction problem: trains converging on the merge and the flat crossing,
especially under `delayed`.

The schedule is an illustrative 17:15-17:25 PM-peak sample rather than GTFS: T1
between Granville and Harris Park, T2 between Granville and Merrylands, and a Blue
Mountains intercity on the Main lines, 10 services in total.

```bash
# assemble the prebuilt env (rail grid + schedule -> agents + timetable)
python scripts/build_scenario2_env.py
```

It uses the same timing model and deadline rule as scenario 1 (1 step = 20 s,
express 1.0, all-stops 0.5). Section 9 of the notebook shows the level and the
schedule, and section 7.3 compares its greedy baseline with scenario 1
(`docs/comparison/scenario2_greedy_normal.gif`, `scenario2_greedy_delayed.gif`).

## Status

- [x] Environment setup (Flatland demo, training env)
- [x] PettingZoo/SuperSuit vectorisation + evaluation/plot infrastructure
- [x] Scenario 1 (Central-Strathfield) with real TfNSW round-trip schedule + delay metrics
- [x] Force-stop reflex (env-enforced scheduled stops)
- [ ] Implement PPO
- [ ] Implement A2C
- [ ] Implement DQN
- [x] Custom scenario 2 (Granville-Harris Park) with a merge, flat crossing and branch
- [ ] Training + hyperparameter tuning
- [ ] Evaluation on hold-out scenarios (normal vs delayed)
- [ ] GUI / dashboard and demo
