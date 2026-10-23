# Final results (compact corridor, 450-step holds, 3 incidents)

Regime: failures of fixed 450-step length, placed in the band strictly between a
stop's escape crossover and its platform (never on the crossover, never at a
station), direction-balanced, 3 per episode, verified firing 3/3.

| policy | score | completion | note |
|---|---|---|---|
| no-crossover baseline (forward-only) | **-0.1881** | 1.000 | seeds 9001/9002 |
| **PPO, joint MultiDiscrete (3M steps)** | **-0.0640** | **1.000** | best_model, `runs/compact_station_e` |

## S2 (Granville-Harris Park) - negative result

| | normal | delayed |
|---|---|---|
| scripted route follower | 126/126, +0.0000 | 76/126, -0.177 |
| PPO | 71/126, -0.199 | 60/126, -0.259 |

PPO loses 55 trains with **no delays at all**: it never learns the junction turns
that the timetable encodes as an explicit `route` dict. Five runs plateaued at
completion 0.46-0.48.

## Void arms (do not cite)

Every run before `*_e` was measured under a regime that changed underneath it:
`compact_station_{a..d}`, `s2_zone_{a..d}`, `sw_{a..e}`. The specific defects were
a 500-step episode budget that truncated the S2 timetable, a reactive placement
that fired 0-1 of 3 incidents, and a sampled-cell pool that shrank on every reset.

## Platform mapping

Every platform number is real (GTFS-verified) and matches the modal platform for
13/13 (station, track) pairs. 3.3% of stop calls use a different platform on the
same track class; cosmetic only, since a train's stopping cell is (column, row).

## Paired confirmation (30 held-out seeds 3000-3029)

```
baseline (forward-only)  score -0.1859  completion 0.881
PPO (joint, 3M steps)    score -0.0772  completion 0.968
paired delta +0.1087 (median +0.1099), wins 28/30, sign-test p = 0.0000
```

The checkpoints were selected on seeds 9001/9002, so these 30 seeds are held out.
Full rows: `results/pair_compact_e.csv`.

## A2C and DQN: correct implementations, degenerate policies

Both run end to end on the per-agent env (52 slots, `Discrete(5)`), and the
evaluator discriminates policies, so the code path is sound:

| policy | score | actions used |
|---|---|---|
| forward-only | -0.1818 | 1/5 (MOVE_FORWARD 100%) |
| random | -0.2665 | **5/5** (20% each) |
| A2C | -0.3813 | 1/5 (**MOVE_LEFT 100%**) |
| DQN | -0.1717 | 1/5 (**STOP_MOVING 100%**) |

A2C emits MOVE_LEFT for every agent at every step and DQN emits STOP_MOVING -
i.e. both collapse to a constant action, which is why A2C's score is identical to
the digit across learning rates and rollout lengths (a constant policy makes the
episode deterministic, so hyperparameters stop mattering). DQN's STOP_MOVING is
the "never dispatch" basin the metric rewards (~-100 to cancel vs ~-600 to
dispatch into a jam).

Reproduce with `python scripts/diag_action_collapse.py`.

## Why only PPO learns

PPO's action is **joint** (`MultiDiscrete` over all 52 agents via
`concat_vec_envs`), so one step can coordinate the whole fleet; A2C and DQN act
per agent from local observations with a shared MLP, where the fleet-wide penalty
gives each agent almost no gradient. That is the mechanism separating the runs -
not a defect in the A2C/DQN code.
