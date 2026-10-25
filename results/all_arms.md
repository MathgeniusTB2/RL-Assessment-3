# Every arm against its baseline

All rows are paired within seed. The baseline and the arm run on the same env, the
same seed and the same disruptions; the delta is the mean per-seed difference.

## Current: corridor, block regime (exogenous blockages, 2 x 400 steps)

30 held-out seeds 3000-3029, observation 1032-d, checkpoints selected on 9001-9004.

| map | arm | baseline | arm | delta | wins | p | completion | crossings / ep |
|---|---|---|---|---|---|---|---|---|
| corridor | PPO | -0.0662 | -0.0005 | +0.0657 | 30/30 | < 1e-4 | 1.000 -> 1.000 | 40 (forward 0) |

Evidence: `results/pair_block_ppo.csv`; curve `results/curves/ppo_corridor_block.csv`;
figure `results/figures/corridor_{forward,ppo}_block.gif`; crossing audit
(`scripts/crossing_audit.py decisions`): 101 blocked visits -> 81 turns, 202 open -> 0.
Controls (measured 2026-10-25 on the GPU box): clean network, normal condition, seeds
3000-3002 -> greedy 0.0000 and PPO 0.0000, delta 0.0000 (`results/clean_normal.csv`), with
0 turns in 546 clear-track crossover visits (`results/clean_audit.csv`); with the blockage
suppressed but the Poisson malfunction layer left on PPO is
-0.0133 against greedy -0.0165, delta +0.0032, 3/3, p = 0.25 (`results/clean_delayed_none.csv`).
Hand-written detour teacher -0.0005;

Train-borne diagnostic (retired, log record only): with the train-borne band regime a
rerouting checkpoint fired none of the incidents it was given and scored the same as with
no disruption at all. The checkpoint was not shipped, so this is a log observation, not a
reproducible result - and it is the reason the corridor moved to an exogenous blockage.

## Legacy: corridor, band regime (retired, 1030-d observation)

Superseded by the block-regime row above. The band regime fired train-borne failures
inside the stretch between a stop's escape crossover and its platform, which measured
0 avoidable incidents in 9 samples, and its band table cannot be compared with the
current arms. Kept here as the record of what was re-run and why.

| map | arm | baseline | arm | delta | wins | p | completion |
|---|---|---|---|---|---|---|---|
| corridor | PPO | -0.0583 | -0.0473 | +0.0109 | 23/30 | 0.0052 | 1.000 -> 1.000 |
| corridor | A2C | -0.0583 | -0.0583 | +0.0000 | 0/30 (29 ties) | 1.0000 | 1.000 -> 1.000 |
| corridor | DQN (gamma 0.999) | -0.0583 | -0.0473 | +0.0109 | 23/30 | 0.0052 | 1.000 -> 1.000 |

## Junction: train-borne regime (3 x 220 steps)

30 held-out seeds, baseline is the scripted route follower, observation 1030-d (these
checkpoints predate the two features added for the corridor and no longer load).

| map | arm | baseline | arm | delta | wins | p | completion (base -> arm) |
|---|---|---|---|---|---|---|---|
| junction | PPO | -0.1769 | -0.0251 | +0.1518 | 30/30 | < 1e-4 | 1.000 -> 0.000 |
| junction | A2C | -0.1769 | -0.0251 | +0.1518 | 30/30 | < 1e-4 | 1.000 -> 0.000 |
| junction | DQN | -0.1769 | -0.0208 | +0.1561 | 30/30 | < 1e-4 | 1.000 -> 0.183 |

Every learned arm "wins" by parking the fleet: the metric charges a cancellation less
than a dispatch into a jam, so this is a metric pathology, not a better policy.
Evidence: `results/v2_ppo_s2.csv`, `results/v2_a2c_s2.csv`, `results/v2_dqn_s2.csv`.
