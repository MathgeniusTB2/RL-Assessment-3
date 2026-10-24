# Every arm against its baseline (paired, 30 seeds 3000-3029, 220-step holds)

> The corridor DQN row below is the **superseded gamma=0.99 arm** (it collapses to
> completion 0.000). The reported corridor DQN result is in `v3_dqn_compact.csv`:
> -0.0577 -> -0.0461, delta +0.0116, 24/30 wins, p = 0.0014.

| env | arm | baseline | arm | delta | wins | p | completion (base -> arm) |
|---|---|---|---|---|---|---|---|
| corridor | A2C | -0.0583 | -0.0583 | -0.0000 | 0/30 (29 ties, 1 loss) | 1.0000 | 1.000 -> 1.000 |
| corridor | DQN | -0.0577 | -0.2890 | -0.2312 | 0/30 | 0.0000 | 1.000 -> 0.000 |
| corridor | PPO | -0.0583 | -0.0473 | +0.0109 | 23/30 | 0.0052 | 1.000 -> 1.000 |
| junction | A2C | -0.1769 | -0.0251 | +0.1518 | 30/30 | 0.0000 | 1.000 -> 0.000 |
| junction | DQN | -0.1769 | -0.0208 | +0.1561 | 30/30 | 0.0000 | 1.000 -> 0.183 |
| junction | PPO | -0.1769 | -0.0251 | +0.1518 | 30/30 | 0.0000 | 1.000 -> 0.000 |
