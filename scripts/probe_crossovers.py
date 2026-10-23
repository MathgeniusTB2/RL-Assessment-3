#!/usr/bin/env python3
"""Do the corridor policies actually use the crossovers?

Runs one episode (seed 3000, the shipped regime) per policy and counts:
  * how many crossovers the level has (cells with a lateral successor)
  * actions emitted per action id
  * lateral *requests* (MOVE_LEFT / MOVE_RIGHT) and how many were legal
  * effective lateral moves, i.e. steps where the train's row changed
  * distinct crossover cells actually traversed
"""
from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.home() / "Documents/Code/RL/Assessment 3/scripts"))

import nb_harness  # noqa: E402
import priority_obs  # noqa: E402
import regimes  # noqa: E402

CKPT = pathlib.Path.home() / "backups/rl-a3-gpu-2026-10-23/runs"
SEED = 3000

nb = nb_harness.load()
priority_obs.install_priority_features(nb)
cls = nb["GymTreeObsForRailEnv"]
_orig = cls.__init__


def patched(self, *a, **kw):
    kw.setdefault("observation_radius", 50)
    _orig(self, *a, **kw)


cls.__init__ = patched
nb["SCENARIO"] = "network-simple-compact"

parallel = nb["make_parallel_env"](scenario="network-simple-compact", condition="normal")
raw = parallel._wrap
regimes.install(raw, "network-simple-compact", 3)
handles = list(parallel.possible_agents)

# --- how many crossovers does the level have? -----------------------------
crossovers = set()
for row in range(raw.height):
    for col in range(raw.width):
        for direction in range(4):
            for nxt, ndir in raw.rail.get_successor_configurations(((row, col), direction)):
                delta = (int(ndir) - direction) % 4
                if delta in (1, 3):
                    crossovers.add((int(nxt[0]), int(nxt[1])))
print(f"level: {raw.width}x{raw.height}, {len(handles)} agents, "
      f"{len(crossovers)} cells with a lateral successor")


def load(name):
    from stable_baselines3 import A2C, DQN, PPO

    path = CKPT / name / "best_model.zip"
    algo = name.split("_")[1]
    return {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo].load(str(path), device="cpu")


def run(policy):
    obs, _ = parallel.reset(seed=SEED)
    prev = {}
    actions_hist = collections.Counter()
    lateral_req = legal_req = lateral_moves = 0
    used_cells = set()
    steps = 0
    for _ in range(int(raw._max_episode_steps)):
        steps += 1
        if policy == "forward":
            acts = {h: 2 for h in handles}
        elif policy == "greedy":
            plan = nb["greedy_actions"](raw)
            acts = {h: int(plan.get(h, 0)) for h in handles}
        else:
            batch = np.stack([obs[h] for h in handles])
            chosen = policy.predict(batch, deterministic=True)[0]
            acts = {h: int(a) for h, a in zip(handles, chosen)}
        for h, a in acts.items():
            actions_hist[a] += 1
        # legal lateral successors before stepping
        legal = set()
        for h in handles:
            cfg = raw.agents[h].current_configuration
            if cfg is None or cfg[0] is None:
                continue
            pos, direction = (int(cfg[0][0]), int(cfg[0][1])), int(cfg[1])
            for nxt, ndir in raw.rail.get_successor_configurations((pos, direction)):
                if (int(ndir) - direction) % 4 in (1, 3):
                    legal.add((h, int(nxt[0]), int(nxt[1])))
        for h, a in acts.items():
            if a in (1, 3):
                lateral_req += 1
        obs, _, terms, _, _ = parallel.step(acts)
        for h in handles:
            cfg = raw.agents[h].current_configuration
            pos = None if cfg is None or cfg[0] is None else (int(cfg[0][0]), int(cfg[0][1]))
            old = prev.get(h)
            if pos is not None and old is not None and pos != old and pos[0] != old[0]:
                lateral_moves += 1
                used_cells.add((pos[0], pos[1]))
            prev[h] = pos
        legal_req += sum(1 for (h, r, c) in legal if acts.get(h) in (1, 3)
                         and tuple((r, c)) == (prev.get(h) or (r, c)))
        if all(v for k, v in terms.items() if k != "__all__"):
            break
    total = sum(actions_hist.values())
    share = " ".join(f"{a}:{100*actions_hist[a]/total:.0f}%" for a in range(5) if actions_hist[a])
    return dict(steps=steps, actions=share, lateral_req=lateral_req,
                lateral_moves=lateral_moves, cells=len(used_cells))


print(f"\n{'policy':<12} {'steps':>5} {'lateral req':>11} {'lateral moves':>13} "
      f"{'crossover cells':>15}   actions")
for label, pol in (("forward-only", "forward"), ("greedy (map)", "greedy"),
                   ("PPO", load("v2_ppo_compact")),
                   ("A2C", load("v2_a2c_compact")),
                   ("DQN", load("v3_dqn_compact"))):
    r = run(pol)
    print(f"{label:<12} {r['steps']:>5} {r['lateral_req']:>11} {r['lateral_moves']:>13} "
          f"{r['cells']:>15}   {r['actions']}")
