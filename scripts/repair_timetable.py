#!/usr/bin/env python3
"""Prototype: repair the scenario-1 timetable so its deadlines are reachable on
the expanded grid, while preserving the real-life schedule structure.

Problem
-------
``scripts/build_scenario1_env.py`` derived each stop's arrival deadline from the
**horizontal column distance** between stations (``abs(cells[i+1][1] - cells[i][1])``).
After the map was expanded the true rail path between stops is longer (detours,
terminus geometry), so deadlines are unreachable: greedy is late even with no
disruptions (on-time = 0, min lateness +1). Separately, some services target a
Central platform cell (e.g. ``(30, 84)``) that is not reachable on the expanded
grid at all, so those agents can never finish.

Fix (two steps, both in this prototype)
---------------------------------------
1. ``fix_targets``: any agent whose target configuration is unreachable from its
   spawn is re-pointed at the reachable platforms of the *same station* (same
   column) and the distance map is rebuilt.
2. ``calibrate_timetable``: run the reference greedy policy once with the
   intermediate holds removed to measure the true per-stop travel, then set each
   arrival deadline to that measured arrival plus the real dwell schedule plus a
   slack. Origin departures and dwell durations are preserved, so the timetable
   stays aligned with the real schedule. A couple of refinement passes absorb
   conflict/queueing delays.

Usage:
    MPLBACKEND=Agg .venv/bin/python scripts/repair_timetable.py
    MPLBACKEND=Agg .venv/bin/python scripts/repair_timetable.py --slack 2 --seed 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions as A
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.rewards import DefaultRewards
from flatland.envs.step_utils.states import TrainState

REPO_ROOT = Path(__file__).resolve().parents[1]
LEVEL_PATH = REPO_ROOT / "data" / "levels" / "central_strathfield.pkl"


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def load_env(seed: int = 0) -> RailEnv:
    env, _ = RailEnvPersister.load_new(
        str(LEVEL_PATH), obs_builder=DummyObservationBuilder(), rewards=DefaultRewards()
    )
    env.seed_history = list(env.seed_history) or [0]
    env.reset(random_seed=seed)
    return env


# --------------------------------------------------------------------------- #
# Config graph helpers (position, direction) over the rail grid
# --------------------------------------------------------------------------- #
def _conf(position, direction) -> tuple[tuple[int, int], int]:
    return (int(position[0]), int(position[1])), int(direction)


def _successors(env: RailEnv, conf):
    return [_conf(p, d) for (p, d) in env.rail.get_successor_configurations(conf)]


def reachable_configs(env: RailEnv, start) -> set:
    from collections import deque

    seen = {start}
    q = deque([start])
    while q:
        for nxt in _successors(env, q.popleft()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


# --------------------------------------------------------------------------- #
# 1. Fix unreachable targets
# --------------------------------------------------------------------------- #
def fix_targets(env: RailEnv, verbose: bool = True) -> list[int]:
    remapped: list[int] = []
    for agent in env.agents:
        ic = agent.initial_configuration
        if ic is None or ic[0] is None:
            continue
        start = _conf(ic[0], ic[1])
        seen = reachable_configs(env, start)

        targets = {_conf(p, d) for (p, d) in agent.targets}
        if targets & seen:
            continue

        # preferred: keep the intended target alternatives that are reachable
        intended_cfgs: set = set()
        for wp in agent.waypoints[-1]:
            pos = (int(wp.position[0]), int(wp.position[1]))
            if wp.direction is None:
                intended_cfgs |= {(pos, d) for d in range(4)}
            else:
                intended_cfgs.add((pos, int(wp.direction)))
        reachable_intended = intended_cfgs & seen
        if reachable_intended:
            agent.targets = set(reachable_intended)
            remapped.append(agent.handle)
            if verbose:
                print(f"  h{agent.handle}: target restored to reachable "
                      f"{sorted(reachable_intended)}")
            continue

        # fallback: nearest reachable platform at the intended station column(s)
        intended = [
            ((int(wp.position[0]), int(wp.position[1])),
             None if wp.direction is None else int(wp.direction))
            for wp in agent.waypoints[-1]
        ]
        cols = {p[1] for p, _ in intended}
        cands = [c for c in seen if c[0][1] in cols]
        if not cands:
            if verbose:
                print(f"  h{agent.handle}: no reachable target at columns {sorted(cols)}")
            continue

        def score(conf):
            pos, d = conf
            dir_pen = 0 if any(wd == d for _, wd in intended) else 1
            row_pen = min((abs(pos[0] - p[0]) for p, _ in intended if p[1] == pos[1]),
                          default=999)
            return (dir_pen, row_pen)

        best = min(cands, key=score)
        # keep every platform alternative with the same column and heading
        new_targets = {c for c in cands if c[1] == best[1] and c[0][1] == best[0][1]}
        agent.targets = set(new_targets)
        agent.waypoints[-1] = [
            Waypoint((int(pos[0]), int(pos[1])), Grid4TransitionsEnum(int(d)))
            for (pos, d) in sorted(new_targets)
        ]
        remapped.append(agent.handle)
        if verbose:
            print(f"  h{agent.handle}: target {sorted(intended)} -> {sorted(new_targets)}")

    if remapped:
        env.distance_map.reset(env.agents, env.rail)
    return remapped


# --------------------------------------------------------------------------- #
# 1b. Fix wrong-way spawns
# --------------------------------------------------------------------------- #
def fix_spawns(env: RailEnv, verbose: bool = True) -> list[int]:
    """Relocate trains spawned on a track whose dominant direction is opposite
    their heading.

    The schedule builder mapped a few return services onto an ``up`` (eastbound)
    track, so they run westbound on row 28 and deadlock head-on with the
    eastbound flow — the map only completes ~2/3 of the fleet because of it.
    Agents on a row where the majority of spawns face the other way are moved to
    the nearest column-compatible spawn whose row flows the agent's way.
    """
    from collections import Counter

    by_row: dict[int, Counter] = {}
    for agent in env.agents:
        row = int(agent.initial_configuration[0][0])
        direction = int(agent.initial_configuration[1])
        by_row.setdefault(row, Counter())[direction] += 1

    dominant = {row: c.most_common(1)[0][0] for row, c in by_row.items()}
    flows = {d: [row for row, dom in dominant.items() if dom == d] for d in range(4)}

    def _valid(row, col, direction) -> bool:
        return bool(env.rail.get_successor_configurations(
            ((int(row), int(col)), int(direction))))

    moved: list[int] = []
    for agent in env.agents:
        row = int(agent.initial_configuration[0][0])
        col = int(agent.initial_configuration[0][1])
        direction = int(agent.initial_configuration[1])
        if dominant.get(row) == direction:
            continue
        candidates = [
            r for r in flows.get(direction, []) if r != row and _valid(r, col, direction)
        ]
        if not candidates:
            continue
        new_row = min(candidates, key=lambda r: abs(r - row))
        agent.initial_configuration = ((new_row, col), Grid4TransitionsEnum(direction))
        agent.initial_position = (new_row, col)
        moved.append(agent.handle)
        if verbose:
            print(f"  h{agent.handle}: spawn (row {row}, dir {direction}) -> "
                  f"(row {new_row}, dir {direction})")
    return moved


# --------------------------------------------------------------------------- #
# Reference policy (greedy + force-stop), from the deliverable notebook
# --------------------------------------------------------------------------- #
def greedy_actions(env: RailEnv) -> dict[int, int]:
    distance_map = env.distance_map.get()
    step = env._elapsed_steps
    actions: dict[int, int] = {}
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        configuration = agent.current_configuration
        if configuration is None or configuration[0] is None:
            actions[handle] = A.MOVE_FORWARD.value
            continue
        position = tuple(int(x) for x in configuration[0])
        direction = int(configuration[1])

        if any(
            tuple(int(x) for x in target[0]) == position
            and (target[1] is None or int(target[1]) == direction)
            for target in agent.targets
        ):
            actions[handle] = A.STOP_MOVING.value
            continue

        at_intermediate = False
        for j in range(1, len(agent.waypoints) - 1):
            if any(
                tuple(int(x) for x in wp.position) == position
                and (wp.direction is None or int(wp.direction) == direction)
                for wp in agent.waypoints[j]
            ):
                at_intermediate = True
                earliest = agent.waypoints_earliest_departure[j]
                if agent.state == TrainState.STOPPED and (earliest is None or step >= earliest):
                    actions[handle] = A.MOVE_FORWARD.value
                else:
                    actions[handle] = A.STOP_MOVING.value
                break
        if at_intermediate:
            continue

        best = None
        for next_position, next_direction in env.rail.get_successor_configurations(
            (position, direction)
        ):
            row, col = int(next_position[0]), int(next_position[1])
            value = distance_map[handle, row, col, int(next_direction)]
            if not np.isfinite(value):
                continue
            delta = (int(next_direction) - direction) % 4
            if delta == 2:
                continue
            if best is None or value < best[0]:
                best = (value, delta)
        if best is None or best[1] == 0:
            actions[handle] = A.MOVE_FORWARD.value
        elif best[1] == 1:
            actions[handle] = A.MOVE_RIGHT.value
        else:
            actions[handle] = A.MOVE_LEFT.value
    return actions


def force_stop_actions(env: RailEnv, actions: dict[int, int], served: dict[int, set]) -> dict[int, int]:
    out = dict(actions)
    for handle in list(out.keys()):
        agent = env.agents[handle]
        cfg = agent.current_configuration
        if cfg is None or cfg[0] is None:
            continue
        pos = tuple(int(x) for x in cfg[0])
        waypoints = agent.waypoints
        if waypoints is None:
            continue
        for j in range(1, len(waypoints) - 1):
            alts = {
                (tuple(int(x) for x in wp.position), int(wp.direction))
                for wp in waypoints[j]
                if wp.direction is not None
            }
            if (pos, int(cfg[1])) not in alts:
                continue
            earliest = agent.waypoints_earliest_departure[j]
            if agent.state == TrainState.STOPPED:
                served.setdefault(handle, set()).add(j)
                if earliest is not None and env._elapsed_steps < earliest:
                    out[handle] = A.STOP_MOVING.value
            else:
                out[handle] = A.STOP_MOVING.value
            break
    return out


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def _matches(wp: Waypoint, pos, direction) -> bool:
    return (
        (int(wp.position[0]), int(wp.position[1])) == pos
        and (wp.direction is None or int(wp.direction) == direction)
    )


def rollout(env: RailEnv, seed: int, record_first: bool = False):
    """Run greedy + force-stop; return (arrival steps, first-match steps)."""
    env.reset(random_seed=seed, regenerate_rail=False, regenerate_schedule=False)
    served = {h: set() for h in env.get_agent_handles()}
    first: dict[int, dict[int, int]] = {h: {} for h in env.get_agent_handles()}
    arrival: dict[int, int] = {}
    for step in range(1, env._max_episode_steps + 1):
        actions = force_stop_actions(env, greedy_actions(env), served)
        _, _, dones, _ = env.step(actions)
        for h in env.get_agent_handles():
            agent = env.agents[h]
            if agent.state == TrainState.DONE and h not in arrival:
                arrival[h] = step
                if record_first:
                    first[h][len(agent.waypoints) - 1] = step
            cfg = agent.current_configuration
            if record_first and cfg is not None and cfg[0] is not None:
                pos = (int(cfg[0][0]), int(cfg[0][1]))
                d = int(cfg[1])
                for j, group in enumerate(agent.waypoints):
                    if j not in first[h] and any(_matches(wp, pos, d) for wp in group):
                        first[h][j] = step
        if dones["__all__"]:
            break
    return arrival, first


def _residual_lateness(env: RailEnv, arrival: dict[int, int]) -> dict[int, int]:
    out = {}
    for h, t in arrival.items():
        scheduled = env.agents[h].latest_arrival
        if scheduled is not None:
            out[h] = t - scheduled
    return out


# --------------------------------------------------------------------------- #
# 2. Calibrate the timetable deadlines
# --------------------------------------------------------------------------- #
def _extend_max_steps(env: RailEnv) -> None:
    ends = max((a.latest_arrival for a in env.agents if a.latest_arrival is not None),
               default=0)
    env._max_episode_steps = max(int(env._max_episode_steps), int(ends) + 200)


def calibrate_timetable(env: RailEnv, slack: int = 1, seed: int = 0,
                        verbose: bool = True) -> dict:
    """Set arrival deadlines to the reference policy's actual normal-condition
    arrivals plus ``slack``. Origin departures and dwell holds are untouched, so
    greedy's behaviour (and thus its arrivals) is unchanged and every train is
    exactly on time under ``normal``; malfunctions under ``delayed`` push
    arrivals past the deadline.
    """
    old_la = [list(a.waypoints_latest_arrival) for a in env.agents]
    _, first = rollout(env, seed, record_first=True)

    shifted = 0
    for h, agent in enumerate(env.agents):
        la = list(agent.waypoints_latest_arrival)
        for j in range(len(la)):
            measured = first.get(h, {}).get(j)
            if measured is None:
                continue
            new = max(old_la[h][j] or 0, int(measured)) + slack
            if new != la[j]:
                shifted += 1
            la[j] = new
        agent.waypoints_latest_arrival = la
        agent.latest_arrival = la[-1]
    _extend_max_steps(env)

    if verbose:
        print(f"  calibrated: {shifted} deadlines shifted (slack={slack})")
    return {"deadlines_shifted": shifted}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def inject_malfunctions(env: RailEnv, rate: float = 1 / 540,
                        duration: tuple[int, int] = (20, 50)) -> None:
    """Attach the ``delayed`` condition's Poisson malfunction generator."""
    env.effects_generator = MalfunctionEffectsGenerator(
        ParamMalfunctionGen(
            MalfunctionParameters(malfunction_rate=rate,
                                  min_duration=duration[0],
                                  max_duration=duration[1])
        )
    )


def report(tag: str, env: RailEnv, seed: int = 0) -> dict:
    arrival, _ = rollout(env, seed)
    values = list(_residual_lateness(env, arrival).values())
    late = np.array(values) if values else np.array([0])
    res = {
        "arrived": len(arrival),
        "n": env.get_num_agents(),
        "min_late": int(late.min()),
        "median_late": float(np.median(late)),
        "worst": int(late.max()),
        "on_time": int((late <= 0).sum()),
    }
    print(
        f"[{tag}] arrived {res['arrived']}/{res['n']} | "
        f"lateness min {res['min_late']} median {res['median_late']:.0f} "
        f"worst {res['worst']} | on-time {res['on_time']}/{res['arrived']}"
    )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slack", type=int, default=1)
    ap.add_argument("--no-fix-targets", action="store_true")
    args = ap.parse_args()

    env = load_env(args.seed)

    bad = 0
    for agent in env.agents:
        ic = agent.initial_configuration
        if ic is None or ic[0] is None:
            continue
        seen = reachable_configs(env, _conf(ic[0], ic[1]))
        if not ({_conf(p, d) for (p, d) in agent.targets} & seen):
            bad += 1
    print(f"agents {env.get_num_agents()} | unreachable targets before: {bad}")

    report("baseline normal", env, args.seed)

    if not args.no_fix_targets:
        remapped = fix_targets(env, verbose=False)
        print(f"fixed unreachable targets: remapped {len(remapped)} agents")
    moved = fix_spawns(env, verbose=False)
    print(f"fixed wrong-way spawns: relocated {len(moved)} agents {moved}")
    calibrate_timetable(env, slack=args.slack, seed=args.seed)

    report("repaired normal", env, args.seed)
    inject_malfunctions(env)
    report("repaired delayed", env, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
