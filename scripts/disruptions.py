"""Deterministic, *avoidable* long disruptions, shared by training and evaluation.

Previous placement was random, so roughly half the failures landed where no train
could get round them and the reflex never had a decision to make.  Here the
failure is deferred until the victim is standing in a window the agent can act on:

    a following train can dodge a blockage at column c on row r only if it can
    change track *before* reaching c, i.e. some lateral move off (r, heading)
    exists at a column in [c - window, c].

Since the destination accepts either same-direction track, the train never has to
cross back - it simply finishes on the other track.  So an "avoidable" incident is
one whose blockage cell sits within `window` cells after a usable crossover.

2-3 incidents per episode, deterministic from the episode seed, durations
lognormal (median 30 min, p90 60 min) scaled by the incident count.
"""

from __future__ import annotations

import math

import numpy as np


def lateral_columns(env) -> dict[tuple[int, int], list[int]]:
    """Columns at which a lateral move is possible, keyed by (row, heading)."""
    table: dict[tuple[int, int], list[int]] = {}
    for row in range(env.height):
        for column in range(env.width):
            if not env.rail.get_full_transitions(row, column):
                continue
            for heading in range(4):
                successors = env.rail.get_successor_configurations(((row, column), heading))
                if any(int(p[0]) != row for p, _ in successors):
                    table.setdefault((row, heading), []).append(column)
    return table


def _avoidable(row, column, heading, table, window) -> bool:
    columns = table.get((row, heading))
    if not columns:
        return False
    before = [c for c in columns if c <= column]
    if not before:
        return False
    return (column - max(before)) <= window


def station_stops(agent, station_cells) -> list[tuple[int, int]]:
    """Platform cells in this agent's own itinerary that are in `station_cells`.

    Only the intermediate waypoint groups count: the first group is the origin
    and the last is the target, so a failure there is not an en-route delay.
    """
    hits = []
    for j in range(1, len(agent.waypoints) - 1):
        for wp in agent.waypoints[j]:
            pos = (int(wp.position[0]), int(wp.position[1]))
            if pos in station_cells:
                hits.append(pos)
    return hits


def plan_for(seed: int, env, incidents: int = 2, eligible=None, dirs=None) -> list[dict]:
    """Pick victims and durations; firing is deferred to an avoidable position.

    ``dirs``: handle -> "up"/"down". When given and at least two incidents are
    requested, one victim is taken from each direction before the rest are
    filled, so an episode is never all one-way.
    """
    rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))
    handles = [a.handle for a in env.agents]
    if eligible is not None:
        handles = [h for h in handles if h in eligible]
    sigma = math.log(180 / 90) / 1.2816
    scale = 1.0 if incidents <= 2 else 0.75  # keep total disruption volume similar
    chosen = []

    picks = list(rng.permutation(handles))
    if dirs and incidents >= 2:
        up = [h for h in picks if dirs.get(h) == "up"]
        down = [h for h in picks if dirs.get(h) == "down"]
        other = [h for h in picks if dirs.get(h) not in ("up", "down")]
        balanced = []
        if up and down:
            balanced = [up.pop(0), down.pop(0)]
        picks = balanced + up + down + other
    for handle in picks[: max(1, incidents)]:
        duration = int(max(10, min(250, round(90 * scale * math.exp(sigma * rng.standard_normal())))))
        chosen.append({"handle": int(handle), "duration": duration, "done": False})
    return chosen


def install(env, incidents: int = 2, window: int = 12,
            station_cells=None, station_commit=None, require_follower=True,
            up_rows=None, avoid_cells=None, min_progress: int = 0,
            fire_rows=None, fire_cols=None, fire_gap: int = 40,
            fire_cells=None, balance_dirs: bool = False,
            duration: int | None = None, sample_cells: bool = False):
    """Fire each planned failure at a station platform, in front of the crossover.

    ``station_cells``: set of (row, col) platform cells where a failure is
    avoidable - a following train on the same track can still reach the other
    same-direction track at ``station_commit[(row, col)]``, which is strictly
    upstream of the platform.  When given, a victim only fails once it is
    STOPPED at one of those cells, i.e. during its dwell at the platform, so the
    escape crossover is never the cell that gets blocked.

    Without ``station_cells`` the legacy behaviour applies: fire the first time
    the victim sits within ``window`` cells after any lateral move (which can,
    and in practice does, put the failure on the crossover itself).
    """
    state = {"plan": [], "table": lateral_columns(env), "window": window,
             "station_cells": set(station_cells) if station_cells else None,
             "commit": dict(station_commit or {}), "fired": [], "missed": [],
             "avoid": set(avoid_cells or ()),
             "up_rows": set(up_rows or ()),
             "min_progress": int(min_progress),
             "fire_rows": set(fire_rows) if fire_rows else None,
             "fire_cols": tuple(fire_cols) if fire_cols else None,
             "fire_cells": set(fire_cells) if fire_cells else None,
             "band": set(fire_cells) if fire_cells else None,
             "sample_cells": bool(sample_cells),
             "balance_dirs": bool(balance_dirs),
             "fire_gap": int(fire_gap), "last_fire": -10 ** 6,
             "durations": [], "zone_fired": set(),
             "require_follower": bool(require_follower and station_cells)}

    def follower_upstream(victim, row, column, commit_col, heading) -> bool:
        """Is another train on this row still upstream of the escape crossover?

        A failure only counts as avoidable if some follower could still take the
        crossover. Firing when the queue has already passed it produces an
        incident no policy can act on.
        """
        eastbound = heading == 1
        for other in env.agents:
            if other is victim or other.handle == victim.handle:
                continue
            cfg = other.current_configuration
            if cfg is None or cfg[0] is None or not other.state.is_on_map_state():
                continue
            orow, ocol = int(cfg[0][0]), int(cfg[0][1])
            if orow != row:
                continue
            if eastbound and ocol < commit_col:
                return True
            if not eastbound and ocol > commit_col:
                return True
        return False
    original_reset = env.reset
    original_step = env.step

    def reset(*args, **kwargs):
        out = original_reset(*args, **kwargs)
        seed = kwargs.get("random_seed")
        if seed is None:
            seed = getattr(env, "random_seed", 0) or 0
        # a fixed zone picks its own victims: the first trains to arrive there
        if state["fire_rows"] is not None or state["fire_cells"] is not None:
            rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))
            sigma = math.log(180 / 90) / 1.2816
            if state["sample_cells"] and state["band"]:
                # sample from the PRISTINE band, not from the previous sample:
                # overwriting it made the pool shrink on every reset (one env is
                # reused for every eval seed), so later episodes fired fewer
                # incidents and the "delayed" score drifted to the normal one
                up = sorted(c for c in state["band"] if c[0] in state["up_rows"])
                down = sorted(c for c in state["band"] if c[0] not in state["up_rows"])
                rng.shuffle(up); rng.shuffle(down)
                chosen, pools = [], [up, down]
                i = 0
                while len(chosen) < incidents and (up or down):
                    pool = pools[i % 2] if pools[i % 2] else pools[(i + 1) % 2]
                    if pool:
                        chosen.append(pool.pop())
                    i += 1
                state["fire_cells"] = set(chosen)

            if duration is not None:
                # a fixed hold makes the delay consistent: every victim loses the
                # same number of steps, so no seed can swallow the failure inside
                # its own schedule slack
                state["durations"] = [int(duration)] * max(1, incidents)
            else:
                state["durations"] = [
                    int(max(10, min(220, round(90 * math.exp(sigma * rng.standard_normal())))))
                    for _ in range(max(1, incidents))
                ]
            state["zone_fired"] = set()
            state["last_fire"] = -10 ** 6

        # remember where each train started so min_progress can be measured
        state["start"] = {}
        for agent in env.agents:
            cfg = agent.initial_configuration
            if cfg is not None and cfg[0] is not None:
                state["start"][agent.handle] = (int(cfg[0][0]), int(cfg[0][1]))

        eligible, dirs = None, None
        if state["station_cells"]:
            # only trains that actually call at a platform the agent can divert
            # around - otherwise the episode is silently unavoidable
            state["stops"] = {}
            for agent in env.agents:
                hits = station_stops(agent, state["station_cells"])
                if hits:
                    state["stops"][agent.handle] = hits
            eligible = set(state["stops"])
            if state["up_rows"]:
                dirs = {}
                for handle, stops in state["stops"].items():
                    row = stops[0][0]
                    dirs[handle] = "up" if row in state["up_rows"] else "down"
        state["plan"] = plan_for(seed, env, incidents, eligible=eligible, dirs=dirs)
        return out

    def on_map(agent) -> bool:
        return agent.current_configuration is not None and agent.state.is_on_map_state()

    def in_zone(row, column) -> bool:
        if state["fire_cells"] is not None:
            return (row, column) in state["fire_cells"]
        if state["fire_rows"] is not None and row not in state["fire_rows"]:
            return False
        if state["fire_cols"] is not None:
            lo, hi = state["fire_cols"]
            if not (lo <= column <= hi):
                return False
        return True

    def direction(row) -> str:
        return "up" if row in state["up_rows"] else "down"

    def fire_zone(*args, **kwargs):
        """Fire the first trains that reach the zone, `fire_gap` steps apart.

        This is the reliable counterpart to the reactive platform rule: instead
        of waiting for one particular victim to dwell somewhere, any train that
        reaches the zone is a candidate, so the incidents actually happen (the
        reactive rule was measured firing 0-1 of 3 on some seeds, which silently
        turned the "delayed" evaluation into a normal-condition one).
        """
        now = int(env._elapsed_steps)
        if now - state["last_fire"] < state["fire_gap"]:
            return
        if len(state["zone_fired"]) >= len(state["durations"]):
            return
        cands = []
        for agent in env.agents:
            if agent.handle in state["zone_fired"] or not on_map(agent):
                continue
            row = int(agent.current_configuration[0][0])
            column = int(agent.current_configuration[0][1])
            if not in_zone(row, column):
                continue
            if state["avoid"] and (row, column) in state["avoid"]:
                continue
            cands.append((agent, row, column))
        if not cands:
            return

        if state["balance_dirs"] and state["up_rows"] and len(state["durations"]) >= 2:
            counts = {"up": 0, "down": 0}
            for f in state["fired"]:
                counts[direction(f[1])] += 1
            want = "up" if counts["up"] <= counts["down"] else "down"
            preferred = [c for c in cands if direction(c[1]) == want]
            if preferred:
                cands = preferred
            elif now - state["last_fire"] < 2 * state["fire_gap"]:
                # hold out for the other direction, but not indefinitely: a
                # third same-direction failure is better than firing nothing
                return

        agent, row, column = cands[0]
        # a cell hosts at most one failure, so the 2-3 incidents spread over the
        # band instead of stacking on the first spot that has traffic
        if state["fire_cells"] is not None:
            state["fire_cells"].discard((row, column))
        idx = len(state["zone_fired"])
        agent.malfunction_handler.malfunction_down_counter = state["durations"][idx]
        state["zone_fired"].add(agent.handle)
        state["last_fire"] = now
        state["fired"].append((int(agent.handle), row, column, now))

    def step(*args, **kwargs):
        if state["fire_rows"] is not None or state["fire_cells"] is not None:
            fire_zone(*args, **kwargs)
            return original_step(*args, **kwargs)
        for item in state["plan"]:
            if item["done"]:
                continue
            agent = env.agents[item["handle"]]
            if not on_map(agent):
                continue
            row = int(agent.current_configuration[0][0])
            column = int(agent.current_configuration[0][1])
            heading = int(agent.current_configuration[1])
            if state["avoid"] and (row, column) in state["avoid"]:
                continue  # keep the failure away from these cells
            if state["min_progress"]:
                start = state["start"].get(agent.handle)
                if start is None or (abs(row - start[0]) + abs(column - start[1])
                                     < state["min_progress"]):
                    continue  # not far enough from origin yet

            if state["station_cells"] is not None:
                from flatland.envs.step_utils.states import TrainState

                if (row, column) not in state["station_cells"]:
                    continue
                if agent.state != TrainState.STOPPED:
                    continue  # only while dwelling at the platform
                commit_col = state["commit"].get((row, column))
                if (state["require_follower"] and commit_col is not None
                        and not follower_upstream(agent, row, column, commit_col, heading)):
                    if item not in state["missed"]:
                        state["missed"].append(item)
                    continue  # nobody could still divert - wait for a better dwell
            elif not _avoidable(row, column, heading, state["table"], state["window"]):
                continue  # wait until the blockage would actually be dodgeable
            agent.malfunction_handler.malfunction_down_counter = item["duration"]
            item["done"] = True
            item["fired_row"] = row
            item["fired_col"] = column
            item["commit_col"] = state["commit"].get((row, column))
            item["direction"] = ("up" if row in state["up_rows"] else "down"
                                 if state["up_rows"] else ("E" if heading == 1 else "W"))
            item["fired_step"] = int(env._elapsed_steps)
            state["fired"].append((int(item["handle"]), row, column, item["fired_step"]))
        return original_step(*args, **kwargs)

    env.reset = reset
    env.step = step
    return state
