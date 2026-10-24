"""Deterministic, avoidable, long disruptions, shared by training and evaluation.

Placement is **zone-driven**: the scenario supplies a fixed set of cells and the
first trains to reach them become victims, ``fire_gap`` steps apart. Every incident
therefore actually happens, whereas the earlier "wait for one chosen victim to
reach a particular state" rule was measured firing 0-1 of 3 on some seeds and
silently turning the "delayed" evaluation into a normal-condition one.

On the compact map the zone is the band strictly between each stop's escape
crossover and its platform (``regimes.compact_fire_band``): a failure there is
avoidable, because a train upstream of the crossover can still take the other
track, and it never blocks the crossover itself. On S2 it is a mid-block band on
the third track row.

Each incident is a **fixed-length hold**, so no seed can swallow it inside its own
schedule slack. Which cells host the incidents is drawn from the episode seed
(direction-balanced), and a cell hosts at most one failure, so the incidents
spread over the band instead of stacking on the first spot with traffic.
"""

from __future__ import annotations

import math

import numpy as np


def install(env, incidents: int = 2, up_rows=None, avoid_cells=None,
            fire_cells=None, fire_gap: int = 40, balance_dirs: bool = False,
            duration: int | None = None, sample_cells: bool = False,
            approach=None, require_approach: bool = False,
            pool_size: int | None = None):
    """Install the disruption zone on a raw ``RailEnv`` and return its state.

    ``fire_cells``     cells that host incidents (required)
    ``fire_gap``       minimum steps between two incidents
    ``up_rows``        rows that count as "up" for direction balancing
    ``balance_dirs``   prefer the direction that has fired less often
    ``avoid_cells``    never fire on these cells (e.g. station platforms)
    ``duration``       fixed hold length; ``None`` draws lognormal (median 90)
    ``sample_cells``   re-draw which cells host the incidents on every reset
    ``approach``       {cell: (commit_column, "up"|"down")} for the crossing band
    ``require_approach``  fire only where a train can still reach its crossover
    ``pool_size``      draw this many candidate cells per episode (default
                       ``incidents``); the spares keep a slot alive when its
                       first choice is never approachable
    """
    if not fire_cells:
        raise ValueError("install() needs a non-empty fire_cells zone")

    state = {
        "fired": [],
        "zone_fired": set(),
        "avoid": set(avoid_cells or ()),
        "up_rows": set(up_rows or ()),
        "fire_cells": set(fire_cells),
        # the pristine zone: re-sampling must draw from THIS, not from the
        # already-sampled set, or the pool shrinks on every reset and later
        # episodes fire fewer incidents (one env is reused for every eval seed)
        "band": set(fire_cells),
        "sample_cells": bool(sample_cells),
        "approach": dict(approach or {}),
        "require_approach": bool(require_approach),
        "balance_dirs": bool(balance_dirs),
        "fire_gap": int(fire_gap),
        "last_fire": -10 ** 6,
        "durations": [],
    }

    def on_map(agent) -> bool:
        return agent.current_configuration is not None and agent.state.is_on_map_state()

    def in_zone(row, column) -> bool:
        return (row, column) in state["fire_cells"]

    def direction(row) -> str:
        return "up" if row in state["up_rows"] else "down"

    def position_of(agent):
        cfg = agent.current_configuration
        if cfg is None or cfg[0] is None:
            return None
        return int(cfg[0][0]), int(cfg[0][1])

    def has_approach(row, column, exclude=None) -> bool:
        """Is another train still on the approach side of this cell's crossover?

        A failure inside the crossing band is only avoidable if somebody can
        still reach the commit crossover before the blockage queue backs up over
        it.  Without this test an incident fires wherever a train happens to be,
        which on a dense corridor is usually a spot every follower has already
        passed, so the policy has no decision to make at all.
        """
        info = state["approach"].get((int(row), int(column)))
        if info is None:
            return True
        commit, dirn = int(info[0]), info[1]
        for other in env.agents:
            if other.handle == exclude or not on_map(other):
                continue
            point = position_of(other)
            if point is None or point[0] != row:
                continue
            if (dirn == "up" and point[1] < commit) or (dirn == "down" and point[1] > commit):
                return True
        return False

    def fire_zone(*args, **kwargs):
        """Hold the first train that reaches an unfired zone cell."""
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
        if state["require_approach"]:
            cands = [c for c in cands if has_approach(c[1], c[2], exclude=c[0].handle)]
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
        # a cell hosts at most one failure, so the incidents spread over the band
        state["fire_cells"].discard((row, column))
        idx = len(state["zone_fired"])
        agent.malfunction_handler.malfunction_down_counter = state["durations"][idx]
        state["zone_fired"].add(agent.handle)
        state["last_fire"] = now
        state["fired"].append((int(agent.handle), row, column, now))

    original_reset = env.reset
    original_step = env.step

    def reset(*args, **kwargs):
        out = original_reset(*args, **kwargs)
        seed = kwargs.get("random_seed")
        if seed is None:
            seed = getattr(env, "random_seed", 0) or 0
        rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))

        if state["sample_cells"]:
            # draw one cell per direction first, so an episode is never all one-way
            up = sorted(c for c in state["band"] if c[0] in state["up_rows"])
            down = sorted(c for c in state["band"] if c[0] not in state["up_rows"])
            rng.shuffle(up)
            rng.shuffle(down)
            chosen, pools = [], [up, down]
            draw = max(int(incidents), int(pool_size or incidents))
            i = 0
            while len(chosen) < draw and (up or down):
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
            sigma = math.log(180 / 90) / 1.2816
            state["durations"] = [
                int(max(10, min(220, round(90 * math.exp(sigma * rng.standard_normal())))))
                for _ in range(max(1, incidents))
            ]
        state["zone_fired"] = set()
        state["last_fire"] = -10 ** 6
        # `fired` drives the direction-balance preference, so it must be cleared
        # too: leaving it to accumulate made a seed's episode depend on every
        # episode before it in the process, which broke the paired comparison
        # (the same baseline seed scored differently in two arm files).
        state["fired"] = []
        return out

    def step(*args, **kwargs):
        fire_zone(*args, **kwargs)
        return original_step(*args, **kwargs)

    env.reset = reset
    env.step = step
    return state

# --------------------------------------------------------------- exogenous blockage
# The train-borne failures above fire only when a train reaches the chosen cell, so
# a policy that reroutes early can make the incident vanish: measured, a rerouting
# checkpoint triggered 0 of 2 incidents and its "delayed" score was identical to the
# clean episode. A blockage therefore removes a cell's rail transitions outright, so
# it exists whether or not traffic goes near it and both arms face the same cells at
# the same steps. Activation is deferred while a cell is occupied, because zeroing
# the transitions under a standing train would strand it permanently.

MAX_DEFER = 60


def install_blockage(env, cells, approach=None, incidents: int = 2, duration: int = 400,
                     up_rows=None, first=(30, 90), gap=(30, 60),
                     balance_rows: bool = False, replace: bool | None = None,
                     seed: int | None = None):
    """Install exogenous blockages drawn from ``cells`` on every reset."""
    if not cells:
        raise ValueError("install_blockage needs a non-empty cell pool")
    up_rows = set(up_rows or ())
    pool = list(cells)
    if replace is None:
        replace = int(incidents) > len(pool)
    state = {
        "fired": [],
        "approach": dict(approach or {}),
        "active": {},
        "plan": [],
        "duration": int(duration),
        "incidents": int(incidents),
        "pool": pool,
    }
    env._blocked_cells = set()

    def occupied(cell) -> bool:
        for agent in env.agents:
            configuration = agent.current_configuration
            if configuration is None or configuration[0] is None:
                continue
            if (int(configuration[0][0]), int(configuration[0][1])) == cell:
                return True
        return False

    def release(cell) -> None:
        info = state["active"].pop(cell, None)
        if info is not None:
            env.rail.set_transitions(cell, info["trans"])
        env._blocked_cells = set(state["active"])

    original_reset = env.reset
    original_step = env.step

    def reset(*args, **kwargs):
        out = original_reset(*args, **kwargs)
        for cell in list(state["active"]):
            release(cell)
        state["fired"].clear()
        state["plan"].clear()

        episode_seed = kwargs.get("random_seed")
        if episode_seed is None:
            episode_seed = seed if seed is not None else getattr(env, "random_seed", 0) or 0
        rng = np.random.default_rng(abs(int(episode_seed)) % (2 ** 31))

        by_row = {}
        for cell in pool:
            by_row.setdefault(cell[0], []).append(cell)
        from_row = {row: list(cell_list) for row, cell_list in by_row.items()}
        for row_cells in by_row.values():
            rng.shuffle(row_cells)
        up_pool = sorted(r for r in by_row if r in up_rows)
        down_pool = sorted(r for r in by_row if r not in up_rows)

        def take(rows):
            live = [r for r in rows if by_row[r]]
            if not live:
                if not replace:
                    return None
                live = [r for r in rows if from_row.get(r)]
                if not live:
                    return None
            if balance_rows:
                row = live[int(rng.integers(0, len(live)))]
            else:
                row = live[0]
            if replace and not by_row[row]:
                candidates = from_row[row]
                return candidates[int(rng.integers(0, len(candidates)))]
            return by_row[row].pop()

        chosen = []
        for index in range(max(1, state["incidents"])):
            want_up = index % 2 == 0
            cell = take(up_pool if want_up else down_pool)
            if cell is None:
                cell = take(down_pool if want_up else up_pool)
            if cell is not None:
                chosen.append(cell)

        when = int(rng.integers(first[0], first[1] + 1))
        for index, cell in enumerate(chosen):
            state["plan"].append((cell, when + index * int(rng.integers(gap[0], gap[1] + 1))))
        return out

    def step(*args, **kwargs):
        now = int(env._elapsed_steps)
        for cell, scheduled in list(state["plan"]):
            if now < scheduled:
                continue
            if occupied(cell):
                if now - scheduled >= MAX_DEFER:
                    state["plan"].remove((cell, scheduled))  # never cleared; drop it
                continue
            saved = env.rail.get_full_transitions(cell[0], cell[1])
            env.rail.set_transitions(cell, 0)
            state["active"][cell] = {"trans": saved, "release": now + state["duration"]}
            state["plan"].remove((cell, scheduled))
            state["fired"].append((len(state["fired"]), cell[0], cell[1], now))
            env._blocked_cells = set(state["active"])
        for cell, info in list(state["active"].items()):
            if now >= info["release"]:
                release(cell)
        return original_step(*args, **kwargs)

    env.reset = reset
    env.step = step
    return state


def active_cells(env):
    """Cells currently blocked on this env (used by the observation builder)."""
    return getattr(env, "_blocked_cells", set())
