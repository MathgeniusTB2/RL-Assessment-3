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
            duration: int | None = None, sample_cells: bool = False):
    """Install the disruption zone on a raw ``RailEnv`` and return its state.

    ``fire_cells``     cells that host incidents (required)
    ``fire_gap``       minimum steps between two incidents
    ``up_rows``        rows that count as "up" for direction balancing
    ``balance_dirs``   prefer the direction that has fired less often
    ``avoid_cells``    never fire on these cells (e.g. station platforms)
    ``duration``       fixed hold length; ``None`` draws lognormal (median 90)
    ``sample_cells``   re-draw which cells host the incidents on every reset
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
