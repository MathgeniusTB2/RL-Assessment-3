"""Shared disruption regimes, one per scenario.

Training, paired evaluation and the notebook must install the SAME regime,
otherwise the curve and the baseline measure different scenarios (that mistake
already produced three false positives in this project).

    network-simple-compact  failures land in the band strictly between each
                            stop's escape crossover and its platform (101 cells,
                            from the station-access probe), so every incident is
                            avoidable and never blocks the crossover itself;
                            direction-balanced, sampled per episode
    granville-harris-park   failures land mid-block on the third track row,
                            columns 11-16, never on a station cell, because S2
                            services have no intermediate stops

Three fixed holds per episode on both maps (see ``DURATION`` / ``DURATION_CROSS``).

Two S1 variants, selected by the ``variant=`` argument or the
``RL_A3_REGIME`` environment variable:

``band``      the default: a 220-step train-borne failure in the band between a
              crossover and its platform (measured: never avoidable in practice,
              because every follower has already passed the crossover)
``crossing``  a 400-step train-borne failure in the first cells after a
              crossover, gated on a train still being on the approach side
``block``     (recommended) a 400-step *exogenous* blockage - the cell's rail
              transitions are zeroed - drawn from the same near-crossover cells
              on all three switchable rows. It fires whether or not traffic goes
              near it, so the delayed condition cannot silently collapse into the
              normal one, and because the pool spans both rows of the up pair the
              right crossing depends on what is actually blocked.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import disruptions

REPO = Path(__file__).resolve().parents[1]

COMPACT_ROWS = {2: "up_suburban", 3: "down_suburban", 4: "up_local", 5: "down_local"}
COMPACT_UP_ROWS = {row for row, track in COMPACT_ROWS.items() if track.startswith("up")}

# --- commit crossovers, the only cells with a lateral move -------------------
# From results/station_access_compact.json (the route-aware accessibility probe):
# (row, column) -> the escape crossover a train can take *before* this cell.
COMMIT_CELLS = frozenset({(2, 6), (2, 25), (2, 62), (4, 51), (5, 4), (5, 22), (5, 59)})
UP_ROWS = COMPACT_UP_ROWS


def next_commit_ahead(row: int, column: int, heading: int, horizon: int = 50):
    """Distance to the next commit crossover on this row, ahead of ``column``."""
    step = 1 if heading == 1 else -1 if heading == 3 else 0
    if step == 0:
        return None
    for offset in range(1, horizon + 1):
        col = column + step * offset
        if col < 0:
            return None
        if (row, col) in COMMIT_CELLS:
            return offset
    return None


S2_STATION_CELLS = {
    (4, 1), (5, 1), (6, 1), (7, 1),
    (4, 18), (5, 18), (6, 18), (7, 18),
    (18, 10), (18, 11),
}
# third track row, mid-block just west of Granville; a band of cells rather than
# one spot, so the seed changes WHERE the failure lands while keeping it in the
# same stretch of running line
S2_FIRE_CELLS = {(6, c) for c in range(11, 17)}
# fixed hold length: the schedule slack is only a few steps, so any longer hold
# guarantees the victim is late on every episode (a lognormal draw could land
# inside the slack and produce a seed with no delay at all).
# 220 steps is the longest hold that still leaves every train enough time to
# arrive: measured over 30 seeds, baseline completion is 1.000 on both maps
# (delay_all 2.9k corridor, 16.8k junction). At 300+ the junction deadlocks and
# completion falls to 0.6-0.8, which would score cancellations instead of delay.
DURATION = 220

# --- crossing regime (S1 only) -------------------------------------------
# The band above spans the whole stretch between a crossover and its platform,
# but a failure deep inside it is only avoidable in principle: measured on
# seeds 3000-3002, 0 of 9 incidents had any train still upstream of the commit,
# so the "delayed" condition contained no diversion decision at all. A failure
# just after the commit does: 92% of trigger steps at lead<=2 and ~60% at
# lead<=6 have a follower on the approach side of the crossover.
CROSS_MIN_LEAD = 2       # cells after the commit crossover
CROSS_MAX_LEAD = 8
CROSS_POOL = 8           # candidate cells drawn per episode (4 up + 4 down)
CROSS_FIRE_GAP = 40      # minimum spacing; the hold is long enough that the
                         # incidents still overlap heavily
# Only the up_sub/up_local and down_local/down_sub pairs can be switched, and the
# two up rows cross at different crossovers (Redfern 25 -> row 4, Central 51 ->
# row 2). Blocking BOTH up rows would send a diverted train straight into the
# other failure, so the crossing regime fires on row 2 (escape to row 4) and
# row 5 (escape to row 3) only; whichever row is blocked, the alternative track
# is clear all the way to the next waypoint.
CROSS_ROWS = (2, 5)
# the exogenous variant draws from every switchable row, so over-diverting can be
# punished: a block on row 4 just after Central's commit catches a train that
# changed track at Burwood to avoid a block on row 2
BLOCK_ROWS = (2, 4, 5)
BLOCK_DURATION = 400
BLOCK_FIRST = (30, 90)   # activation window of the first blockage (steps)
BLOCK_GAP = (30, 60)     # spacing to the second
DURATION_CROSS = 400     # ~40% of a 999-step episode: the queue that waits
                         # loses far more than the 2-cell detour premium
# default regime for the compact corridor: the reported exogenous blockage.
# "band" (the retired train-borne band) and "crossing" are still reachable with
# variant=... or RL_A3_REGIME=..., so a bare training call reproduces the report
# instead of quietly training the retired regime.
VARIANT = "block"


def selected_variant(variant: str | None = None) -> str:
    return variant or os.environ.get("RL_A3_REGIME") or VARIANT


def compact_geometry():
    """(platform cells, commit crossover column) from the accessibility probe."""
    path = REPO / "results" / "station_access_compact.json"
    with path.open() as fh:
        data = json.load(fh)
    cells, commit = set(), {}
    for row in data:
        if not row.get("alt_available") or row.get("commit_from_col") is None:
            continue
        pos = (int(row["row"]), int(row["station_col"]))
        cells.add(pos)
        commit[pos] = int(row["commit_from_col"])
    return cells, commit


def compact_fire_band():
    """Every cell strictly between the escape crossover and the platform.

    A failure anywhere in this band is still avoidable: a train upstream of the
    crossover can take the other track before reaching it. Excluding the
    crossover cell itself matters - a failure ON the crossover blocks the escape
    - and excluding the platform keeps it a running-line blockage rather than a
    dwell failure at a station.
    """
    _cells, commit = compact_geometry()
    band = set()
    for (row, col), commit_col in commit.items():
        lo, hi = sorted((col, int(commit_col)))
        for c in range(lo + 1, hi):  # strictly between, both ends excluded
            band.add((row, c))
    return band


def compact_crossing_band(min_lead: int = CROSS_MIN_LEAD, max_lead: int = CROSS_MAX_LEAD):
    """{cell: (commit_column, direction)} for the cells just past each crossover.

    Only the first few cells of each station's band are used. That is where a
    train upstream of the commit can still reach the crossover before the
    blockage queue backs up over it, and where the alternative track rejoins
    the same stop/platform group, so a diverted train still reaches its next
    goal (the waypoint groups already accept both same-direction platforms).
    """
    _cells, commit = compact_geometry()
    band = {}
    for (row, col), commit_col in commit.items():
        commit_col, col = int(commit_col), int(col)
        up = row in COMPACT_UP_ROWS
        for lead in range(int(min_lead), int(max_lead) + 1):
            c = commit_col + lead if up else commit_col - lead
            if (up and commit_col < c < col) or ((not up) and col < c < commit_col):
                band[(row, c)] = (commit_col, "up" if up else "down")
    return band


def is_compact(scenario: str) -> bool:
    return scenario in ("network-simple-compact", "network-simple_compact", "compact")


def is_s2(scenario: str) -> bool:
    return scenario in ("granville-harris-park", "granville_harris_park", "s2", "scenario2")


def install(env, scenario: str, incidents: int = 2, variant: str | None = None):
    """Install the scenario's disruption regime on a raw RailEnv."""
    if is_compact(scenario):
        chosen = selected_variant(variant)
        if chosen in ("none", "clean"):
            # control condition: no disruption installed at all
            return {}
        if chosen == "block":
            rows = tuple(int(x) for x in (
                os.environ.get("RL_A3_BLOCK_ROWS") or ",".join(map(str, BLOCK_ROWS))
            ).split(","))
            band = {c: v for c, v in compact_crossing_band().items() if c[0] in rows}
            def _pair(name, default):
                raw = os.environ.get(name)
                if not raw:
                    return default
                lo, hi = (int(x) for x in raw.split(","))
                return (lo, hi)

            return disruptions.install_blockage(
                env, cells=sorted(band), approach=band, incidents=incidents,
                duration=int(os.environ.get("RL_A3_BLOCK_DURATION", BLOCK_DURATION)),
                up_rows=COMPACT_UP_ROWS,
                first=_pair("RL_A3_BLOCK_FIRST", BLOCK_FIRST),
                gap=_pair("RL_A3_BLOCK_GAP", BLOCK_GAP),
                balance_rows=True,
            )
        if chosen == "crossing":
            # failures land just past a commit crossover and only fire when a
            # train is still on the approach side, so every incident is a real
            # crossing decision; the hold lasts most of the episode
            band = {c: v for c, v in compact_crossing_band().items() if c[0] in CROSS_ROWS}
            return disruptions.install(
                env, incidents=incidents, fire_cells=set(band), approach=band,
                require_approach=True, pool_size=CROSS_POOL, fire_gap=CROSS_FIRE_GAP,
                up_rows=COMPACT_UP_ROWS, balance_dirs=True,
                duration=DURATION_CROSS, sample_cells=True,
            )
        # a band of avoidable cells between each escape crossover and its
        # platform, so failures land in many places but never on the crossover
        # or at a station; zone-driven so they actually fire every episode
        return disruptions.install(
            env, incidents=incidents, fire_cells=compact_fire_band(), fire_gap=40,
            up_rows=COMPACT_UP_ROWS, balance_dirs=True, duration=DURATION,
            sample_cells=True,
        )
    if is_s2(scenario):
        return disruptions.install(
            env, incidents=incidents, avoid_cells=S2_STATION_CELLS,
            # with a long hold the junction jams within a few steps of the
            # first failure, so all three have to land almost at once (gap 1);
            # otherwise no train ever reaches the third cell and it never fires
            fire_cells=S2_FIRE_CELLS, fire_gap=1, duration=DURATION,
            sample_cells=True,
        )
    raise ValueError(f"no disruption regime for scenario {scenario!r}")
