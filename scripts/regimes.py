"""Shared disruption regimes, one per scenario.

Training, paired evaluation and the render scripts must install the SAME regime,
otherwise the curve and the baseline measure different scenarios (that mistake
already produced three false positives in this project).

    network-simple-compact  failure fires while the victim dwells at a platform
                            whose (station, track) has a same-direction escape
                            (the 7 stop types found by probe_station_access);
                            incidents are drawn one per direction when >= 2
    granville-harris-park   failure fires mid-route in a fixed zone - third track
                            row, column 14 - never on a station cell, because S2
                            services have no intermediate stops

Both spellings of the scenario name are accepted; anything else falls back to
the legacy mid-corridor placement.
"""

from __future__ import annotations

import json
from pathlib import Path

import disruptions

REPO = Path(__file__).resolve().parents[1]

COMPACT_ROWS = {2: "up_suburban", 3: "down_suburban", 4: "up_local", 5: "down_local"}
COMPACT_UP_ROWS = {row for row, track in COMPACT_ROWS.items() if track.startswith("up")}

S2_STATION_CELLS = {
    (4, 1), (5, 1), (6, 1), (7, 1),
    (4, 18), (5, 18), (6, 18), (7, 18),
    (18, 10), (18, 11),
}
# third track row, mid-block just west of Granville; a band of cells rather than
# one spot, so the seed changes WHERE the failure lands while keeping it in the
# same stretch of running line
S2_FIRE_ROWS = {6}
S2_FIRE_COLS = (11, 16)
S2_FIRE_CELLS = {(6, c) for c in range(S2_FIRE_COLS[0], S2_FIRE_COLS[1] + 1)}
# fixed hold length: the schedule slack is only a few steps, so any longer hold
# guarantees the victim is late on every episode (a lognormal draw could land
# inside the slack and produce a seed with no delay at all)
# long holds: 5x the original 90-step hold. At 90 the queue often cleared inside
# the scheduled dwell and the incident barely moved the score; 450 steps is ~45 %
# of a compact episode, so every failure leaves a standing queue behind it.
DURATION = 450


def compact_geometry():
    """(platform cells, commit crossover column) from the accessibility probe."""
    data = json.load(open(REPO / "results" / "station_access_compact.json"))
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


def is_compact(scenario: str) -> bool:
    return scenario in ("network-simple-compact", "network-simple_compact", "compact")


def is_s2(scenario: str) -> bool:
    return scenario in ("granville-harris-park", "granville_harris_park", "s2", "scenario2")


def install(env, scenario: str, incidents: int = 2, require_follower: bool = False):
    """Install the scenario's disruption regime on a raw RailEnv.

    ``require_follower`` only applies to the compact map: when True a failure
    waits for a train that could still take the crossover. That is the stricter,
    more faithful rule but it fires far less often (measured: 0-2 of 2-3 per
    episode), so training uses the platform-only rule to keep the signal.
    """
    if is_compact(scenario):
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
            # with a 450-step hold the junction jams within a few steps of the
            # first failure, so all three have to land almost at once (gap 1);
            # otherwise no train ever reaches the third cell and it never fires
            fire_cells=S2_FIRE_CELLS, fire_gap=1, duration=DURATION,
            sample_cells=True,
        )
    return disruptions.install(env, incidents=incidents)
