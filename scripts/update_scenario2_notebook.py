#!/usr/bin/env python3
"""Point the Scenario-2 notebook at the adopted 20x20 Y-link level.

Updates cells 7, 8, 9 (layout + segments) and 13/26 (level-free flyover) so the
notebook matches `data/levels/scenario2_env.mpk` (rows 4-7 corridor, Merrylands
at (18,10)/(18,11)) and the rebuilt 126-service timetable.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "Scenario2" / "Scenario2.ipynb"

CELL7 = r"""**Scenario 2 layout (`scenario2_env.mpk`, 20 x 20 grid)**

Harris Park is the west end and Granville the east end; as in Scenario 1, trains running EAST are "Up" (towards Sydney). Four parallel tracks, Main on top, Suburban below -- the real Granville / Harris Park arrangement:

| Row | Track | Direction |
|---|---|---|
| 4 | Up Main | East (Harris Park -> Granville) |
| 5 | Down Main | West (Granville -> Harris Park) |
| 6 | Up Suburban | East |
| 7 | Down Suburban | West |

The **Old Main South** leaves the Suburban pair near Granville (col 12) and runs down to Merrylands; the **Y-link** leaves the Suburban pair near Harris Park (col 5) and joins the same branch -- the real "two tracks of the Y link". Merrylands is a two-platform station at the bottom, (18,10)/(18,11).

| Feature | Cell(s) | Notes |
|---|---|---|
| Old Main South (down) | turns south at (7,12), then col 11 down | Granville Suburban -> Merrylands |
| Y-link (down) | turns south at (7,5), runs east on row 9, joins col 11 | Harris Park Suburban -> Merrylands |
| Up road | col 10 up, then col 9 (to Granville) / row 10 (to Harris Park) | Merrylands -> Harris Park / Granville |
| Flyover | (9,9), (10,9) | Up Old Main South crosses the Y-link -- level-free diamonds |

The two branch directions use **separate columns** (col 11 down, col 10 up), and the crossing is the real grade-separated flyover. Scenario 1 is a **parallel-track rerouting** problem; Scenario 2 is a **junction, merge and branch (conflict and convergence)** problem.

**Real timetable (`scenario2_timetable_20261015.json`, built by `build_scenario2_timetable.py`)**

From the TfNSW Greater Sydney GTFS for Thursday 15 October 2026, 16:00-19:00: every T1/T2/T5 service that traverses a modelled link, whether it stops or merely passes (inferred from the shape geometry). That gives **126 movements**:

| Segment | Line | Down | Up | Total |
|---|---|---|---|---|
| Granville <-> Harris Park | T1 (Main) | 35 | 23 | 58 |
| Granville <-> Harris Park | T2 (Suburban) | 12 | 12 | 24 |
| Granville <-> Merrylands | T2 (Old Main South) | 18 | 12 | 30 |
| Harris Park <-> Merrylands | T5 (Y-link) | 8 | 6 | 14 |

One step is 20 seconds, so step 0 is 16:00 and the window ends at step 540.
"""

CELL8 = r'''# --------------------------------------------------
# Defining directions
# --------------------------------------------------
NORTH = 0
EAST = 1    # Up direction   (towards Granville / Sydney)
SOUTH = 2
WEST = 3    # Down direction (towards Harris Park)

# --------------------------------------------------
# Defining the train stations
# --------------------------------------------------
# Harris Park = west end (col 1), Granville = east end (col 18).
# Four tracks, Main on top: Up Main, Down Main, Up Suburban, Down Suburban.
# Merrylands sits at the bottom of the branch (row 18), one platform per direction.

STATIONS = {
    "Harris Park": {
        "up_main": (4, 1),
        "down_main": (5, 1),
        "up_suburban": (6, 1),
        "down_suburban": (7, 1),
    },
    "Granville": {
        "up_main": (4, 18),
        "down_main": (5, 18),
        "up_suburban": (6, 18),
        "down_suburban": (7, 18),
    },
    "Merrylands": {
        "up": (18, 10),     # northbound (towards Harris Park / Granville)
        "down": (18, 11),   # southbound (arriving from the corridor)
    },
}

# --------------------------------------------------
# Defining the junction
# --------------------------------------------------
JUNCTIONS = {
    "oms_diverge": (7, 12),    # Down Suburban turns south onto the Old Main South
    "ylink_diverge": (7, 5),   # Down Suburban turns south onto the Y-link
    "flyover_a": (9, 9),       # Up Old Main South crosses the Y-link ...
    "flyover_b": (10, 9),      # ... on a grade-separated flyover (level-free)
}

# --------------------------------------------------
# Environment settings
# --------------------------------------------------
# The Up Old Main South crosses the Y-link on a flyover, so those diamonds are
# level-free (the real arrangement is partially grade separated).
# True  -> flyover: the two streams never block each other
# False -> flat crossing: trains wait for each other
LEVEL_FREE_CROSSING = True

# --------------------------------------------------
# Timetable settings (same time scale as Scenario 1)
# --------------------------------------------------
# Find the repository from either its root or this notebook directory.
REPO_ROOT = next(
    root for root in [Path.cwd(), *Path.cwd().parents]
    if (root / "data" / "levels" / "scenario2_env.mpk").is_file()
)
ENV_FILE = REPO_ROOT / "data" / "levels" / "scenario2_env.mpk"
TIMETABLE_FILE = REPO_ROOT / "data" / "schedules" / "scenario2_timetable_20261015.json"   # built by build_scenario2_timetable.py
STEP_SECONDS = 20       # 1 step = 20 seconds
WINDOW_START = "16:00"  # step 0
DEADLINE_SLACK = 1      # steps added to the reference arrival (as in Scenario 1)
EXTRA_STEPS = 200       # steps kept after the last deadline (as in Scenario 1)

# --------------------------------------------------
# Operating conditions (same as Scenario 1)
# --------------------------------------------------
# "normal" is deterministic, "delayed" injects Poisson malfunctions (the paper's 1/540 interval)
CONDITIONS = {
    "normal": None,
    "delayed": 1 / 540,
}
MALFUNCTION_DURATION = (20, 50)   # steps a broken-down train stays stopped
SEEDS = [0, 1, 2]
'''

CELL9 = r'''# --------------------------------------------------
# Defining the trains (or agents) from the real timetable
# --------------------------------------------------
# Each service enters the map at its origin station and leaves at its destination.
# T1 runs on the Main pair (rows 4/5), T2 on the Suburban pair (rows 6/7), and T5
# on the Y-link.  "route" = switch cells where the train has to turn
# (default is MOVE_FORWARD).

SEGMENT_MAP = {
    # --- Main Western, Main pair: T1 (mostly pass-through expresses)
    ("T1", "Harris Park", "Granville"): {
        "start": STATIONS["Harris Park"]["up_main"],
        "target": STATIONS["Granville"]["up_main"],
        "direction": EAST,
    },
    ("T1", "Granville", "Harris Park"): {
        "start": STATIONS["Granville"]["down_main"],
        "target": STATIONS["Harris Park"]["down_main"],
        "direction": WEST,
    },
    # --- Main Western, Suburban pair: T2
    ("T2", "Harris Park", "Granville"): {
        "start": STATIONS["Harris Park"]["up_suburban"],
        "target": STATIONS["Granville"]["up_suburban"],
        "direction": EAST,
    },
    ("T2", "Granville", "Harris Park"): {
        "start": STATIONS["Granville"]["down_suburban"],
        "target": STATIONS["Harris Park"]["down_suburban"],
        "direction": WEST,
    },
    # --- Old Main South: T2 Granville <-> Merrylands
    ("T2", "Granville", "Merrylands"): {
        "start": STATIONS["Granville"]["down_suburban"],
        "target": STATIONS["Merrylands"]["down"],
        "direction": WEST,
        "route": {
            JUNCTIONS["oms_diverge"]: RailEnvActions.MOVE_LEFT,   # (7,12) turn south
            (10, 12): RailEnvActions.MOVE_RIGHT,                  # (10,12) onto col 11
            (10, 11): RailEnvActions.MOVE_LEFT,                   # (10,11) turn south
        },
    },
    ("T2", "Merrylands", "Granville"): {
        "start": STATIONS["Merrylands"]["up"],
        "target": STATIONS["Granville"]["up_suburban"],
        "direction": NORTH,
        "route": {
            (13, 10): RailEnvActions.MOVE_LEFT,    # (13,10) across to col 9
            (13, 9): RailEnvActions.MOVE_RIGHT,    # (13,9)  north
            (8, 9): RailEnvActions.MOVE_RIGHT,     # (8,9)   east onto row 8
            (8, 11): RailEnvActions.MOVE_LEFT,     # (8,11)  north
            (6, 11): RailEnvActions.MOVE_RIGHT,    # (6,11)  merge east into the Up Suburban
        },
    },
    # --- Y-link: T5 Harris Park <-> Merrylands
    ("T5", "Harris Park", "Merrylands"): {
        "start": STATIONS["Harris Park"]["down_suburban"],
        "target": STATIONS["Merrylands"]["down"],
        "direction": EAST,
        "route": {
            JUNCTIONS["ylink_diverge"]: RailEnvActions.MOVE_RIGHT,  # (7,5) turn south
            (9, 5): RailEnvActions.MOVE_LEFT,                        # (9,5) east onto row 9
            (9, 11): RailEnvActions.MOVE_RIGHT,                      # (9,11) onto col 11
        },
    },
    ("T5", "Merrylands", "Harris Park"): {
        "start": STATIONS["Merrylands"]["up"],
        "target": STATIONS["Harris Park"]["down_suburban"],
        "direction": NORTH,
        "route": {
            (10, 10): RailEnvActions.MOVE_LEFT,    # (10,10) west onto row 10
            (10, 3): RailEnvActions.MOVE_RIGHT,    # (10,3)  north
            (7, 3): RailEnvActions.MOVE_LEFT,      # (7,3)   west into Harris Park
        },
    },
}


def to_step(clock, start=WINDOW_START):
    """Clock time 'HH:MM:SS' -> step number after the start of the window."""
    h, m, s = (int(x) for x in clock.split(":"))
    start_h, start_m = (int(x) for x in start.split(":"))
    seconds = (h * 3600 + m * 60 + s) - (start_h * 3600 + start_m * 60)
    return round(seconds / STEP_SECONDS)


def load_timetable_config(path=TIMETABLE_FILE):
    """Turn the extracted GTFS services into the train config used by the functions below."""
    with open(path) as f:
        timetable = json.load(f)

    train_config = []
    for service in timetable["services"]:
        segment = SEGMENT_MAP[(service["line"], service["origin"], service["destination"])]
        train_config.append({
            "line": service["line"],

            "start": segment["start"],
            "target": segment["target"],

            "direction": segment["direction"],
            "departure": to_step(service["departure"]),

            "stops": [],

            "route": segment.get("route", {}),

            # Real times, kept for the results table
            "real_departure": service["departure"][:5],
            "real_arrival": service["arrival"][:5],
        })

    return train_config


REALISTIC_TIMETABLE_CONFIG = load_timetable_config()

print("Trains loaded:", len(REALISTIC_TIMETABLE_CONFIG))
print(Counter(f"{t['real_departure'][:2]}:00 hour" for t in REALISTIC_TIMETABLE_CONFIG))
'''

CELL13 = r'''def load_environment(level_free_crossing=LEVEL_FREE_CROSSING, disruptions=None, malfunction_rate=None):

    if level_free_crossing:
        level_free_positions = [JUNCTIONS["flyover_a"], JUNCTIONS["flyover_b"]]
    else:
        level_free_positions = []

    effects = Scenario2Effects(level_free_positions, disruptions)
'''

CELL26 = r"""The research question for Scenario 2 is whether the agent can manage trains converging around a disruption. Under the delayed condition, trains break down at random (Poisson, rate 1/540, 20-50 steps), as in Scenario 1. The Old Main South (from Granville) and the Y-link (from Harris Park) merge onto the same two-track branch down to Merrylands, and the Up Old Main South crosses the Y-link on a flyover, so a breakdown near the merge can delay both streams. The scripted routes below are the baseline the RL agents have to beat."""


def main() -> None:
    nb = json.loads(NB.read_text())

    def setcode(i, text):
        nb["cells"][i]["source"] = text.splitlines(keepends=True)
        if nb["cells"][i]["cell_type"] == "code":
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None

    setcode(7, CELL7)
    setcode(8, CELL8)
    setcode(9, CELL9)
    setcode(26, CELL26)

    # cell 13: swap only the level-free lines
    src = "".join(nb["cells"][13]["source"]).splitlines(keepends=True)
    out, skip = [], False
    for line in src:
        if "if level_free_crossing:" in line:
            out.append(line); skip = True; continue
        if skip and "level_free_positions = [JUNCTIONS" in line:
            out.append("        level_free_positions = [JUNCTIONS[\"flyover_a\"], JUNCTIONS[\"flyover_b\"]]\n")
            skip = False; continue
        if skip and "level_free_positions = []" in line:
            out.append(line); skip = False; continue
        out.append(line)
    nb["cells"][13]["source"] = out
    nb["cells"][13]["outputs"] = []
    nb["cells"][13]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"updated {NB}")


if __name__ == "__main__":
    main()
