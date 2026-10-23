"""Build the simple 4-track scenario from the trimmed level + the real timetable.

Mapping derived from the level metadata and the timetable's own track codes:

    col  1 Strathfield | 14 Burwood | 40 Redfern | 84 Central   (west -> east)
    row  2 up_suburban |  4 down_suburban |  6 up_local |  8 down_local
    row  3 = bay at Strathfield (real platform 5)

Every stop is placed on the cell for (station column, track class of that leg), so
an up leg always runs east on an up row and a down leg always runs west on a down
row. Direction-consistency is therefore structural: the head-on class of deadlock
cannot be built. Services whose legs use the main tracks (removed from the map)
are dropped and reported.

Outputs
    data/levels/network_simple_rail.mpk
    data/levels/network_simple_mapping.json
    data/schedules/network_simple_timetable.json
    data/levels/network_simple.pkl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import msgpack
import numpy as np

REPO = Path(__file__).resolve().parents[1]

STATION_COL = {"Strathfield": 1, "Burwood": 14, "Redfern": 40, "Central": 84}
UP_ROW = {"up_suburban": 2, "up_local": 6}
DOWN_ROW = {"down_suburban": 4, "down_local": 8}

# compact layout: the four through tracks become adjacent (rows 2..5); the
# lateral rows that used to sit between them are dropped
COMPACT_ROWS = {2: 2, 4: 3, 6: 4, 8: 5}
UP_ROW_COMPACT = {"up_suburban": 2, "up_local": 4}
DOWN_ROW_COMPACT = {"down_suburban": 3, "down_local": 5}
REAL_PLATFORM = {
    "Strathfield": {"up_suburban": 4, "down_suburban": 6, "up_local": 7, "down_local": 8},
    "Burwood": {"up_suburban": 3, "down_suburban": None, "up_local": 5, "down_local": 6},
    "Redfern": {"up_suburban": 3, "down_suburban": 4, "up_local": 5, "down_local": 6},
    "Central": {"up_suburban": 16, "down_suburban": 18, "up_local": 17, "down_local": 19},
}
SECS_PER_STEP = 20
SPEED_EXPRESS = 1.0
SPEED_LOCAL = 0.5
PAD = 2
DEADLINE_SLACK = 4  # measured residual with force-stop dwell (was 1 without dwell)


def trim(src: Path, out: Path, pad: int = PAD) -> tuple[int, int, int, int]:
    # load_env_dict applies flatland's numpy codec; raw msgpack returns a dict
    from flatland.envs.persistence import RailEnvPersister

    data = dict(RailEnvPersister.load_env_dict(str(src)))
    grid = np.asarray(data["grid"], dtype=np.uint16)
    height, width = grid.shape
    used = np.argwhere(grid != 0)
    r0 = max(0, int(used[:, 0].min()) - pad)
    r1 = min(height - 1, int(used[:, 0].max()) + pad)
    c0 = max(0, int(used[:, 1].min()) - pad)
    c1 = min(width - 1, int(used[:, 1].max()) + pad)
    cropped = grid[r0 : r1 + 1, c0 : c1 + 1]

    data["grid"] = cropped.tolist()
    data["agents"] = []
    data["level_free_positions"] = [
        (int(r) - r0, int(c) - c0) for r, c in (data.get("level_free_positions") or [])
    ]
    editor = dict(data.get("railenv_editor") or {})
    if editor.get("stations"):
        editor["stations"] = [(int(x) - c0, int(y) - r0) for x, y in editor["stations"]]
    if isinstance(editor.get("origin"), (list, tuple)) and len(editor["origin"]) == 2:
        ox, oy = editor["origin"]
        editor["origin"] = (int(ox) - c0, int(oy) - r0)
    data["railenv_editor"] = editor
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(msgpack.packb(data, use_bin_type=True))
    print(f"trimmed {width}x{height} -> {cropped.shape[1]}x{cropped.shape[0]} "
          f"(row offset {r0}, col offset {c0})")
    return r0, c0, cropped.shape[1], cropped.shape[0]


def compact(src: Path, out: Path) -> tuple[int, int]:
    """Drop the lateral rows, moving the four through tracks adjacent (rows 2-5)."""
    from flatland.envs.persistence import RailEnvPersister
    import msgpack as _msgpack

    data = dict(RailEnvPersister.load_env_dict(str(src)))
    grid = np.asarray(data["grid"], dtype=np.uint16)
    height, width = grid.shape
    new = np.zeros((8, width), dtype=np.uint16)
    for old_row, new_row in COMPACT_ROWS.items():
        if old_row < height:
            new[new_row] = grid[old_row]
    data["grid"] = new.tolist()
    data["agents"] = []
    # station metadata: shift rows 28/30/32/34 -> 2/3/4/5, drop the lateral rows
    editor = dict(data.get("railenv_editor") or {})
    kept = []
    for x, y in (editor.get("stations") or []):
        row = int(y) + 20
        if row in COMPACT_ROWS:
            kept.append((int(x), COMPACT_ROWS[row] - 20))
    editor["stations"] = kept
    data["railenv_editor"] = editor
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_msgpack.packb(data, use_bin_type=True))
    print(f"compacted {width}x{height} -> {width}x8 (rows 2,4,6,8 -> 2,3,4,5)")
    return 8, width


def write_mapping(path: Path, height: int, width: int, compact: bool = False) -> dict:
    up = UP_ROW_COMPACT if compact else UP_ROW
    down = DOWN_ROW_COMPACT if compact else DOWN_ROW
    mapping = {
        "note": "row = track class, col = station, derived from level geometry + timetable track codes",
        "level_height": height,
        "level_width": width,
        "columns": STATION_COL,
        "rows": dict(up, **down),
        # in the compact map the ladder rows are gone, so the Strathfield bay has
        # no dedicated row of its own
        "bay": None if compact else {"station": "Strathfield", "row": 3, "real_platform": 5},
        "real_platform": REAL_PLATFORM,
        "dropped_tracks": ["up_main", "down_main"],
        "directions": {"up": "EAST", "down": "WEST"},
    }
    path.write_text(json.dumps(mapping, indent=2))
    print(f"mapping -> {path}")
    return mapping


def simple_schedule(src: Path, out: Path) -> list[dict]:
    services = json.loads(src.read_text())["services"]
    kept = [s for s in services if s.get("track") in UP_ROW and s.get("return_track") in DOWN_ROW]
    dropped = [s["id"] for s in services if s not in kept]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": "network-simple", "services": kept}, indent=1))
    print(f"timetable -> {out}: kept {len(kept)}/{len(services)} "
          f"(dropped {len(dropped)} main-track services)")
    return kept


def build(rail: Path, services: list[dict], out: Path,
          up_row=None, down_row=None) -> None:
    up_row = up_row or UP_ROW
    down_row = down_row or DOWN_ROW
    from flatland.core.grid.grid4 import Grid4TransitionsEnum
    from flatland.envs.agent_utils import EnvAgent
    from flatland.envs.persistence import RailEnvPersister
    from flatland.envs.rail_trainrun_data_structures import Waypoint
    from flatland.envs.timetable_utils import Line, Timetable

    env, _ = RailEnvPersister.load_new(str(rail))
    east, west = Grid4TransitionsEnum.EAST, Grid4TransitionsEnum.WEST
    gmin = min(min(s["outbound"][0]["dep"], s["return"][0]["dep"]) for s in services)

    waypoints, speeds, earliest, latest = {}, [], [], []
    for service_index, service in enumerate(services):
        speed = SPEED_EXPRESS if service.get("express") else SPEED_LOCAL
        for leg, row_map, heading in (
            ("outbound", UP_ROW, east),
            ("return", DOWN_ROW, west),
        ):
            stops = service[leg]
            track = service["track"] if leg == "outbound" else service["return_track"]
            row = (up_row["up_local"] if service_index % 2 else up_row["up_suburban"]) \
                if leg == "outbound" else \
                (down_row["down_local"] if service_index % 2 else down_row["down_suburban"])
            cells = [(row, STATION_COL[s["station"]]) for s in stops]
            # stops and the target accept EITHER same-direction track, so the
            # agent chooses its route (and whether to cross/overtake); the origin
            # stays a single cell because that is where the train is placed
            rows_ok = (up_row["up_suburban"], up_row["up_local"]) if leg == "outbound" \
                else (down_row["down_suburban"], down_row["down_local"])
            groups = [
                [Waypoint((row, STATION_COL[st["station"]]), heading)]
                if i == 0 else
                [Waypoint((r, STATION_COL[st["station"]]), heading) for r in rows_ok]
                for i, st in enumerate(stops)
            ]
            first_dep = stops[0]["dep"] if stops[0]["dep"] is not None else stops[0]["arr"]
            ed0 = 1 + int(round((first_dep - gmin) / SECS_PER_STEP))
            ed_list, la_list = [ed0], [None]
            travelled = dwell_acc = 0
            for j in range(1, len(stops)):
                travelled += abs(cells[j][1] - cells[j - 1][1])
                arrival = ed0 + int(math.ceil(travelled / speed)) + dwell_acc
                dwell = 0
                if stops[j].get("arr") is not None and stops[j].get("dep") is not None:
                    dwell = max(0, int(round((stops[j]["dep"] - stops[j]["arr"]) / SECS_PER_STEP)))
                ed_list.append(arrival + dwell if j < len(stops) - 1 else None)
                la_list.append(arrival + DEADLINE_SLACK)
                dwell_acc += dwell
            handle = len(waypoints)
            waypoints[handle] = groups
            speeds.append(speed)
            earliest.append(ed_list)
            latest.append(la_list)

    max_steps = max(v[-1] or 0 for v in latest) + 200
    timetable = Timetable(earliest_departures=earliest, latest_arrivals=latest,
                          max_episode_steps=max_steps)
    line = Line(agent_waypoints=waypoints, agent_speeds=speeds)
    agents = EnvAgent.apply_timetable(EnvAgent.from_line(line), timetable)
    for agent in agents:
        # from_line builds `targets` from the FIRST waypoint of the last group, so
        # a multi-track target group would pin every train to one row; rebuild the
        # target set from the whole group (any listed configuration ends the trip)
        group = waypoints[agent.handle][-1]
        agent.targets = {
            ((int(w.position[0]), int(w.position[1])), int(w.direction)) for w in group
        }
        env.add_agent(agent)
    env._max_episode_steps = max_steps
    env.reset(regenerate_rail=False, regenerate_schedule=False)
    RailEnvPersister.save(env, str(out))
    print(f"env -> {out}: {env.get_num_agents()} agents, max_episode_steps={max_steps}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/james/Downloads/network(1).mpk")
    ap.add_argument("--schedule", default=str(REPO / "data/schedules/central_strathfield_roundtrips.json"))
    ap.add_argument("--rail-out", default=str(REPO / "data/levels/network_simple_rail.mpk"))
    ap.add_argument("--map-out", default=str(REPO / "data/levels/network_simple_mapping.json"))
    ap.add_argument("--schedule-out", default=str(REPO / "data/schedules/network_simple_timetable.json"))
    ap.add_argument("--env-out", default=str(REPO / "data/levels/network_simple.pkl"))
    ap.add_argument("--compact", action="store_true", help="put the four tracks adjacent (8-row map)")
    args = ap.parse_args()

    _, _, width, height = trim(Path(args.src).expanduser(), Path(args.rail_out))
    if args.compact:
        height, width = compact(Path(args.rail_out), Path(args.rail_out))
    write_mapping(Path(args.map_out), height, width, compact=args.compact)
    services = simple_schedule(Path(args.schedule), Path(args.schedule_out))
    if args.compact:
        build(Path(args.rail_out), services, Path(args.env_out),
              up_row=UP_ROW_COMPACT, down_row=DOWN_ROW_COMPACT)
    else:
        build(Path(args.rail_out), services, Path(args.env_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
