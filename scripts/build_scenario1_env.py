#!/usr/bin/env python3
"""Build the scenario-1 Flatland env and save it as a full .mpk.

Scenario 1 is the Central-Strathfield (Main Suburban) corridor on the 92x37
turnback level, run on the real round-trip schedule: each physical round trip
goes out to Central, reverses at a dead-end turnback platform, then returns to
Strathfield. This script turns the rail grid plus the schedule into agents and
persists the whole environment (rail + agents + line + timetable), so the
notebook can simply load it:

    env, _ = RailEnvPersister.load_new("data/levels/central_strathfield.pkl", ...)

Inputs:
    data/levels/central_strathfield_rail.mpk       rail grid + editor metadata
    data/schedules/central_strathfield_roundtrips.json
Output:
    data/levels/central_strathfield.pkl           full env (loaded by the notebook)

The full env is saved as pickle (`.pkl`), not msgpack (`.mpk`), because live
agents carry a `SpeedCounter` that msgpack cannot serialise.

Usage:
    python scripts/build_scenario1_env.py
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAIL = REPO_ROOT / "data" / "levels" / "central_strathfield_rail.mpk"
DEFAULT_SCHEDULE = REPO_ROOT / "data" / "schedules" / "central_strathfield_roundtrips.json"
DEFAULT_OUT = REPO_ROOT / "data" / "levels" / "central_strathfield.pkl"

EAST = int(Grid4TransitionsEnum.EAST)
WEST = int(Grid4TransitionsEnum.WEST)
SECS_PER_STEP = 20
SPEED_EXPRESS = 1.0
SPEED_LOCAL = 0.5


@dataclass(frozen=True)
class Spec:
    """Minimal scenario spec passed to the schedule builder."""

    level_path: Path
    stations: dict
    platforms: dict
    secs_per_step: int
    speed_express: float
    speed_local: float
    warm_start_t0: int | None = None


def load_level(spec: Spec):
    """Load a scenario's rail level (no fleet yet)."""
    return RailEnvPersister.load_new(str(spec.level_path))


def stations_from_editor(env_dict: dict) -> tuple[dict, dict]:
    """Recover station and platform cells from a level's editor metadata.

    Editor coords map to env cells as ``col = x - 2``, ``row = y + 20``. Columns
    run west to east and name Strathfield, Burwood, Redfern, Central; rows 24-34
    are the six main tracks (up/down x main/suburban/local).
    """
    from collections import defaultdict

    editor = env_dict.get("railenv_editor", {}) or {}
    by_col: dict[int, list[int]] = defaultdict(list)
    for x, y in editor.get("stations", []):
        by_col[x - 2].append(y + 20)
    cols = sorted(by_col)
    names = ["Strathfield", "Burwood", "Redfern", "Central"]
    main_rows = [24, 26, 28, 30, 32, 34]
    track_names = ["up_main", "down_main", "up_suburban", "down_suburban",
                   "up_local", "down_local"]
    stations: dict = {}
    platforms: dict = {}
    for name, col in zip(names, cols):
        rows = sorted(by_col[col])
        platforms[name] = [(r, col) for r in rows]
        stations[name] = {tr: (r, col) for tr, r in zip(track_names, main_rows) if r in rows}
    return stations, platforms


def _patch_env_agent_targets() -> None:
    """Make ``EnvAgent.from_line`` direction-aware but ``None``-safe (idempotent).

    Flatland's default target is every heading at the final *position*. On a round
    trip the origin and terminus share a position, so the target must be the
    ``(position, direction)`` configuration of the final waypoint. A waypoint with
    ``direction=None`` (the one-way final target) keeps the default all-headings
    behaviour, so this patch is harmless outside round trips.
    """
    from flatland.envs.agent_utils import EnvAgent

    if getattr(EnvAgent, "_direction_aware_targets", False):
        return
    original = EnvAgent.__dict__["from_line"].__func__

    def _from_line(cls, line):
        agents = original(cls, line)
        for i, agent in enumerate(agents):
            targets = set()
            for wp in line.agent_waypoints[i][-1]:
                if wp.direction is None:
                    targets |= {(wp.position, d) for d in Grid4TransitionsEnum}
                else:
                    targets.add((wp.position, wp.direction))
            agent.targets = targets
        return agents

    EnvAgent.from_line = classmethod(_from_line)
    EnvAgent._direction_aware_targets = True


def _services_to_schedule_roundtrip(
    services: Sequence[dict], spec: ScenarioSpec
) -> tuple[Line, Timetable, list[dict]]:
    """Convert real round-trip services into a Flatland ``Line`` + ``Timetable``.

    Each service is a physical round trip: outbound (west -> Central), a reversal
    at one of Central's dead-end turnback platforms, then the return leg back to
    Strathfield (the final target). Every stop is a **preferred-first** waypoint
    group (preferred platform first, the other platforms of that station as
    fallbacks); Central's group is its turnback platform. ``spec.warm_start_t0``
    optionally drops stops already past at that time (spawn mid-corridor).
    """
    _patch_env_agent_targets()
    env, _ = load_level(spec)
    platforms = spec.platforms
    central_cells = platforms.get("Central")
    if not central_cells:
        col = spec.stations["Central"]["up_main"][1]
        central_cells = [(r, col) for r in range(0, 35, 2)]
    central_col = central_cells[0][1]
    central_rows = sorted({r for r, _ in central_cells})
    turnback_rows = sorted(r for r, c in central_cells if env.rail.is_dead_end((r, c)))
    turnback_rows = turnback_rows or [central_cells[0][0]]
    last_row = max(r for r, c in central_cells)
    turnback_set = set(turnback_rows)
    terminal_rows = sorted(r for r, c in central_cells if r not in turnback_set)

    # Map a service's corridor track to the Central row it should use. The
    # editor's platform codes do not line up with the synthetic terminus, so we
    # derive the rows from the actual rail: an eastbound service must end at a
    # cell that can be reached heading east, a westbound service must depart from
    # a cell whose west exit exists. Westbound departures are only drawn on the
    # down-side stubs (rows 30/34); using an up row causes a head-on deadlock on
    # the corridor.
    UP_ROW = {"up_main": 24, "up_suburban": 28, "up_local": 32}
    DOWN_ROW = {"down_main": 26, "down_suburban": 30, "down_local": 34}

    def _reachable(starts):
        seen = set(starts)
        stack = list(starts)
        while stack:
            for pos, direction in env.rail.get_successor_configurations(stack.pop()):
                conf = ((int(pos[0]), int(pos[1])), int(direction))
                if conf not in seen:
                    seen.add(conf)
                    stack.append(conf)
        return seen

    _starts = [
        ((row, 1), direction)
        for row in sorted(set(UP_ROW.values()) | set(DOWN_ROW.values()))
        for direction in (EAST, WEST)
        if env.rail.get_successor_configurations(((row, 1), direction))
    ]
    _reached = _reachable(_starts)
    east_target_rows = [r for r in central_rows if ((r, central_col), EAST) in _reached]
    west_depart_rows = [
        r for r in central_rows
        if env.rail.get_successor_configurations(((r, central_col), WEST))
    ]
    down_depart_rows = [r for r in west_depart_rows if r in set(DOWN_ROW.values())]

    def _up_arrive_row(track):
        pool = east_target_rows or terminal_rows
        return min(pool, key=lambda r: abs(r - UP_ROW.get(track, 28)))

    def _down_depart_row(return_track):
        pool = down_depart_rows or west_depart_rows or terminal_rows
        return min(pool, key=lambda r: abs(r - DOWN_ROW.get(return_track, 30)))

    def _row_for_code(code):
        try:
            return min(2 * (int(code) - 1), last_row)
        except (TypeError, ValueError):
            return None

    def _origin_dep(service):
        first = service["outbound"][0]
        return first["dep"] if first["dep"] is not None else first["arr"]

    def _station_group(station, direction, preferred_track):
        pref = tuple(spec.stations[station][preferred_track])
        cells = list(platforms.get(station, [pref]))
        cells.sort(key=lambda cell: (tuple(cell) != pref, abs(cell[0] - pref[0])))
        return [Waypoint(tuple(cell), Grid4TransitionsEnum(direction)) for cell in cells]

    def _central_turnback_group(row):
        pref = (row, central_col)
        alts = [(r, central_col) for r in sorted(turnback_set) if r != row]
        return [Waypoint(pref, Grid4TransitionsEnum(EAST))] + [
            Waypoint(c, Grid4TransitionsEnum(EAST)) for c in alts
        ]

    def _central_terminal_group(row, direction):
        pref = (row, central_col)
        alts = [(r, central_col) for r in terminal_rows if r != row]
        return [Waypoint(pref, Grid4TransitionsEnum(direction))] + [
            Waypoint(c, Grid4TransitionsEnum(direction)) for c in alts
        ]

    configs = list(services)
    if not configs:
        raise ValueError("No services in round-trip timetable")
    gmin = min(_origin_dep(s) for s in configs)
    t0 = getattr(spec, "warm_start_t0", None)

    waypoints: dict[int, list[list[Waypoint]]] = {}
    speeds: list[float] = []
    earliest: list[list[int | None]] = []
    latest: list[list[int | None]] = []
    meta: list[dict] = []
    ends = [0]

    def _ed0_for(stop):
        if t0 is not None:
            return 1
        when = stop["dep"] if stop["dep"] is not None else stop["arr"]
        return 1 + int(round((when - gmin) / spec.secs_per_step))

    def _trim(stops):
        if t0 is None:
            return 0, stops
        start = len(stops) - 1
        for i, st in enumerate(stops):
            when = st["dep"] if st["dep"] is not None else st["arr"]
            if when is not None and when >= t0:
                start = i
                break
        seq = stops[start:]
        return (start, seq) if len(seq) >= 2 else (0, stops)

    def _timing(seq, cells, ed0, speed):
        seg = [abs(cells[i + 1][1] - cells[i][1]) for i in range(len(cells) - 1)]
        ed_list: list = [ed0]
        la_list: list = [None]
        travel = 0
        dwell_acc = 0
        for j, stop in enumerate(seq):
            if j == 0:
                continue
            travel += seg[j - 1]
            arr = ed0 + int(math.ceil(travel / max(speed, 1e-6))) + dwell_acc
            dwell = 0
            if stop.get("arr") is not None and stop.get("dep") is not None:
                dwell = max(0, int(round((stop["dep"] - stop["arr"]) / spec.secs_per_step)))
            ed_list.append(arr + dwell if j < len(seq) - 1 else None)
            la_list.append(arr)
            dwell_acc += dwell
        return ed_list, la_list

    def _meta(service, kind, seq, cells, ed_list, la_list, direction):
        return {
            "id": f"{service['id']}:{kind}",
            "line": service["line"],
            "direction": direction,
            "track": service["track"] if direction == "up" else service["return_track"],
            "express": bool(service.get("express")),
            "central_platform": service.get("central_platform", ""),
            "origin_departure": ed_list[0],
            "scheduled_arrival": la_list[-1],
            "stops": [
                {
                    "station": st["station"],
                    "cell": cells[i],
                    "scheduled_departure": ed_list[i],
                    "scheduled_arrival": la_list[i],
                }
                for i, st in enumerate(seq)
            ],
        }

    def _add(groups, speed, ed_list, la_list, m):
        h = len(waypoints)
        waypoints[h] = groups
        speeds.append(speed)
        earliest.append(ed_list)
        latest.append(la_list)
        meta.append(m)
        ends[0] = max(ends[0], la_list[-1] or 0)

    for service in configs:
        track = service["track"]
        return_track = service["return_track"]
        speed = spec.speed_express if service.get("express") else spec.speed_local
        out = list(service["outbound"])
        ret = list(service["return"])

        out_row = _row_for_code(service.get("central_platform", ""))
        if out_row is None:
            out_row = min(turnback_set)

        if out_row in turnback_set:
            # Turnback: one agent does the full round trip (Central bounce).
            stops = out + [s for s in ret if s["station"] != "Central"]
            if out and out[-1]["station"] == "Central":
                stops[len(out) - 1]["dep"] = ret[0]["dep"]   # real turnaround
            start, seq = _trim(stops)
            n_out = max(1, len(out) - start)
            cells: list = []
            groups: list = []
            for j, st in enumerate(seq):
                direction = EAST if j < n_out else WEST
                if st["station"] == "Central":
                    group = _central_turnback_group(out_row)
                else:
                    group = _station_group(
                        st["station"], direction, track if j < n_out else return_track
                    )
                cells.append(tuple(group[0].position))
                groups.append(group)
            ed_list, la_list = _timing(seq, cells, _ed0_for(seq[0]), speed)
            _add(groups, speed, ed_list, la_list,
                 _meta(service, "turnback", seq, cells, ed_list, la_list, "up"))
        else:
            # Terminal at Central (spawn/delete like Strathfield): the outbound
            # agent ends at Central, the return trip is a separate agent spawning
            # at Central. Each follows its own trip.
            _, out_seq = _trim(out)
            out_target_row = _up_arrive_row(track)
            out_cells: list = []
            out_groups: list = []
            for st in out_seq:
                group = (
                    _central_terminal_group(out_target_row, EAST)
                    if st["station"] == "Central"
                    else _station_group(st["station"], EAST, track)
                )
                out_cells.append(tuple(group[0].position))
                out_groups.append(group)
            out_ed, out_la = _timing(out_seq, out_cells, _ed0_for(out_seq[0]), speed)
            _add(out_groups, speed, out_ed, out_la,
                 _meta(service, "up", out_seq, out_cells, out_ed, out_la, "up"))

            ret_row = _down_depart_row(return_track)
            _, ret_seq = _trim(ret)
            ret_cells: list = []
            ret_groups: list = []
            for st in ret_seq:
                group = (
                    _central_terminal_group(ret_row, WEST)
                    if st["station"] == "Central"
                    else _station_group(st["station"], WEST, return_track)
                )
                ret_cells.append(tuple(group[0].position))
                ret_groups.append(group)
            ret_ed, ret_la = _timing(ret_seq, ret_cells, _ed0_for(ret_seq[0]), speed)
            _add(ret_groups, speed, ret_ed, ret_la,
                 _meta(service, "down", ret_seq, ret_cells, ret_ed, ret_la, "down"))

    max_episode_steps = max(400, ends[0] + 200)
    line = Line(agent_waypoints=waypoints, agent_speeds=speeds)
    timetable = Timetable(
        earliest_departures=earliest,
        latest_arrivals=latest,
        max_episode_steps=max_episode_steps,
    )
    return line, timetable, meta


def build(rail_path: Path, schedule_path: Path, out_path: Path,
          warm_start_t0: int | None = None) -> Path:
    """Build the full scenario env from the rail grid and round-trip schedule."""
    rail_dict = RailEnvPersister.load_env_dict(str(rail_path))
    stations, platforms = stations_from_editor(rail_dict)
    spec = Spec(rail_path, stations, platforms, SECS_PER_STEP, SPEED_EXPRESS,
                SPEED_LOCAL, warm_start_t0)

    services = json.loads(Path(schedule_path).read_text())["services"]
    line, timetable, meta = _services_to_schedule_roundtrip(services, spec)

    env = RailEnv(
        width=len(rail_dict["grid"][0]),
        height=len(rail_dict["grid"]),
        rail_generator=rail_from_file(env_dict=rail_dict),
        line_generator=lambda *a, **k: line,
        timetable_generator=lambda *a, **k: timetable,
        number_of_agents=len(line.agent_waypoints),
        obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=0)
    env.number_of_agents = env.get_num_agents()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    RailEnvPersister.save(env, str(out_path))
    n_turnback = sum(1 for m in meta if m["id"].endswith(":turnback"))
    print(
        f"built {out_path}: {env.width}x{env.height}, {env.get_num_agents()} agents "
        f"from {len(services)} round trips ({n_turnback} turnbacks), "
        f"max_episode_steps={env._max_episode_steps}"
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rail", type=Path, default=DEFAULT_RAIL)
    ap.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--warm-start", default=None,
                    help="optional HH:MM to drop stops already past (spawn mid-corridor)")
    args = ap.parse_args()
    warm = None
    if args.warm_start:
        hh, mm = args.warm_start.split(":")
        warm = int(hh) * 3600 + int(mm) * 60
    build(args.rail, args.schedule, args.out, warm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
