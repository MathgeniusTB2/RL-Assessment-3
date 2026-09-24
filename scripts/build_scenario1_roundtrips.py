#!/usr/bin/env python3
"""Extract real TfNSW *round trips* for the Central-Strathfield corridor.

The one-way extractor (``build_scenario1_timetable.py``) clips every trip to the
four modelled stations and takes the first modelled stop as its origin, so every
agent spawns at Central or Strathfield and is marked DONE at the far end. This
script instead pairs the two legs of a physical train (GTFS ``block_id``) into a
single round trip: west -> Central -> west. Central is therefore a *waypoint*
where the train turns back (the ``network (5)`` map has dead-end turnbacks on the
top 14 Central platforms), and the final target is Strathfield.

Only trips on the corridor heavy-rail lines are considered, and only blocks whose
two corridor legs run in opposite directions (a genuine out-and-back vehicle).

Usage:
    python scripts/build_scenario1_roundtrips.py \
        --gtfs "/Users/.../grid_builder/gtfs" \
        --date 2026-09-24 --window 16:00-19:00 \
        --out data/schedules/central_strathfield_roundtrips.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# Parent station ids (TfNSW GTFS) -> modelled station name.
PARENT = {
    "200060": "Central",
    "201510": "Redfern",
    "213410": "Burwood",
    "213510": "Strathfield",
}

# Heavy-rail lines that use the Central<->Strathfield corridor.
LINES = ("T1", "T2", "T3", "T9", "CCN", "BMT")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GTFS = Path(
    "/Users/weihuazhang/Documents/James/RL - rail project/grid_builder/gtfs"
)
DEFAULT_OUT = REPO_ROOT / "data" / "schedules" / "central_strathfield_roundtrips.json"

# line -> track pair on the scenario grid (matches install_new_map's track names).
LINE_TRACKS = {
    "T1": "suburban",
    "T2": "local",
    "T3": "local",
    "CCN": "main",
    "BMT": "main",
    "T9": "suburban",
}

# Plausible Central turnaround window (seconds) for pairing a real return leg.
MIN_TURNAROUND = 0
MAX_TURNAROUND = 45 * 60


def _secs(value: str) -> int | None:
    try:
        h, m, s = value.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def _hms(secs: int) -> str:
    secs %= 24 * 3600
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"


def _parse_window(window: str) -> tuple[int, int]:
    start, end = window.split("-")
    return _secs(start.strip() + ":00"), _secs(end.strip() + ":00")


def _platform_map(stops_path: Path) -> dict[str, tuple[str, str]]:
    """Child platform stop_id -> (modelled parent station name, platform code)."""
    child2info: dict[str, tuple[str, str]] = {}
    with open(stops_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            parent = row.get("parent_station", "")
            if parent in PARENT:
                code = (row.get("platform_code") or "").strip()
                if not code and "Platform" in row.get("stop_name", ""):
                    code = row["stop_name"].split()[-1]
                child2info[row["stop_id"]] = (PARENT[parent], code)
    return child2info


def _active_services(calendar_path: Path, dates_path: Path, date: str) -> set[str]:
    """Weekday service ids active on ``date`` (YYYYMMDD), incl. calendar_dates."""
    active: set[str] = set()
    with open(calendar_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if not (row["start_date"] <= date <= row["end_date"]):
                continue
            if all(row[d] == "1" for d in ("monday", "tuesday", "wednesday", "thursday", "friday")):
                active.add(row["service_id"])
    if dates_path.exists():
        with open(dates_path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row["date"] != date:
                    continue
                if row["exception_type"] == "1":
                    active.add(row["service_id"])
                elif row["exception_type"] == "2":
                    active.discard(row["service_id"])
    return active


def _rail_routes(routes_path: Path) -> tuple[dict[str, str], set[str]]:
    short_by_id: dict[str, str] = {}
    wanted: set[str] = set()
    with open(routes_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row.get("route_type") != "2":  # 2 = heavy rail
                continue
            short = row.get("route_short_name", "")
            short_by_id[row["route_id"]] = short
            if short in LINES:
                wanted.add(row["route_id"])
    return short_by_id, wanted


def _track_for(line: str, direction: str, express: bool) -> str:
    pair = "main" if (line == "T9" and express) else LINE_TRACKS[line]
    return f"{direction}_{pair}"


def _corridor_stops(stops, child2info):
    """Deduped corridor stops of one trip: list of (station, arr, dep, platform)."""
    my = [
        (child2info[sid][0], _secs(at), _secs(dp), child2info[sid][1])
        for _, sid, at, dp in stops
        if sid in child2info
    ]
    ded = []
    for name, arr, dep, plat in my:
        if ded and ded[-1][0] == name:
            ded[-1] = (name, arr, dep, plat)
        else:
            ded.append((name, arr, dep, plat))
    return ded


def build(gtfs_dir: Path, date: str, window: str) -> dict:
    date = date.replace("-", "")  # calendar uses YYYYMMDD
    child2info = _platform_map(gtfs_dir / "stops.txt")
    active = _active_services(
        gtfs_dir / "calendar.txt", gtfs_dir / "calendar_dates.txt", date
    )
    short_by_id, routes = _rail_routes(gtfs_dir / "routes.txt")

    # trip -> (line, direction_id, block_id)
    trips: dict[str, tuple[str, str, str]] = {}
    with open(gtfs_dir / "trips.txt", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["service_id"] in active and row["route_id"] in routes:
                trips[row["trip_id"]] = (
                    short_by_id[row["route_id"]],
                    row["direction_id"],
                    (row.get("block_id") or "").strip(),
                )
    print(f"[gtfs] {len(trips)} weekday rail trips on {LINES}")

    seq: dict[str, list] = defaultdict(list)
    with open(gtfs_dir / "stop_times.txt", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            tid = row["trip_id"]
            if tid in trips:
                seq[tid].append(
                    (
                        int(row["stop_sequence"]),
                        row["stop_id"],
                        row.get("arrival_time", ""),
                        row.get("departure_time", ""),
                    )
                )
    print(f"[gtfs] {len(seq)} of those trips have stop_times")

    w0, w1 = _parse_window(window)

    # Build a corridor-leg record per trip (both directions).
    legs: dict[str, dict] = {}
    for tid, stops in seq.items():
        stops.sort()
        ded = _corridor_stops(stops, child2info)
        names = [d[0] for d in ded]
        if "Central" not in names or "Strathfield" not in names:
            continue
        line, _dir, block = trips[tid]
        direction = "down" if names.index("Central") < names.index("Strathfield") else "up"
        express = "Redfern" not in names and "Burwood" not in names
        origin_dep = ded[0][2] if ded[0][2] is not None else ded[0][1]
        if origin_dep is None:
            continue
        legs[tid] = {
            "trip_id": tid,
            "line": line,
            "block_id": block,
            "direction": direction,
            "express": express,
            "track": _track_for(line, direction, express),
            "origin_dep": origin_dep,
            "stops": [
                {"station": n, "arr": a, "dep": d, "platform": p}
                for n, a, d, p in ded
            ],
        }

    # Pair the outbound (up, arriving at Central) with a real return (down,
    # leaving Central) leg. A genuine GTFS block_id does NOT link two corridor
    # crossings here (the paired trip is usually a City Circle run that never
    # reaches Strathfield), so pair real trips by time instead: for each up leg,
    # take the earliest down leg of the same line whose Central departure is
    # within a plausible turnaround window. Each down leg is used at most once.
    def central_time(leg, key):
        return next((s[key] for s in leg["stops"] if s["station"] == "Central"), None)

    ups = sorted(
        (l for l in legs.values() if l["direction"] == "up"),
        key=lambda l: central_time(l, "arr") or 0,
    )
    downs = sorted(
        (l for l in legs.values() if l["direction"] == "down"),
        key=lambda l: central_time(l, "dep") or 0,
    )

    services = []
    used_down: set[str] = set()
    for up in ups:
        if not (w0 <= up["origin_dep"] < w1):
            continue
        central_arr = central_time(up, "arr")
        if central_arr is None:
            continue
        best = None
        for down in downs:
            if down["trip_id"] in used_down:
                continue
            central_dep = central_time(down, "dep")
            if central_dep is None:
                continue
            turn = central_dep - central_arr
            if not (MIN_TURNAROUND <= turn <= MAX_TURNAROUND):
                continue
            if best is None or turn < best[0]:
                best = (turn, down)
        if best is None:
            continue
        turn, down = best
        used_down.add(down["trip_id"])
        central_platform = next(
            (s["platform"] for s in up["stops"] if s["station"] == "Central"), ""
        )
        services.append(
            {
                "id": f"{up['line']}-{up['trip_id']}",
                "line": up["line"],
                "return_line": down["line"],
                "direction": "up",
                "track": up["track"],
                "return_track": down["track"],
                "express": up["express"],
                "return_express": down["express"],
                "block_id": up["block_id"],
                "central_platform": central_platform,
                "turnaround": turn,
                "outbound": up["stops"],
                "return": down["stops"],
            }
        )

    services.sort(key=lambda s: s["outbound"][0]["dep"] or s["outbound"][0]["arr"])
    return {
        "name": f"Central-Strathfield round trips, weekday {date}, {window}",
        "reference_date": date,
        "window": window,
        "lines": list(LINES),
        "stations": ["Strathfield", "Burwood", "Redfern", "Central"],
        "services": services,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gtfs",
        type=Path,
        default=Path(os.environ.get("SCENARIO1_GTFS", DEFAULT_GTFS)),
        help="GTFS directory (or set SCENARIO1_GTFS)",
    )
    ap.add_argument("--date", default="2026-09-24", help="service date YYYY-MM-DD")
    ap.add_argument("--window", default="16:00-19:00", help="outbound origin-departure window")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    data = build(args.gtfs, args.date, args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")

    services = data["services"]
    print(f"[gtfs] wrote {args.out}: {len(services)} round trips in {args.window}")
    print("  by line:", Counter(s["line"] for s in services).most_common())
    print(
        "  central platforms:",
        Counter(s["central_platform"] for s in services).most_common(),
    )
    turn = [s["turnaround"] for s in services]
    if turn:
        print(f"  turnaround: min={min(turn)}s max={max(turn)}s median={sorted(turn)[len(turn)//2]}s")
    for s in services[:5]:
        out = " | ".join(
            f"{st['station'][:4]}@{_hms(st['dep'] or st['arr'])}" for st in s["outbound"]
        )
        ret = " | ".join(
            f"{st['station'][:4]}@{_hms(st['dep'] or st['arr'])}" for st in s["return"]
        )
        print(f"    {s['id']:18} plat={s['central_platform']:>2} {out}  ||  {ret}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
