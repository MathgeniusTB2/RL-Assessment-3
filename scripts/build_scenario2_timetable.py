#!/usr/bin/env python3
"""Extract a real TfNSW timetable for scenario 2 (Granville-Harris Park-Merrylands).

Scenario 2 models a three-station junction on the Main Western / Old Main South:

    Granville <-> Harris Park      (Main Western; T1 on the Main tracks, T2 on Suburban)
    Granville <-> Merrylands       (Old Main South; T2)
    Harris Park <-> Merrylands     (Cumberland "Y-link"; T5)

Unlike the scenario-1 builder, this keeps **every** heavy-rail service that
*traverses* one of those links, whether or not it stops at the modelled stations:
a T1 express running Parramatta -> Clyde passes Granville and Harris Park but
stops at neither, and it still occupies the corridor.

Traversal is inferred from the GTFS **shape** geometry (shapes.txt): a modelled
station counts as passed when the trip's polyline comes within ``--tolerance``
kilometres of the station.  Pass times for stations that are not stops are
linearly interpolated between the surrounding stops using position along the
shape.  This is robust where a stop-sequence/graph method is ambiguous, because
Granville-HarrisPark-Merrylands form a topological triangle (the Old Main South
and the Main Western + Y-link both connect the same three stations).

Usage:
    python scripts/build_scenario2_timetable.py \
        --gtfs data/gtfs --date 2026-10-15 --window 16:00-19:00 \
        --out data/schedules/scenario2_timetable_20261015.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GTFS = REPO_ROOT / "data" / "gtfs"
DEFAULT_OUT = REPO_ROOT / "data" / "schedules" / "scenario2_timetable_20261015.json"

# TfNSW parent-station ids for the three modelled stations.
MODELLED = {
    "214240": "Granville",
    "215010": "Harris Park",
    "216010": "Merrylands",
}
# Heavy-rail lines that can use the modelled links.
LINES = ("T1", "T2", "T5")

# Movements that run towards the City (east) are "up"; the rest are "down".
UP_MOVEMENTS = {
    ("Harris Park", "Granville"),
    ("Merrylands", "Granville"),
    ("Merrylands", "Harris Park"),
}


def _rows(path: Path):
    with open(path, encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


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


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Equirectangular approximation, good to <1% over a few km."""
    mid = math.radians((lat1 + lat2) / 2)
    return math.hypot((lon1 - lon2) * 111.32 * math.cos(mid), (lat1 - lat2) * 110.57)


def _active_services(cal_path: Path, dates_path: Path, date: str, dow: str) -> set[str]:
    active: set[str] = set()
    for r in _rows(cal_path):
        if r["start_date"] <= date <= r["end_date"] and r[dow] == "1":
            active.add(r["service_id"])
    if dates_path.exists():
        for r in _rows(dates_path):
            if r["date"] != date:
                continue
            if r["exception_type"] == "1":
                active.add(r["service_id"])
            elif r["exception_type"] == "2":
                active.discard(r["service_id"])
    return active


def _platform(stop_name: str) -> str | None:
    """'Granville Station, Platform 3' -> '3'."""
    marker = "Platform "
    if marker in stop_name:
        return stop_name.split(marker, 1)[1].strip()
    return None


def build(gtfs: Path, date: str, window: str, tolerance: float) -> dict:
    day = date.replace("-", "")
    dow = ("monday", "tuesday", "wednesday", "thursday", "friday",
           "saturday", "sunday")[
        __import__("datetime").date(int(day[:4]), int(day[4:6]), int(day[6:])).weekday()
    ]
    active = _active_services(gtfs / "calendar.txt", gtfs / "calendar_dates.txt", day, dow)

    # station coordinates (location_type == 1)
    coord: dict[str, tuple[float, float]] = {}
    names: dict[str, str] = {}
    for r in _rows(gtfs / "stops.txt"):
        if r.get("location_type") == "1":
            coord[r["stop_id"]] = (float(r["stop_lat"]), float(r["stop_lon"]))
            names[r["stop_id"]] = r["stop_name"].replace(" Station", "")
    for sid in MODELLED:
        if sid not in coord:
            raise SystemExit(f"station id {sid} not found in stops.txt")

    rail_routes = {
        r["route_id"]: r["route_short_name"]
        for r in _rows(gtfs / "routes.txt")
        if r.get("route_type") == "2" and r.get("route_short_name") in LINES
    }

    trips: dict[str, tuple[str, str, str]] = {}  # trip_id -> (line, shape_id, headsign)
    for r in _rows(gtfs / "trips.txt"):
        if r["service_id"] in active and r["route_id"] in rail_routes:
            trips[r["trip_id"]] = (
                rail_routes[r["route_id"]],
                r["shape_id"],
                r.get("trip_headsign", ""),
            )
    print(f"[gtfs] {len(trips)} active {LINES} trips on {date} ({dow})")

    # shape polylines for just those trips
    wanted_shapes = {s for _, s, _ in trips.values()}
    shapes: dict[str, list[tuple[int, float, float]]] = collections.defaultdict(list)
    for r in _rows(gtfs / "shapes.txt"):
        sid = r["shape_id"]
        if sid in wanted_shapes:
            shapes[sid].append(
                (int(r["shape_pt_sequence"]), float(r["shape_pt_lat"]), float(r["shape_pt_lon"]))
            )
    for s in shapes.values():
        s.sort()
    print(f"[gtfs] {len(shapes)} shapes ({sum(len(v) for v in shapes.values())} points)")

    # every stop id -> its parent station id, plus platform labels
    stop_to_station: dict[str, str] = {}
    platform_of: dict[str, str | None] = {}
    for r in _rows(gtfs / "stops.txt"):
        sid = r["stop_id"]
        parent = r.get("parent_station", "")
        if parent:
            stop_to_station[sid] = parent
        elif r.get("location_type") == "1":
            stop_to_station[sid] = sid
        platform_of[sid] = _platform(r["stop_name"])

    # stop sequences for the trips of interest
    seq: dict[str, list[tuple[int, str, str, str]]] = collections.defaultdict(list)
    for r in _rows(gtfs / "stop_times.txt"):
        t = r["trip_id"]
        if t in trips:
            seq[t].append(
                (int(r["stop_sequence"]), r["stop_id"], r.get("arrival_time", ""), r.get("departure_time", ""))
            )
    print(f"[gtfs] {len(seq)} trips have stop_times")
    return _assemble(seq, trips, shapes, coord, names, stop_to_station, platform_of,
                     window, tolerance, date)


def _assemble(seq, trips, shapes, coord, names, stop_to_station, platform_of,
              window, tolerance, date) -> dict:
    w0, w1 = _parse_window(window)
    services: list[dict] = []
    for trip, stops in seq.items():
        line, shape_id, gtfs_headsign = trips[trip]
        poly = shapes.get(shape_id)
        if not poly:
            continue
        pts = [(lat, lon) for _, lat, lon in poly]

        # modelled stations the trip shape passes (nearest approach <= tolerance)
        passed: list[tuple[int, str]] = []
        for sid, station in MODELLED.items():
            la, lo = coord[sid]
            best_d, best_i = min(
                ((_km(pla, plo, la, lo), i) for i, (pla, plo) in enumerate(pts)),
                key=lambda t: t[0],
            )
            if best_d <= tolerance:
                passed.append((best_i, station))
        if len(passed) < 2:
            continue
        passed.sort()

        stops_sorted = sorted(stops)
        # map every stop to its nearest shape index, with arrival/departure
        stop_marks: list[tuple[int, int | None, int | None, str, str | None]] = []
        terminus = "?"
        for _, stop_id, arr, dep in stops_sorted:
            parent = stop_to_station.get(stop_id)
            if not parent or parent not in coord:
                continue
            plat = platform_of.get(stop_id) or platform_of.get(parent)
            la, lo = coord[parent]
            idx = min(range(len(pts)), key=lambda i: _km(pts[i][0], pts[i][1], la, lo))
            stop_marks.append((idx, _secs(arr), _secs(dep), names.get(parent, "?"), plat))
            terminus = names.get(parent, terminus)
        if not stop_marks:
            continue
        stop_marks.sort()
        idxs = [m[0] for m in stop_marks]

        def time_at(k: int, stop_marks=stop_marks) -> int | None:
            prev = next((m for m in reversed(stop_marks) if m[0] <= k), stop_marks[0])
            nxt = next((m for m in stop_marks if m[0] >= k), stop_marks[-1])
            if prev[0] == nxt[0]:
                return prev[2] if prev[2] is not None else prev[1]
            t0 = prev[2] if prev[2] is not None else prev[1]
            t1 = nxt[1] if nxt[1] is not None else nxt[2]
            if t0 is None or t1 is None:
                return t0 if t0 is not None else t1
            f = (k - prev[0]) / (nxt[0] - prev[0])
            return int(round(t0 + f * (t1 - t0)))

        # station info keyed by where it sits along the shape
        by_station: dict[str, tuple[int, int | None, str | None, bool]] = {}
        for idx, station in passed:
            stop_here = idx in idxs
            plat = next((m[4] for m in stop_marks if m[0] == idx), None) if stop_here else None
            by_station[station] = (idx, time_at(idx), plat, stop_here)

        order = [s for _, s in passed]
        for a, b in zip(order, order[1:], strict=False):
            _, dep, plat_o, stopped_o = by_station[a]
            _, arr, plat_d, _ = by_station[b]
            if dep is None or not (w0 <= dep < w1):
                continue
            direction = "up" if (a, b) in UP_MOVEMENTS else "down"
            services.append(
                {
                    "id": trip,
                    "line": line,
                    "headsign": terminus,
                    "direction": direction,
                    "origin": a,
                    "destination": b,
                    "departure": _hms(dep),
                    "arrival": _hms(arr) if arr is not None else None,
                    "origin_platform": plat_o,
                    "destination_platform": plat_d,
                    "stops_at_origin": stopped_o,
                    "stops_at_destination": by_station[b][3],
                    "gtfs_headsign": gtfs_headsign,
                }
            )

    services.sort(key=lambda s: s["departure"] or "")
    return {
        "name": f"Granville - Harris Park - Merrylands, weekday {date}, {window}",
        "reference_date": date.replace("-", ""),
        "window": window,
        "rule": (
            "every heavy-rail T1/T2/T5 service that traverses a modelled link, "
            f"stopping or passing (shape within {tolerance} km)"
        ),
        "lines": list(LINES),
        "stations": ["Granville", "Harris Park", "Merrylands"],
        "links": [["Granville", "Harris Park"], ["Granville", "Merrylands"],
                  ["Harris Park", "Merrylands"]],
        "services": services,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtfs", type=Path, default=DEFAULT_GTFS)
    ap.add_argument("--date", default="2026-10-15", help="service date YYYY-MM-DD (a Thursday)")
    ap.add_argument("--window", default="16:00-19:00", help="entry-departure window")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="km a shape must come within to count as passing a station")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    data = build(args.gtfs, args.date, args.window, args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")

    services = data["services"]
    print(f"\n[gtfs] wrote {args.out}: {len(services)} movements in {args.window}")
    by_move = collections.Counter(
        (s["line"], s["direction"], s["origin"], s["destination"]) for s in services
    )
    for (line, direction, o, d), n in sorted(by_move.items()):
        print(f"    {line} {direction:4} {o:12} -> {d:12} {n}")
    print("  by line:  ", collections.Counter(s["line"] for s in services).most_common())
    print("  pass-through (no stop at origin or dest):",
          sum(1 for s in services if not (s["stops_at_origin"] and s["stops_at_destination"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
