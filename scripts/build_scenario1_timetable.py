#!/usr/bin/env python3
"""Extract a real TfNSW timetable for scenario 1 (Central-Strathfield corridor).

Reads a Greater Sydney GTFS bundle and emits a compact JSON of every heavy-rail
service that crosses the modelled corridor (Central, Redfern, Burwood,
Strathfield) inside a time window, with real per-stop arrival/departure times.

The scenario grid only models four stations, so intermediate inner-west stops
(e.g. Newtown...Croydon served by T2/T3) are represented by their Redfern /
Burwood stops only.

Usage:
    python scripts/build_scenario1_timetable.py \
        --gtfs "/Users/.../RL - rail project/grid_builder/gtfs" \
        --date 2026-09-24 --window 17:00-18:00 \
        --out data/schedules/central_strathfield_one_way.json

The GTFS bundle is large (~3.4 GB unpacked); ``stop_times.txt`` is streamed once.
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
DEFAULT_OUT = REPO_ROOT / "data" / "schedules" / "central_strathfield_one_way.json"

# line (+ express) -> track pair on the scenario grid.
LINE_TRACKS = {
    "T1": "suburban",
    "T2": "local",
    "T3": "local",
    "CCN": "main",
    "BMT": "main",
    "T9": "suburban",
}


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


def _platform_map(stops_path: Path) -> dict[str, str]:
    """Child platform stop_id -> modelled parent station name."""
    child2name: dict[str, str] = {}
    with open(stops_path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            parent = row.get("parent_station", "")
            if parent in PARENT:
                child2name[row["stop_id"]] = PARENT[parent]
    return child2name


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


def build(gtfs_dir: Path, date: str, window: str) -> dict:
    date = date.replace("-", "")  # calendar uses YYYYMMDD
    child2name = _platform_map(gtfs_dir / "stops.txt")
    active = _active_services(
        gtfs_dir / "calendar.txt", gtfs_dir / "calendar_dates.txt", date
    )
    short_by_id, routes = _rail_routes(gtfs_dir / "routes.txt")

    trips: dict[str, tuple[str, str]] = {}
    with open(gtfs_dir / "trips.txt", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["service_id"] in active and row["route_id"] in routes:
                trips[row["trip_id"]] = (
                    short_by_id[row["route_id"]],
                    row["direction_id"],
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
    services = []
    for tid, stops in seq.items():
        stops.sort()
        my = [
            (child2name[sid], _secs(at), _secs(dp))
            for _, sid, at, dp in stops
            if sid in child2name
        ]
        ded: list[tuple[str, int | None, int | None]] = []
        for name, arr, dep in my:
            if ded and ded[-1][0] == name:
                ded[-1] = (name, arr, dep)
            else:
                ded.append((name, arr, dep))
        names = [d[0] for d in ded]
        if "Central" not in names or "Strathfield" not in names:
            continue

        origin_dep = ded[0][2] if ded[0][2] is not None else ded[0][1]
        if origin_dep is None or not (w0 <= origin_dep < w1):
            continue

        line, _ = trips[tid]
        direction = "down" if names.index("Central") < names.index("Strathfield") else "up"
        express = "Redfern" not in names and "Burwood" not in names
        track = _track_for(line, direction, express)

        services.append(
            {
                "id": f"{line}-{direction}-{len(services):02d}",
                "line": line,
                "direction": direction,
                "track": track,
                "express": express,
                "stops": [
                    {"station": name, "arr": arr, "dep": dep}
                    for name, arr, dep in ded
                ],
            }
        )

    services.sort(key=lambda s: s["stops"][0]["dep"] or s["stops"][0]["arr"])
    return {
        "name": f"Central-Strathfield corridor, weekday {date}, {window}",
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
    ap.add_argument(
        "--date",
        default="2026-09-24",
        help="service date YYYY-MM-DD (default is a normal Thursday; some dates "
        "are trackwork-affected in the feed)",
    )
    ap.add_argument("--window", default="17:00-18:00", help="origin-departure window")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    data = build(args.gtfs, args.date, args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + "\n")

    services = data["services"]
    print(f"[gtfs] wrote {args.out}: {len(services)} services in {args.window}")
    if len(services) < 10:
        print(
            "  WARNING: very few services - this date may be trackwork-affected "
            "in the feed; try another Thursday."
        )
    print("  by line:", Counter(s["line"] for s in services).most_common())
    print(
        "  by direction:",
        Counter(s["direction"] for s in services).most_common(),
    )
    print("  by track:", Counter(s["track"] for s in services).most_common())
    print("  express:", sum(1 for s in services if s["express"]))
    for s in services[:5]:
        times = " | ".join(
            f"{st['station']}@{_hms(st['dep'] or st['arr'])}" for st in s["stops"]
        )
        print(f"    {s['id']:12} {s['track']:14} {times}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
