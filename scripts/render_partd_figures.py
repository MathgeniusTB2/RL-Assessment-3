#!/usr/bin/env python3
"""Render the Part-D (Environment/Interface Design) figures.

Produces headless PIL/SVG renders of the two project scenarios for the
Assignment-3 Part-D report:

    docs/partd_scenario1_central_strathfield.png   full 92x37 corridor
    docs/partd_scenario1_corridor_zoom.png         zoom on a busy station area
    docs/partd_scenario1_delayed_malfunction.png   delayed condition, train held
    docs/partd_scenario2_granville_harris_park.png full 20x20 junction
    docs/partd_scenario2_junction_zoom.png         zoom on the merge / diamond

Both scenarios are the real project envs already stored under ``data/``; the
script only loads them, rolls out a scripted forward policy for a warm-up and
renders one frame, so the figures are reproducible from the repository root:

    python scripts/render_partd_figures.py

Scenario 1 uses the prebuilt env (``data/levels/central_strathfield.pkl``).
Scenario 2 uses the rail grid (``data/levels/scenario2_env.mpk``) plus the real
T2 timetable (``data/schedules/scenario2_timetable_20261015.json``), rebuilt
here with the same agent/timetable code as ``notebooks/Scenario2/Scenario2.ipynb``.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions as A
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"

S1_LEVEL = REPO_ROOT / "data" / "levels" / "central_strathfield.pkl"
S2_LEVEL = REPO_ROOT / "data" / "levels" / "scenario2_env.mpk"
S2_TIMETABLE = REPO_ROOT / "data" / "schedules" / "scenario2_timetable_20261015.json"

STEP_SECONDS = 20

# ---------------------------------------------------------------------------
# Scenario 2 layout (mirrors notebooks/Scenario2/Scenario2.ipynb)
# ---------------------------------------------------------------------------
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

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
        "up": (18, 10),
        "down": (18, 11),
    },
}

JUNCTIONS = {
    "oms_diverge": (7, 12),
    "ylink_diverge": (7, 5),
    "flyover_a": (9, 9),
    "flyover_b": (10, 9),
}

SEGMENT_MAP = {
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
    ("T2", "Granville", "Merrylands"): {
        "start": STATIONS["Granville"]["down_suburban"],
        "target": STATIONS["Merrylands"]["down"],
        "direction": WEST,
        "route": {
            JUNCTIONS["oms_diverge"]: A.MOVE_LEFT,
            (10, 12): A.MOVE_RIGHT,
            (10, 11): A.MOVE_LEFT,
        },
    },
    ("T2", "Merrylands", "Granville"): {
        "start": STATIONS["Merrylands"]["up"],
        "target": STATIONS["Granville"]["up_suburban"],
        "direction": NORTH,
        "route": {
            (13, 10): A.MOVE_LEFT,
            (13, 9): A.MOVE_RIGHT,
            (8, 9): A.MOVE_RIGHT,
            (8, 11): A.MOVE_LEFT,
            (6, 11): A.MOVE_RIGHT,
        },
    },
    ("T5", "Harris Park", "Merrylands"): {
        "start": STATIONS["Harris Park"]["down_suburban"],
        "target": STATIONS["Merrylands"]["down"],
        "direction": EAST,
        "route": {
            JUNCTIONS["ylink_diverge"]: A.MOVE_RIGHT,
            (9, 5): A.MOVE_LEFT,
            (9, 11): A.MOVE_RIGHT,
        },
    },
    ("T5", "Merrylands", "Harris Park"): {
        "start": STATIONS["Merrylands"]["up"],
        "target": STATIONS["Harris Park"]["down_suburban"],
        "direction": NORTH,
        "route": {
            (10, 10): A.MOVE_LEFT,
            (10, 3): A.MOVE_RIGHT,
            (7, 3): A.MOVE_LEFT,
        },
    },
}


def _to_step(clock: str, start: str = "16:00") -> int:
    h, m, s = (int(x) for x in clock.split(":"))
    sh, sm = (int(x) for x in start.split(":"))
    return round(((h * 3600 + m * 60 + s) - (sh * 3600 + sm * 60)) / STEP_SECONDS)


def _load_s2_config() -> list[dict]:
    services = json.loads(S2_TIMETABLE.read_text())["services"]
    config = []
    for service in services:
        segment = SEGMENT_MAP[(service["line"], service["origin"], service["destination"])]
        config.append(
            {
                "line": service["line"],
                "start": segment["start"],
                "target": segment["target"],
                "direction": segment["direction"],
                "departure": _to_step(service["departure"]),
                "stops": [],
                "route": segment.get("route", {}),
            }
        )
    return config


def _load_scenario2_env():
    env, env_dict = RailEnvPersister.load_new(
        str(S2_LEVEL), obs_builder=GlobalObsForRailEnv()
    )
    env.seed_history = list(env.seed_history) or [0]
    return env


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _render(env, screen_width: int, out_path: Path) -> Image.Image:
    height = max(200, int(screen_width * env.height / max(env.width, 1)))
    renderer = RenderTool(env, gl="PILSVG", screen_width=screen_width, screen_height=height)
    frame = renderer.render_env(
        show=False, show_agents=True, show_observations=False, return_image=True
    )
    renderer.close_window()
    img = Image.fromarray(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({img.width}x{img.height})")
    return img


def _zoom(img: Image.Image, box_frac, scale: float, out_path: Path) -> None:
    """Crop ``img`` to a fractional box and upscale (nearest) for legibility."""
    w, h = img.size
    x0, y0, x1, y1 = box_frac
    crop = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({crop.width}x{crop.height})")


def _active(env) -> int:
    return sum(
        1
        for a in env.agents
        if a.current_configuration is not None and a.state not in (TrainState.DONE, TrainState.WAITING)
    )


# ---------------------------------------------------------------------------
# Scenario 1 (Central-Strathfield, prebuilt env)
# ---------------------------------------------------------------------------
def scenario1(warmup: int = 300, seed: int = 42, width: int = 1600) -> Image.Image:
    env, _ = RailEnvPersister.load_new(str(S1_LEVEL))
    env.seed_history = list(env.seed_history) or [0]
    env.reset(random_seed=seed)
    for _ in range(warmup):
        env.step({h: A.MOVE_FORWARD.value for h in env.get_agent_handles()})
    print(f"S1: {_active(env)}/{len(env.agents)} trains active after {warmup} steps")
    return _render(env, width, DOCS / "partd_scenario1_central_strathfield.png")


def scenario1_delayed(warmup: int = 220, seed: int = 7, width: int = 1600) -> Image.Image:
    env, _ = RailEnvPersister.load_new(str(S1_LEVEL))
    env.seed_history = list(env.seed_history) or [0]
    env.reset(random_seed=seed)
    for _ in range(warmup):
        env.step({h: A.MOVE_FORWARD.value for h in env.get_agent_handles()})
        # Once a handful of trains are running, break a few to emulate the
        # delayed condition (Poisson malfunctions, 20-50 step holds).
    active = [a for a in env.agents if a.current_configuration is not None]
    broken = 0
    for agent in active[:: max(1, len(active) // 5)]:
        agent.malfunction_handler.malfunction_down_counter = 30
        broken += 1
        if broken >= 5:
            break
    env.step({h: A.MOVE_FORWARD.value for h in env.get_agent_handles()})
    print(f"S1 delayed: {_active(env)} trains active, {broken} held by a malfunction")
    return _render(env, width, DOCS / "partd_scenario1_delayed_malfunction.png")


# ---------------------------------------------------------------------------
# Scenario 2 (Granville-Harris Park, real T2 timetable on the 20x20 grid)
# ---------------------------------------------------------------------------
def _s2_create_agents(config: list[dict]) -> list[EnvAgent]:
    waypoints = {
        h: [
            [Waypoint(position=t["start"], direction=t["direction"])],
            [Waypoint(position=t["target"], direction=None)],
        ]
        for h, t in enumerate(config)
    }
    line = Line(agent_waypoints=waypoints, agent_speeds=[1.0] * len(config))
    return EnvAgent.from_line(line)


def _s2_create_timetable(config: list[dict]) -> Timetable:
    earliest = [[t["departure"], None] for t in config]
    latest = [[None, t["departure"] + 300] for t in config]
    max_steps = max(t["departure"] for t in config) + 200
    return Timetable(
        earliest_departures=earliest,
        latest_arrivals=latest,
        max_episode_steps=max_steps,
    )


def _s2_scripted_actions(env, config: list[dict]) -> dict:
    actions = {}
    for agent in env.agents:
        actions[agent.handle] = A.MOVE_FORWARD.value
        if agent.current_configuration is None:
            continue
        position = agent.current_configuration[0]
        route = config[agent.handle].get("route", {})
        if position in route:
            actions[agent.handle] = route[position].value
    return actions


def scenario2(seed: int = 0, width: int = 900) -> Image.Image:
    """Render scenario 2 at the busiest instant of a normal-condition episode.

    The 20x20 segment is short and the real timetable is sparse (126 services
    spread over the 16:00-19:00 window), so only a few trains share the map at
    once. A first pass finds the step with the most trains on the map; the second
    pass (same seed, deterministic) re-runs to that step and renders it.
    """
    config = _load_s2_config()

    def _make_and_reset():
        env = _load_scenario2_env()
        agents = _s2_create_agents(config)
        EnvAgent.apply_timetable(agents, _s2_create_timetable(config))
        for agent in agents:
            env.add_agent(agent)
        env._max_episode_steps = max(t["departure"] for t in config) + 200
        env.dones = {a.handle: False for a in env.agents}
        env.dones["__all__"] = False
        env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=seed)
        return env

    env = _make_and_reset()
    best_step, best_count = 0, 0
    step = 0
    while True:
        step += 1
        _, _, dones, _ = env.step(_s2_scripted_actions(env, config))
        count = _active(env)
        if count >= best_count:
            best_step, best_count = step, count
        if dones.get("__all__") or step >= 700:
            break

    env = _make_and_reset()
    for _ in range(best_step):
        env.step(_s2_scripted_actions(env, config))
    print(f"S2: busiest instant is step {best_step} with {_active(env)}/{len(env.agents)} trains")
    return _render(env, width, DOCS / "partd_scenario2_granville_harris_park.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1-warmup", type=int, default=300)
    args = ap.parse_args()

    s1 = scenario1(warmup=args.s1_warmup)
    _zoom(s1, (0.28, 0.55, 0.72, 1.0), 2.2, DOCS / "partd_scenario1_corridor_zoom.png")

    scenario1_delayed()

    s2 = scenario2()
    _zoom(s2, (0.25, 0.18, 1.0, 0.85), 2.0, DOCS / "partd_scenario2_junction_zoom.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
