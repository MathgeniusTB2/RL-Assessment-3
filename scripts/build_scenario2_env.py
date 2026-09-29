#!/usr/bin/env python3
"""Build the scenario-2 env (Granville-Harris Park) from its rail grid and schedule.
Usage: python scripts/build_scenario2_env.py"""
import json
import math
from collections import deque
from pathlib import Path

from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable

REPO_ROOT = Path(__file__).resolve().parents[1]
RAIL_PATH = REPO_ROOT / "data" / "levels" / "granville_harris_park_rail.mpk"
SCHEDULE_PATH = REPO_ROOT / "data" / "schedules" / "granville_harris_park.json"
OUT_PATH = REPO_ROOT / "data" / "levels" / "granville_harris_park.pkl"

# Same timing model as scenario 1 (build_scenario1_env.py)
SECS_PER_STEP = 20
SPEED_EXPRESS = 1.0
SPEED_LOCAL = 0.5
DEADLINE_EXTRA_STEPS = 0   # scenario 1 deadlines have no extra step either

# Platform cells (row, col), the same cells as the station markers in the rail .mpk
PLATFORMS = {
    "Harris Park": {
        "up_main": (5, 2),
        "down_main": (7, 2),
        "up_suburban": (9, 2),
        "down_suburban": (11, 2)
    },
    "Granville": {
        "up_main": (5, 17),
        "down_main": (7, 17),
        "up_suburban": (9, 17),
        "down_suburban": (11, 17)
    },
    "Merrylands": {
        "up_sw": (14, 15),
        "down_sw": (13, 17)
    }
}


def shortest_path_length(rail, start, direction, target):

    # Breadth-first search over (cell, direction) pairs
    queue = deque([(start, direction, 0)])
    visited = {(start, direction)}

    while queue:
        position, heading, length = queue.popleft()

        if position == target:
            return length

        transitions = rail.get_transitions((position, heading))

        for new_heading in range(4):
            if not transitions[new_heading]:
                continue

            new_position = get_new_position(position, new_heading)
            inside = 0 <= new_position[0] < rail.height and 0 <= new_position[1] < rail.width

            if inside and (new_position, new_heading) not in visited:
                visited.add((new_position, new_heading))
                queue.append((new_position, new_heading, length + 1))

    return None


def start_direction(rail, start, target):

    # The direction to face at the start cell, and the path length from there
    best = None

    for direction in range(4):
        length = shortest_path_length(rail, start, direction, target)
        if length is not None and (best is None or length < best[1]):
            best = (direction, length)

    if best is None:
        raise ValueError(f"No path from {start} to {target}")

    return best


def running_steps(rail, service):

    origin = service["outbound"][0]
    destination = service["outbound"][-1]
    start = PLATFORMS[origin["station"]][origin["platform"]]
    target = PLATFORMS[destination["station"]][destination["platform"]]

    direction, length = start_direction(rail, start, target)
    speed = SPEED_EXPRESS if service["express"] else SPEED_LOCAL

    return start, target, direction, speed, math.ceil(length / speed)


def check_station_markers(rail_dict):

    # The station markers saved by the editor must be exactly the platform cells above
    editor = rail_dict["railenv_editor"]
    origin_x, origin_y = editor["origin"]
    markers = {(y - origin_y, x - origin_x) for x, y in editor["stations"]}

    platform_cells = set()
    for station in PLATFORMS.values():
        platform_cells.update(station.values())

    if markers != platform_cells:
        raise ValueError(f"Station markers {sorted(markers)} do not match PLATFORMS")


def build():

    rail_dict = RailEnvPersister.load_env_dict(str(RAIL_PATH))
    check_station_markers(rail_dict)
    rail_env, _ = RailEnvPersister.load_new(str(RAIL_PATH))

    services = json.loads(SCHEDULE_PATH.read_text())["services"]
    first_departure = min(service["outbound"][0]["dep"] for service in services)

    waypoints = {}
    speeds = []
    earliest = []
    latest = []

    for handle, service in enumerate(services):
        start, target, direction, speed, steps = running_steps(rail_env.rail, service)

        # As in scenario 1: the first train departs at step 1,
        # the latest arrival is the departure plus the running time at the train's speed
        departure = 1 + int(round((service["outbound"][0]["dep"] - first_departure) / SECS_PER_STEP))
        arrival = departure + steps + DEADLINE_EXTRA_STEPS

        waypoints[handle] = [[Waypoint(start, direction)], [Waypoint(target, None)]]
        speeds.append(speed)
        earliest.append([departure, None])
        latest.append([None, arrival])

    last_arrival = max(row[1] for row in latest)
    line = Line(agent_waypoints=waypoints, agent_speeds=speeds)
    timetable = Timetable(
        earliest_departures=earliest,
        latest_arrivals=latest,
        max_episode_steps=max(400, last_arrival + 200)
    )

    env = RailEnv(
        width=len(rail_dict["grid"][0]),
        height=len(rail_dict["grid"]),
        rail_generator=rail_from_file(env_dict=rail_dict),
        line_generator=lambda *args, **kwargs: line,
        timetable_generator=lambda *args, **kwargs: timetable,
        number_of_agents=len(services),
        obs_builder_object=DummyObservationBuilder(),
    )
    env.reset(random_seed=0)
    env.number_of_agents = env.get_num_agents()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RailEnvPersister.save(env, str(OUT_PATH))
    print(f"built {OUT_PATH}: {env.width}x{env.height}, {env.get_num_agents()} agents "
          f"from {len(services)} services, max_episode_steps={env._max_episode_steps}")


if __name__ == "__main__":
    build()
