"""Scenario 2 (Granville-Harris Park) environment assembly.

Extracted verbatim from notebooks/Scenario2/Scenario2.ipynb so the submitted
notebook has no second-notebook dependency. Every block below is the source of
one cell of that study, in the order s2_env.py used to exec them:
    (6, 8, 9, 10, 12, 13, 14, 15, 16, 19, 18, 20)
"""


# --- Scenario2.ipynb cell 6 ----------------------------------------------
import json
from pathlib import Path

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.step_utils.states import TrainState
from flatland.core.effects_generator import EffectsGenerator, make_multi_effects_generator
from flatland.envs.malfunction_effects_generators import MalfunctionEffectsGenerator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

from flatland.utils.rendertools import RenderTool
from IPython.display import display, clear_output
from PIL import Image
import PIL

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.timetable_utils import Line
from flatland.envs.rail_trainrun_data_structures import Waypoint

from flatland.envs.timetable_utils import Timetable
from collections import Counter, deque
import statistics
import time


# --- Scenario2.ipynb cell 8 ----------------------------------------------
# --------------------------------------------------
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


# --- Scenario2.ipynb cell 9 ----------------------------------------------
# --------------------------------------------------
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


# --- Scenario2.ipynb cell 10 ----------------------------------------------
# --------------------------------------------------
# Defining the disruption
# --------------------------------------------------
# Random malfunctions of the "delayed" condition are added in load_environment().
# Fixed breakdowns can still be added here for demonstrations:
# "train"    = index of the train in REALISTIC_TIMETABLE_CONFIG
# "step"     = step (as printed by run_manual_test) at which it breaks down
# "duration" = number of steps it stays broken down

DISRUPTION_CONFIG = []


# --- Scenario2.ipynb cell 12 ----------------------------------------------
class Scenario2Effects(EffectsGenerator):
    """
    Hooks into env.reset() and env.step() to:
    1. re-apply the level-free crossing. env.reset(regenerate_rail=False) rebuilds
       the resource map and drops the level-free cell loaded from the .mpk file
    2. trigger the scheduled disruptions (train breakdowns)
    """

    def __init__(self, level_free_positions=(), disruptions=None):
        super().__init__()
        self.level_free_positions = set(level_free_positions)
        self.disruptions = disruptions or []

    def on_episode_start(self, env, *args, **kwargs):
        env.resource_map.level_free_positions = set(self.level_free_positions)
        return env

    def on_episode_step_start(self, env, *args, **kwargs):
        # env._elapsed_steps has already been incremented here, so
        # "Step 0" in run_manual_test is env._elapsed_steps == 1
        step = env._elapsed_steps - 1

        for disruption in self.disruptions:
            agent = env.agents[disruption["train"]]

            if disruption["step"] == step and agent.state != TrainState.DONE:
                agent.malfunction_handler.malfunction_down_counter = disruption["duration"]

        return env


# --- Scenario2.ipynb cell 13 ----------------------------------------------
def load_environment(level_free_crossing=LEVEL_FREE_CROSSING, disruptions=None, malfunction_rate=None):

    if level_free_crossing:
        level_free_positions = [JUNCTIONS["flyover_a"], JUNCTIONS["flyover_b"]]
    else:
        level_free_positions = []

    effects = Scenario2Effects(level_free_positions, disruptions)

    # "delayed" condition: Poisson malfunctions, the same generator as Scenario 1
    if malfunction_rate is not None:
        malfunctions = MalfunctionEffectsGenerator(
            ParamMalfunctionGen(
                MalfunctionParameters(
                    malfunction_rate=malfunction_rate,
                    min_duration=MALFUNCTION_DURATION[0],
                    max_duration=MALFUNCTION_DURATION[1]
                )
            )
        )
        effects = make_multi_effects_generator(effects, malfunctions)

    env, env_dict = RailEnvPersister.load_new(
        str(ENV_FILE),
        obs_builder=GlobalObsForRailEnv(),
        effects_generator=effects
    )

    # The .mpk loads seed_history as a tuple, so reset(random_seed=...) cannot
    # append to it. Use a list (same fix as the Scenario 1 loader)
    env.seed_history = list(env.seed_history) or [0]

    return env


# --- Scenario2.ipynb cell 14 ----------------------------------------------
def shortest_running_time(env, start, direction, target):
    """
    Minimum number of moves from start (facing direction) to target,
    found with a breadth-first search over the rail transitions.
    """
    moves = {
        NORTH: (-1, 0),
        EAST: (0, 1),
        SOUTH: (1, 0),
        WEST: (0, -1)
    }

    queue = deque([(start, direction, 0)])
    visited = {(start, direction)}

    while queue:
        position, heading, distance = queue.popleft()

        if position == target:
            return distance

        transitions = env.rail.get_transitions((position, heading))

        for new_heading in range(4):
            if transitions[new_heading]:
                d_row, d_col = moves[new_heading]
                new_position = (position[0] + d_row, position[1] + d_col)

                if (new_position, new_heading) not in visited:
                    visited.add((new_position, new_heading))
                    queue.append((new_position, new_heading, distance + 1))

    raise ValueError(f"Target {target} cannot be reached from {start} facing {direction}")


# --- Scenario2.ipynb cell 15 ----------------------------------------------
def create_agents(train_config):

    agent_waypoints = {}

    # Create waypoints for each train
    for handle, train in enumerate(train_config):

        agent_waypoints[handle] = [
            [
               Waypoint(
                    position=train["start"],
                    direction=train["direction"]
                )
            ],
            [
                Waypoint(
                    position=train["target"],
                    direction=None
                )
            ]
        ]

    # Give every train normal speed
    agent_speeds = [1.0] * len(train_config)

    # Create the line
    line = Line(
        agent_waypoints=agent_waypoints,
        agent_speeds=agent_speeds
    )

    # Create Flatland agents
    agents = EnvAgent.from_line(line)

    return agents


# --- Scenario2.ipynb cell 16 ----------------------------------------------
def create_timetable(train_config, env):

    earliest_departures = []
    latest_arrivals = []

    for train in train_config:

        # Arrival on a clear line: Flatland puts a train on the map at
        # max(departure, 1) and it arrives one step after entering its target.
        # This is replaced by the calibrated deadline in setup_environment().
        running_time = shortest_running_time(
            env, train["start"], train["direction"], train["target"]
        )
        running_time += len(train["stops"])

        earliest_departures.append(
            [train["departure"], None]
        )

        latest_arrivals.append(
            [None, max(train["departure"], 1) + running_time + 1]
        )

    max_episode_steps = max(arrival for _, arrival in latest_arrivals) + EXTRA_STEPS

    timetable = Timetable(
        earliest_departures=earliest_departures,
        latest_arrivals=latest_arrivals,
        max_episode_steps=max_episode_steps
    )

    return timetable


# --- Scenario2.ipynb cell 19 ----------------------------------------------
def run_manual_test(env, agents, train_config, max_steps=None,
                    render=True, delay=2, verbose=True):

    if max_steps is None:
        max_steps = env._max_episode_steps

    # Define the renderer
    env_renderer = RenderTool(env) if render else None

    # Keep track of stations each agent has already stopped at
    completed_stops = {}
    for agent in agents:
        completed_stops[agent.handle] = set()

    for step in range(max_steps):

        actions = {}

        # ---------------------------------------------
        # Determine action for each agent
        # ---------------------------------------------
        for agent in agents:

            handle = agent.handle
            train = train_config[handle]

            # Default action = move forward
            actions[handle] = RailEnvActions.MOVE_FORWARD

            # Agent may be off the map because it has
            # not departed yet or has already finished
            if agent.current_configuration is None:
                continue

            # Get current position
            current_position = agent.current_configuration[0]

            # -----------------------------------------
            # Check if this is a scheduled station stop
            # -----------------------------------------
            if (
                current_position in train["stops"]
                and current_position not in completed_stops[handle]
            ):

                # Stop train for this timestep
                actions[handle] = RailEnvActions.STOP_MOVING

                # Remember that this stop has been completed
                completed_stops[handle].add(current_position)

                if verbose:
                    print(
                        f"Agent {handle} stopping at "
                        f"{current_position}"
                    )

            # -----------------------------------------
            # Check if the train has to turn here
            # (junction / crossover on its route)
            # -----------------------------------------
            elif current_position in train.get("route", {}):

                actions[handle] = train["route"][current_position]

                if verbose:
                    print(
                        f">>> Agent {handle} taking "
                        f"{actions[handle].name} at {current_position}"
                    )

        # ---------------------------------------------
        # Take next action
        # ---------------------------------------------
        observations, rewards, dones, info = env.step(actions)

        # ---------------------------------------------
        # Render environment
        # ---------------------------------------------
        if render:
            render_env(env, env_renderer=env_renderer)

            # Pause so movement is visible
            time.sleep(delay)

            # ---------------------------------------------
            # Print agent information
            # ---------------------------------------------
            print(f"\nStep {step}")

            for agent in env.agents:
                malfunction = agent.malfunction_handler.malfunction_down_counter
                print(
                    f"Agent {agent.handle}:",
                    agent.current_configuration,
                    "state:",
                    agent.state.name,
                    f"(broken down for {malfunction} more steps)" if malfunction > 0 else ""
                )

        # ---------------------------------------------
        # End episode when all agents are finished
        # ---------------------------------------------
        if dones["__all__"]:
            if verbose:
                print("All agents finished!")
            break


# --- Scenario2.ipynb cell 18 ----------------------------------------------
_REFERENCE_ARRIVALS = {}


def reference_arrivals(train_config):
    """
    Arrival step of every train when all trains follow their scripted route
    under normal conditions (no disruptions). Used to set reachable deadlines,
    the same way Scenario 1 calibrates its timetable.
    """
    key = id(train_config)

    if key not in _REFERENCE_ARRIVALS:
        env, agents = setup_environment(train_config, condition="normal", calibrate=False, verbose=False)
        env.reset(regenerate_rail=False, regenerate_schedule=False)
        run_manual_test(env, agents, train_config, render=False, verbose=False)
        _REFERENCE_ARRIVALS[key] = {agent.handle: agent.arrival_time for agent in env.agents}

    return _REFERENCE_ARRIVALS[key]


def setup_environment(train_config, level_free_crossing=LEVEL_FREE_CROSSING,
                      disruptions=None, condition="normal", calibrate=True, verbose=True):

    # Load environment
    env = load_environment(level_free_crossing, disruptions, CONDITIONS[condition])

    # Create the agents
    agents = create_agents(train_config)

    # Create the timetable
    timetable = create_timetable(train_config, env)

    # Apply the timetable
    agents = EnvAgent.apply_timetable(agents, timetable)

    # Calibrate the deadlines: reference arrival under normal conditions + slack
    if calibrate:
        arrivals = reference_arrivals(train_config)
        for agent in agents:
            measured = arrivals.get(agent.handle)
            if measured is not None:
                deadline = max(agent.latest_arrival, measured) + DEADLINE_SLACK
                agent.waypoints_latest_arrival = [None, deadline]
                agent.latest_arrival = deadline

    # Add fully configured agents to environment
    for agent in agents:
        env.add_agent(agent)

    # The .mpk keeps its own max_episode_steps (500), so apply our own value
    env._max_episode_steps = max(agent.latest_arrival for agent in agents) + EXTRA_STEPS

    # Initialise done status
    env.dones = {
        agent.handle: False
        for agent in agents
    }

    env.dones["__all__"] = False

    if not verbose:
        return env, agents

    # Print information
    departures = [train["real_departure"] for train in train_config]
    print("Number of agents:", len(env.agents))
    print("Lines:", dict(Counter(train["line"] for train in train_config)))
    print("Departures:", min(departures), "to", max(departures))
    print("Condition:", condition, "| Level-free crossing:", level_free_crossing)
    print("Fixed disruptions:", disruptions)
    print("Max episode steps:", env._max_episode_steps)

    return env, agents


# --- Scenario2.ipynb cell 20 ----------------------------------------------
def print_results(env, train_config, show_trains=True):
    """
    Delay = steps late compared with the timetable (latest arrival)
    Lost  = steps lost to conflicts / disruptions compared with a clear run
    """

    if show_trains:
        print(f"{'Agent':<7}{'Line':<6}{'Real dep':>9}{'Departure':>10}{'Arrival':>9}{'Latest':>8}{'Delay':>7}{'Lost':>6}   State")

    total_delay = 0
    total_lost = 0
    arrived = 0
    on_time = 0

    for agent in env.agents:
        train = train_config[agent.handle]

        # Arrival on a clear line: Flatland puts a train on the map at
        # max(departure, 1) and it arrives one step after entering its target
        running_time = shortest_running_time(
            env, train["start"], train["direction"], train["target"]
        )
        running_time += len(train["stops"])
        clear_run_arrival = max(train["departure"], 1) + running_time + 1

        if agent.arrival_time is None:
            arrival = "-"
            delay = "-"
            lost = "-"
        else:
            arrived += 1
            arrival = agent.arrival_time
            delay = max(0, agent.arrival_time - agent.latest_arrival)
            lost = agent.arrival_time - clear_run_arrival
            total_delay += delay
            total_lost += lost
            on_time += delay == 0

        if show_trains:
            print(
                f"{agent.handle:<7}{train['line']:<6}{train['real_departure']:>9}{agent.earliest_departure:>10}"
                f"{arrival:>9}{agent.latest_arrival:>8}{delay:>7}{lost:>6}   {agent.state.name}"
            )

    n = len(env.agents)
    print()
    print(f"Trains arrived: {arrived}/{n}")
    print(f"On time: {on_time}/{n}")
    print(f"Total delay: {total_delay} steps ({total_delay * STEP_SECONDS / 60:.1f} min)")
    print(f"Mean delay per arrived train: {total_delay / max(arrived, 1):.2f} steps")
    print(f"Total time lost to conflicts / disruptions: {total_lost} steps ({total_lost * STEP_SECONDS / 60:.1f} min)")

    return {"arrived": arrived, "on_time": on_time, "total_delay": total_delay, "total_lost": total_lost}
