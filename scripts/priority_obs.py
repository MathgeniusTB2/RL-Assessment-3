"""Priority / conflict features: the discriminator the competition RL agents used.

From the Flatland 2020 competition paper: the best RL entry (JBR_HSE, PPO) put the
occupying agent's **identity** on every tree edge so priority could be reasoned
about, and the runner-up (Netcetera, Ape-X) used a **conflict-graph priority** with
a tree of depth *one* — "priority led to significant improvement". Depth was not
the discriminator; conflict/priority information was.

Our tree reports occupancy *counts* only, so an agent cannot tell "someone is 40
cells ahead on my single-track row" from "there are three trains somewhere near".
This module appends a compact, per-agent conflict block computed once per step
from the env's own state and distance map:

    slack_steps        time-to-deadline minus shortest-path time remaining  /100
    slack_rank         rank of that slack within the fleet (0 = most urgent)
    dist_any_ahead     distance along my route to the first occupied cell     /50
    dist_oncoming      distance along my route to the first opposite-direction
                       agent                                                      /50
    blocked_on_route   number of malfunctioning trains sitting on my route       /5
    malfunction_left   my own remaining malfunction steps                        /50
    speed_fraction     current speed
    on_map             1 while the train is on the grid
    route_len          my remaining route length                                 /100
    queue_ahead        number of STALLED trains (stopped or malfunctioning) on my
                       route ahead of me, i.e. the queue I am about to join       /6
    dist_blocked_ahead distance to the nearest impassable cell on my own row,
                       ahead in my heading (1.0 = nothing blocked)               /50
    dist_commit_ahead  distance to the next commit crossover on my row ahead,
                       i.e. how far I still have to change track (1.0 = none)    /50

The last two make the crossing decision expressible: "the track ahead is
blocked *and* I can still reach a crossover before it". `dist_blocked_ahead`
reads the exogenous blockage registry (`blockage.active_cells`), because a
zeroed rail cell is invisible to the occupancy-based features above.

Everything is clipped to roughly [-1, 1] so it can share the MLP input with the
already-normalised tree.
"""

from __future__ import annotations

import numpy as np

from regimes import next_commit_ahead

N_PRIORITY = 12
HORIZON = 50  # cells of route inspected


def _route_ahead(path, position, horizon=HORIZON):
    """Cells strictly ahead of ``position`` along ``path`` (bounded)."""
    cells = []
    started = False
    for waypoint in path:
        cell = (int(waypoint.position[0]), int(waypoint.position[1]))
        if not started:
            if cell == position:
                started = True
            continue
        cells.append(cell)
        if len(cells) >= horizon:
            break
    if not started:  # position not on the cached path (e.g. just re-routed)
        for waypoint in path[:horizon]:
            cells.append((int(waypoint.position[0]), int(waypoint.position[1])))
    return cells


def install_priority_features(nb, horizon: int = HORIZON) -> None:
    """Wrap the current observation builder to append the priority block."""
    import gymnasium as gym

    base_cls = nb["GymTreeObsForRailEnv"]
    if getattr(base_cls, "n_priority", 0):
        # idempotent: the deliverable notebook installs this block as well, and
        # nb_harness execs that cell before a CLI trainer asks for it again
        return

    class PriorityObs(base_cls):
        n_priority = N_PRIORITY

        def reset(self, env):
            super().reset(env)
            self._priority = {}

        def get_many(self, handles=None):
            self._priority = self._compute_priority(self.env, horizon)
            return super().get_many(handles)

        @staticmethod
        def _compute_priority(env, horizon):

            blocked_cells = set(getattr(env, "_blocked_cells", ()) or ())
            try:
                paths = env.distance_map.get_shortest_paths()
            except Exception:
                paths = {}

            handles = env.get_agent_handles()
            occupancy = {}
            down_cells = set()
            on_map = {}
            for handle in handles:
                agent = env.agents[handle]
                configuration = agent.current_configuration
                if configuration is None or agent.state.is_off_map_state():
                    on_map[handle] = False
                    continue
                on_map[handle] = True
                cell = (int(configuration[0][0]), int(configuration[0][1]))
                from flatland.envs.step_utils.states import TrainState as _TS
                stalled = agent.state in (_TS.STOPPED, _TS.MALFUNCTION)
                occupancy[cell] = (int(configuration[1]), stalled)
                if agent.malfunction_handler.malfunction_down_counter > 0:
                    down_cells.add(cell)

            # fleet-wide slack ranking (priority = earliest deadline first)
            slack = {}
            for handle in handles:
                agent = env.agents[handle]
                if not on_map[handle] or agent.waypoints_latest_arrival[-1] is None:
                    slack[handle] = -1e9
                    continue
                remaining = agent.get_travel_time_on_shortest_path(env.distance_map)
                slack[handle] = float(
                    agent.waypoints_latest_arrival[-1] - env._elapsed_steps - remaining
                )
            ordered = sorted(handles, key=lambda h: -slack[h])
            rank = {handle: i / max(1, len(ordered) - 1) for i, handle in enumerate(ordered)}

            block = {}
            for handle in handles:
                agent = env.agents[handle]
                features = np.zeros(N_PRIORITY, dtype=np.float64)
                features[0] = np.clip(slack[handle] / 100.0, -1.0, 1.0)
                features[1] = rank[handle]
                features[5] = np.clip(
                    agent.malfunction_handler.malfunction_down_counter / 50.0, 0.0, 1.0
                )
                features[7] = 1.0 if on_map[handle] else 0.0
                if on_map[handle]:
                    configuration = agent.current_configuration
                    position = (int(configuration[0][0]), int(configuration[0][1]))
                    row = position[0]
                    heading = int(configuration[1])
                    route = _route_ahead(paths.get(handle) or [], position, horizon)
                    features[8] = np.clip(len(route) / 100.0, 0.0, 1.0)
                    distance_any = horizon
                    distance_oncoming = horizon
                    blocked = 0
                    queue_ahead = 0
                    for offset, cell in enumerate(route, start=1):
                        if cell in down_cells:
                            blocked += 1
                        entry = occupancy.get(cell)
                        if entry is None:
                            continue
                        direction, stalled = entry
                        distance_any = min(distance_any, offset)
                        if direction == (heading + 2) % 4:
                            distance_oncoming = min(distance_oncoming, offset)
                        if stalled:
                            queue_ahead += 1
                    features[2] = np.clip(distance_any / horizon, 0.0, 1.0)
                    features[3] = np.clip(distance_oncoming / horizon, 0.0, 1.0)
                    features[4] = np.clip(blocked / 5.0, 0.0, 1.0)
                    features[6] = float(agent.speed_counter.speed)
                    features[9] = min(queue_ahead / 6.0, 1.0)
                    # exogenous blockage / remaining chance to change track
                    step = 1 if heading == 1 else -1 if heading == 3 else 0
                    distance_blocked = horizon
                    if step and blocked_cells:
                        for offset in range(1, horizon + 1):
                            column = position[1] + step * offset
                            if column < 0:
                                break
                            if (row, column) in blocked_cells:
                                distance_blocked = offset
                                break
                    features[10] = np.clip(distance_blocked / horizon, 0.0, 1.0)
                    commit = (next_commit_ahead(row, position[1], heading, horizon)
                              if step else None)
                    features[11] = np.clip(
                        (commit if commit is not None else horizon) / horizon, 0.0, 1.0
                    )
                block[handle] = features
            return block

        def get(self, handle: int = 0):
            base = np.asarray(super().get(handle), dtype=np.float64)
            extra = self._priority.get(handle)
            if extra is None:
                extra = np.zeros(N_PRIORITY, dtype=np.float64)
            return np.concatenate((base, extra))

        def get_observation_space(self, handle: int = 0) -> gym.Space:
            base = super().get_observation_space(handle)
            return gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(int(np.prod(base.shape)) + N_PRIORITY,),
                dtype=np.float64,
            )

    nb["GymTreeObsForRailEnv"] = PriorityObs
