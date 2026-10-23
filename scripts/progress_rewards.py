"""The reward the Flatland competition RL entries actually trained on.

JBR_HSE (best RL entry, NeurIPS 2020) used

    +0.01 * Δdistance_to_destination  −  5 * is_deadlocked  +  10 * has_finished

and every other strong entry used the same family: dense distance progress, an
explicit deadlock term, and a completion bonus. Nobody trained on the official
scoring function, because it pays almost nothing until the episode ends and its
cancellation term makes "never depart" cheaper than "depart and queue".

This class implements that objective. It deliberately does **not** try to equal
the metric: the agent is trained here and scored with `DefaultRewards`.

Terms
    progress   w * (previous shortest-path time to target - current), per step
    deadlock   penalty every step an on-map train is MOVING but did not move
    arrival    one-off bonus when the train reaches DONE
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from flatland.envs.rewards import Rewards


class ProgressRewards(Rewards):
    """Dense distance-progress reward (competition recipe, see module docstring)."""

    def __init__(self, progress: float = 0.01, deadlock: float = -5.0, arrival: float = 10.0):
        super().__init__()
        self.progress = float(progress)
        self.deadlock = float(deadlock)
        self.arrival = float(arrival)
        self._previous_distance: Dict[int, float] = {}
        self._previous_state: Dict[int, object] = {}
        self._previous_position: Dict[int, object] = {}

    # ------------------------------------------------------------------ setup
    def reset(self, env=None):  # noqa: D401 - optional hook
        self._previous_distance.clear()
        self._previous_state.clear()
        self._previous_position.clear()

    # ---------------------------------------------------------------- rewards
    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps) -> float:
        from flatland.envs.step_utils.states import TrainState

        reward = 0.0
        remaining = float(agent.get_travel_time_on_shortest_path(distance_map))
        previous = self._previous_distance.get(agent.handle)
        if previous is not None:
            reward += self.progress * (previous - remaining)
        self._previous_distance[agent.handle] = remaining

        configuration = agent.current_configuration
        position = None if configuration is None else (
            int(configuration[0][0]), int(configuration[0][1])
        )
        previous_position = self._previous_position.get(agent.handle)
        previous_state = self._previous_state.get(agent.handle)

        if (
            position is not None
            and previous_position is not None
            and position == previous_position
            and agent.state == TrainState.MOVING
        ):
            reward += self.deadlock

        if agent.state == TrainState.DONE and previous_state not in (None, TrainState.DONE):
            reward += self.arrival

        self._previous_position[agent.handle] = position
        self._previous_state[agent.handle] = agent.state
        return reward

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps) -> float:
        return 0.0

    def cumulate(self, *rewards) -> float:
        return float(sum(0.0 if r is None else float(r) for r in rewards))

    def normalize(self, *rewards: np.ndarray, num_agents: int, max_episode_steps: int) -> float:
        values = np.asarray(rewards, dtype=float).ravel()
        return float(np.mean(values)) if values.size else 0.0

    def empty(self) -> float:
        return 0.0
