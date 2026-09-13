"""Paper-style reward for Flatland (Mohanty et al. 2020, arXiv:2012.05893).

Equation 1 in the paper gives each agent a local reward of -1 per step while it
is moving or stopped, 0 once it reaches its target, plus an optional global +1
when all agents are done and a (zero, in the benchmark) illegal-move penalty.

The modern Flatland ``DefaultRewards`` instead awards partial credit for
intermediate stops and penalises lateness/early departure, which decouples the
normalized score from the completion rate. This class restores the paper's
simpler objective so that maximising the score means completing trains.
"""

from __future__ import annotations

import numpy as np

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.rewards import Rewards
from flatland.envs.step_utils.env_utils import AgentTransitionData
from flatland.envs.step_utils.states import TrainState


class PaperRewards(Rewards[float]):
    def step_reward(
        self,
        agent: EnvAgent,
        agent_transition_data: AgentTransitionData,
        distance_map: DistanceMap,
        elapsed_steps: int,
    ) -> float:
        return 0.0 if agent.state == TrainState.DONE else -1.0

    def end_of_episode_reward(
        self, agent: EnvAgent, distance_map: DistanceMap, elapsed_steps: int
    ) -> float:
        return 0.0

    def cumulate(self, *rewards: float) -> float:
        return float(sum(rewards))

    def empty(self) -> float:
        return 0.0

    def normalize(
        self, *rewards: np.ndarray, num_agents: int, max_episode_steps: int
    ) -> float:
        if len(rewards) == num_agents:
            sum_per_agent = np.array(rewards, dtype=float)
        else:
            sum_per_agent = np.sum(
                np.reshape(np.array(rewards, dtype=float), (num_agents, -1), order="F"),
                axis=1,
            )
        capped = np.maximum(sum_per_agent, -max_episode_steps)
        return float(sum(capped) / (max_episode_steps * num_agents) + 1)
