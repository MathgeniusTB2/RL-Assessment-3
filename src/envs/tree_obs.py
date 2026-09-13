"""Gym-compatible flattened/normalized tree observation for Flatland.

This is a ray-free re-implementation of
``flatland.ml.observations.flatten_tree_observation_for_rail_env``
(which imports RLlib purely for a type hint). It keeps the dependency set to
the repo's stated stack (Flatland + SB3) instead of pulling in all of Ray.

Adapted from https://github.com/instadeepai/Mava (Apache-2.0).
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from flatland.core.env_observation_builder import AgentHandle
from flatland.envs.observations import Node, TreeObsForRailEnv


class GymTreeObsForRailEnv(TreeObsForRailEnv):
    """Flattened, normalized tree observation with a gym observation space.

    The tree is grouped into three feature groups (data, distance, agent data),
    each traversed depth-first in pre-order, then concatenated and normalized.
    """

    NUM_FEATURES = 12
    NUM_BRANCHES = 4
    NUM_DATA_FEATURE_GROUP = 6
    NUM_DISTANCE_FEATURE_GROUP = 1
    NUM_AGENT_DATA_FEATURE_GROUP = 5

    def __init__(self, observation_radius: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.observation_radius = observation_radius
        self._len_data = self._len(self.max_depth, self.NUM_DATA_FEATURE_GROUP)
        self._len_distance = self._len(self.max_depth, self.NUM_DISTANCE_FEATURE_GROUP)
        self._len_agent_data = self._len(self.max_depth, self.NUM_AGENT_DATA_FEATURE_GROUP)

    @classmethod
    def _len(cls, tree_depth: int, num_features: int) -> int:
        k = num_features
        for _ in range(tree_depth):
            k = k * cls.NUM_BRANCHES + num_features
        return k

    @staticmethod
    def _split_node_into_feature_groups(node: Node):
        data = np.zeros(6)
        distance = np.zeros(1)
        agent_data = np.zeros(5)

        data[0] = node.dist_own_target_encountered
        data[1] = node.dist_other_target_encountered
        data[2] = node.dist_other_agent_encountered
        data[3] = node.dist_potential_conflict
        data[4] = node.dist_unusable_switch
        data[5] = node.dist_to_next_branch

        distance[0] = node.dist_min_to_target

        agent_data[0] = node.num_agents_same_direction
        agent_data[1] = node.num_agents_opposite_direction
        agent_data[2] = node.num_agents_malfunctioning
        agent_data[3] = node.speed_min_fractional
        agent_data[4] = node.num_agents_ready_to_depart

        return data, distance, agent_data

    def _split_subtree_into_feature_groups(self, node, current_tree_depth: int, max_tree_depth: int):
        if node == -np.inf:
            remaining_depth = max_tree_depth - current_tree_depth
            num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
            return (
                [-np.inf] * num_remaining_nodes * self.NUM_DATA_FEATURE_GROUP,
                [-np.inf] * num_remaining_nodes * self.NUM_DISTANCE_FEATURE_GROUP,
                [-np.inf] * num_remaining_nodes * self.NUM_AGENT_DATA_FEATURE_GROUP,
            )

        data, distance, agent_data = self._split_node_into_feature_groups(node)

        if not node.childs:
            return data, distance, agent_data

        for direction in self.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(
                node.childs[direction], current_tree_depth + 1, max_tree_depth
            )
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))

        return data, distance, agent_data

    def split_tree_into_feature_groups(self, tree, max_tree_depth: int):
        data, distance, agent_data = self._split_node_into_feature_groups(tree)
        for direction in self.tree_explored_actions_char:
            sub_data, sub_distance, sub_agent_data = self._split_subtree_into_feature_groups(
                tree.childs[direction], 1, max_tree_depth
            )
            data = np.concatenate((data, sub_data))
            distance = np.concatenate((distance, sub_distance))
            agent_data = np.concatenate((agent_data, sub_agent_data))
        return data, distance, agent_data

    @staticmethod
    def _max_lt(seq, val):
        _max = 0
        idx = len(seq) - 1
        while idx >= 0:
            if seq[idx] < val and seq[idx] >= 0 and seq[idx] > _max:
                _max = seq[idx]
            idx -= 1
        return _max

    @staticmethod
    def _min_gt(seq, val):
        _min = np.inf
        idx = len(seq) - 1
        while idx >= 0:
            if seq[idx] >= val and seq[idx] < _min:
                _min = seq[idx]
            idx -= 1
        return _min

    def _norm_obs_clip(self, obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
        if fixed_radius > 0:
            max_obs = fixed_radius
        else:
            max_obs = max(1, self._max_lt(obs, 1000)) + 1

        min_obs = 0
        if normalize_to_range:
            min_obs = self._min_gt(obs, 0)
        if min_obs > max_obs:
            min_obs = max_obs
        if max_obs == min_obs:
            return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
        norm = np.abs(max_obs - min_obs)
        return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        data = obs[: self._len_data]
        distance = obs[self._len_data : self._len_data + self._len_distance]
        agent_data = obs[-self._len_agent_data :]

        data = self._norm_obs_clip(data, fixed_radius=self.observation_radius)
        distance = self._norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        return np.concatenate((np.concatenate((data, distance)), agent_data))

    def get(self, handle: Optional[AgentHandle] = 0) -> np.ndarray:
        observation = super().get(handle)
        data, distance, agent_data = self.split_tree_into_feature_groups(observation, self.max_depth)
        flattened = np.concatenate((np.concatenate((data, distance)), agent_data))
        return self.normalize_obs(flattened)

    def get_len_flattened(self) -> int:
        k = self.NUM_FEATURES
        for _ in range(self.max_depth):
            k = k * self.NUM_BRANCHES + self.NUM_FEATURES
        return k

    def get_observation_space(self, handle: int = 0) -> gym.Space:
        k = self.get_len_flattened()
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(k,), dtype=np.float64)
