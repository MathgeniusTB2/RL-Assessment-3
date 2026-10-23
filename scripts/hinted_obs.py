"""Observation upgrade: tree + the heuristic's recommended action ("advice").

Measured problem: a 99.0 %-accurate imitation of the distance-map greedy still
completes only 40/74 agents, because the 1 % of wrong switch decisions compound
into jams.  From scratch, PPO never leaves max entropy (2M steps, score still
-0.21, completion 0).  So the routing decision itself is the bottleneck, not the
disruption behaviour we want to learn.

Fix: append a 5-dim one-hot of the greedy controller's action for this agent to
the tree observation.  The policy then starts at the baseline's competence, and
the experiment is exactly the question we care about -- *when should the agent
override the heuristic?* -- which is where a distance-map controller is
structurally blind (no yielding, no rerouting, no notion of a blocked
single-track section).

The hint is computed once per environment step for every agent (the notebook's
own `greedy_actions`), so it stays a property of the observation, not a
hard-coded policy.
"""

from __future__ import annotations

import numpy as np

N_ACTIONS = 5


def install_hint_builder(nb) -> None:
    """Patch the namespace's observation class to append the advice one-hot."""
    import gymnasium as gym

    base_cls = nb["GymTreeObsForRailEnv"]
    greedy_fn = nb["greedy_actions"]

    class HintedTreeObs(base_cls):
        """Tree observation + one-hot of the greedy controller's action."""

        greedy = staticmethod(greedy_fn)
        n_actions = N_ACTIONS

        def reset(self, env):
            super().reset(env)
            self._hint = {}

        def get_many(self, handles=None):
            # one call for the whole fleet, not one per agent
            self._hint = self.greedy(self.env) or {}
            return super().get_many(handles)

        def get(self, handle: int = 0):
            tree = np.asarray(super().get(handle), dtype=np.float64)
            onehot = np.zeros(self.n_actions, dtype=np.float64)
            onehot[int(self._hint.get(handle, 0))] = 1.0
            return np.concatenate((tree, onehot))

        def get_observation_space(self, handle: int = 0) -> gym.Space:
            base = super().get_observation_space(handle)
            return gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(int(np.prod(base.shape)) + self.n_actions,),
                dtype=np.float64,
            )

    nb["GymTreeObsForRailEnv"] = HintedTreeObs
