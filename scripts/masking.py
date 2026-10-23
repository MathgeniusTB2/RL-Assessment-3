"""Action masking for the Flatland vec env (competition-proven sample efficiency).

In the 2020 Flatland competition every strong entry masked invalid actions: an
agent on a straight cell has exactly one legal movement, so sampling among five
actions wastes most of the rollout and teaches nothing about the two that matter.
Masking removes that noise without changing the dynamics.

The chain built by `train_ppo.build_train_env` is

    ScaleReward -> SB3VecEnvWrapper -> ConcatVecEnv -> [MarkovVectorEnv] x copies
                                                          -> PettingZoo parallel env
                                                             -> RailEnv  (agents here)

so masks are computed from the RailEnv of each copy and concatenated in the same
order as the observations.
"""

from __future__ import annotations

import numpy as np

N_ACTIONS = 5
DO_NOTHING, LEFT, FORWARD, RIGHT, STOP = 0, 1, 2, 3, 4


def _raw_envs(vec):
    """Reach the per-copy RailEnv objects through the supersuit stack."""
    node = vec
    for _ in range(6):
        if hasattr(node, "vec_envs"):
            raws = []
            for sub in node.vec_envs:
                raw = None
                for attribute in ("par_env", "vec_env", "env"):
                    candidate = getattr(sub, attribute, None)
                    if candidate is not None:
                        raw = getattr(candidate, "_wrap", candidate)
                        break
                raws.append(raw)
            return raws
        for attribute in ("venv", "vec_env", "env"):
            if hasattr(node, attribute):
                node = getattr(node, attribute)
                break
        else:
            break
    raise RuntimeError("could not find the underlying RailEnv in the vec env chain")


def agent_mask(raw) -> np.ndarray:
    """One mask row per agent: legal movements given the current configuration."""
    from flatland.envs.step_utils.states import TrainState

    masks = np.zeros((len(raw.agents), N_ACTIONS), dtype=np.int8)
    masks[:, DO_NOTHING] = 1
    masks[:, STOP] = 1
    for handle, agent in enumerate(raw.agents):
        configuration = agent.current_configuration
        if configuration is None or agent.state.is_off_map_state():
            masks[handle, FORWARD] = 1  # a movement action departs
            continue
        position = (int(configuration[0][0]), int(configuration[0][1]))
        heading = int(configuration[1])
        for _, next_direction in raw.rail.get_successor_configurations((position, heading)):
            delta = (int(next_direction) - heading) % 4
            if delta == 0:
                masks[handle, FORWARD] = 1
            elif delta == 1:
                masks[handle, RIGHT] = 1
            elif delta == 3:
                masks[handle, LEFT] = 1
    return masks


def compute_masks(vec) -> np.ndarray:
    raws = _raw_envs(vec)
    return np.concatenate([agent_mask(raw) for raw in raws], axis=0)


def make_masked(scale_wrapper):
    """Wrap a VecEnv so `action_masks()` reflects the current state."""
    from stable_baselines3.common.vec_env import VecEnvWrapper

    class MaskedVecEnv(VecEnvWrapper):
        def __init__(self, venv):
            super().__init__(venv)
            self._masks = compute_masks(venv)

        def reset(self, **kwargs):
            observation = self.venv.reset(**kwargs)
            self._masks = compute_masks(self.venv)
            return observation

        def step_wait(self):
            observations, rewards, dones, infos = self.venv.step_wait()
            self._masks = compute_masks(self.venv)
            return observations, rewards, dones, infos

        def action_masks(self):
            return self._masks

        def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
            # same reason: supersuit's ConcatVecEnv has no env_method, so serve
            # the mask query locally
            if method_name == "action_masks":
                return self._masks
            return self.venv.env_method(
                method_name, *method_args, indices=indices, **method_kwargs
            )

        def get_attr(self, attr_name, indices=None):
            if attr_name == "action_masks":
                return [self._masks] * self.num_envs
            return self.venv.get_attr(attr_name, indices)

        def has_attr(self, attr_name):
            # BaseVecEnv.has_attr() recurses through .venv and supersuit's
            # ConcatVecEnv does not implement it; answer locally instead
            if attr_name == "action_masks":
                return True
            try:
                object.__getattribute__(self, attr_name)
                return True
            except AttributeError:
                return False

    return MaskedVecEnv(scale_wrapper)
