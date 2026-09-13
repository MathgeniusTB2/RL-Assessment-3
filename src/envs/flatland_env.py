"""Flatland environment factory approximating the paper's ``small_v0`` scenario.

Paper: Mohanty et al., "Flatland-RL: Multi-Agent Reinforcement Learning on
Trains", arXiv:2012.05893. The original generator config
(``envs/flatland/generator_configs/small_v0.yaml``) is:

    width: 25, height: 25, number_of_agents: 5
    max_num_cities: 4, grid_mode: False
    max_rails_between_cities: 2, max_rails_in_city: 3
    regenerate_rail_on_reset: True, regenerate_schedule_on_reset: True

Modern Flatland names ``max_rails_in_city`` as ``max_rail_pairs_in_city``.
"""

from __future__ import annotations

from typing import Any

from flatland.env_generation.env_generator import env_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.ml.pettingzoo.wrappers import PettingzooFlatland

from src.envs.rewards import PaperRewards
from src.envs.tree_obs import GymTreeObsForRailEnv

# Paper's `small_v0` scenario, mapped onto `env_generator` kwargs.
# NOTE: the 2020 generator's `max_rails_in_city: 3` is not directly comparable
# to modern `max_rail_pairs_in_city` (pairs of tracks). `1` pair is the largest
# value that reliably fits 4 cities in a 25x25 grid with the modern generator.
PAPER_SMALL_V0: dict[str, Any] = dict(
    n_agents=5,
    x_dim=25,
    y_dim=25,
    n_cities=4,
    grid_mode=False,
    max_rails_between_cities=2,
    max_rail_pairs_in_city=1,
    # Disruptions (malfunctions): paper used duration 20-50, interval 540.
    malfunction_duration_min=20,
    malfunction_duration_max=50,
    malfunction_interval=540,
)


def make_raw_env(
    seed: int | None = None,
    max_depth: int = 2,
    predictor_depth: int = 30,
    paper_reward: bool = True,
    **overrides: Any,
) -> RailEnv:
    """Build a single Flatland ``RailEnv`` matching the paper's setup.

    Rail and schedule are regenerated on every ``reset`` (Flatland's default),
    matching the paper's ``reset_env_freq: 1``.
    """
    params = {**PAPER_SMALL_V0, **overrides}
    obs_builder = GymTreeObsForRailEnv(
        max_depth=max_depth,
        predictor=ShortestPathPredictorForRailEnv(max_depth=predictor_depth),
    )
    rewards = PaperRewards() if paper_reward else None
    env, _, _ = env_generator(
        obs_builder_object=obs_builder, seed=seed, rewards=rewards, **params
    )
    return env


def make_parallel_env(seed: int | None = None, **overrides: Any):
    """PettingZoo parallel env wrapping the Flatland env (SB3-compatible)."""
    return PettingzooFlatland(make_raw_env(seed=seed, **overrides)).parallel_env()
