"""Scenario 2 (Granville-Harris Park) as an RL env matching scenario 1's interface.

The S2 notebook is a scripted-rollout study: it loads the .mpk with
`GlobalObsForRailEnv` (not gym-spaced), builds agents at runtime from the
timetable, and drives a hand-written route follower.  This module wraps the same
scenario into the interface `train_ppo.py` / `eval_pair.py` expect:

    nb["make_env"](scenario=..., condition=..., seed=...)
    nb["make_parallel_env"](...)
    nb["greedy_actions"](env)

The observation builder is swapped to the same `GymTreeObsForRailEnv` used on
scenario 1, so the two scenarios share one policy interface; the space is small
enough (20x20) that the tree already covers the whole junction.

The "greedy" baseline is the notebook's scripted route follower (stop once at
each station, take the recorded turn at junction cells), which is what the S2
study is measured against.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]

S2_CELLS = (6, 8, 9, 10, 12, 13, 14, 15, 16, 19, 18, 20)
# 6 imports / 8,9,10 layout+timetable constants / 12 effects / 13 loader /
# 14 shortest running time / 15 create_agents / 16 create_timetable /
# 19 scripted rollout / 18 setup_environment (needs 19 for reference_arrivals)
# 20 print_results


def _load_notebook(namespace: Dict[str, Any]) -> Dict[str, Any]:
    import sys
    import types

    os.chdir(ROOT)
    path = ROOT / "notebooks" / "Scenario2" / "Scenario2.ipynb"
    cells = json.loads(path.read_text())["cells"]
    # exec into a real, registered module: supersuit cloudpickles the env when it
    # copies it, and dynamic classes defined in an ad-hoc namespace fail the
    # "lookup by module + qualname" step otherwise
    module = types.ModuleType("a3_scenario2_ns")
    module.__file__ = str(path)
    module.__dict__.update(namespace)
    sys.modules[module.__name__] = module
    for i in S2_CELLS:
        exec(compile("".join(cells[i]["source"]), f"{path.name}[cell {i}]", "exec"), module.__dict__)
    # the module dict IS the namespace, so later patches (obs builder, greedy)
    # are what the notebook's own functions resolve at call time
    return module.__dict__


def load(s1_nb: Dict[str, Any]) -> Dict[str, Any]:
    """Return an S1-compatible namespace backed by the scenario-2 notebook."""
    ns: Dict[str, Any] = _load_notebook(
        {"__name__": "s2_env", "__file__": str(Path(__file__))}
    )

    from flatland.envs.observations import TreeObsForRailEnv  # noqa: F401
    from flatland.envs.predictions import ShortestPathPredictorForRailEnv

    s1_nb["silence_tree_walk_warnings"]()
    GymTree = s1_nb["GymTreeObsForRailEnv"]
    depth = int(s1_nb.get("MAX_DEPTH", 3))
    predictor_depth = int(s1_nb.get("PREDICTOR_DEPTH", 50))

    # the loader references these module-level names at call time; keep the class
    # itself in the namespace so observation patches (e.g. the advice one-hot)
    # apply to scenario 2 as well
    ns["GymTreeObsForRailEnv"] = GymTree

    def _obs_builder(*args, **kwargs):
        return ns["GymTreeObsForRailEnv"](
            max_depth=depth,
            predictor=ShortestPathPredictorForRailEnv(max_depth=predictor_depth),
        )

    ns["GlobalObsForRailEnv"] = _obs_builder

    train_config = ns["REALISTIC_TIMETABLE_CONFIG"]
    setup_environment = ns["setup_environment"]

    def _fresh_env(condition: str, seed: int | None = None):
        env, agents = setup_environment(train_config, condition=condition, verbose=False)
        # `setup_environment` sets the budget from the timetable (last arrival +
        # EXTRA_STEPS), but `RailEnv.reset()` reinstates the value baked into the
        # .mpk (500). With the real timetable some trains arrive after step 500,
        # so the episode ended with 15 trains still running - they were scored as
        # never-arrived no matter what the policy did. Re-apply after every reset.
        budget = int(env._max_episode_steps)
        original_reset = env.reset

        def reset(*a, **kw):
            out = original_reset(*a, **kw)
            env._max_episode_steps = budget
            return out

        env.reset = reset
        env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=seed)
        return env

    def make_env(scenario=None, condition="normal", seed=None, **kwargs):
        return _fresh_env(condition, seed)

    def make_parallel_env(scenario=None, condition="normal", seed=None, force_stop=True, **kwargs):
        import gymnasium as gym
        from flatland.ml.pettingzoo.wrappers import PettingzooFlatland

        env = _fresh_env(condition, seed)
        parallel = PettingzooFlatland(env).parallel_env()
        # the .mpk has no agents, so wrap.number_of_agents is 0 and the wrapper
        # builds an empty action_spaces dict; rebuild it from the runtime agents
        parallel.action_spaces = {
            handle: gym.spaces.Discrete(5) for handle in parallel.possible_agents
        }
        return parallel

    def greedy_actions(env):
        """Scripted route follower: stop once per station, obey recorded turns."""
        from flatland.envs.rail_env_action import RailEnvActions

        completed = greedy_actions.completed
        if (
            completed is None
            or len(completed) != len(env.agents)
            or int(getattr(env, "_elapsed_steps", 0) or 0) <= 1
        ):
            completed = greedy_actions.completed = {a.handle: set() for a in env.agents}
        actions = {}
        for agent in env.agents:
            handle = agent.handle
            train = train_config[handle]
            actions[handle] = RailEnvActions.MOVE_FORWARD.value
            if agent.current_configuration is None:
                continue
            position = agent.current_configuration[0]
            if position in train["stops"] and position not in completed[handle]:
                actions[handle] = RailEnvActions.STOP_MOVING.value
                completed[handle].add(position)
            elif position in train.get("route", {}):
                actions[handle] = int(train["route"][position].value)
        return actions

    greedy_actions.completed = None

    ns.update(
        {
            "make_env": make_env,
            "make_parallel_env": make_parallel_env,
            "greedy_actions": greedy_actions,
            "SCENARIO": "granville-harris-park",
            "CONDITION": "normal",
            "CONDITIONS": ns.get("CONDITIONS", {"normal": None, "delayed": 1 / 540}),
            "MAX_DEPTH": depth,
            "PREDICTOR_DEPTH": predictor_depth,
            "_s2_train_config": train_config,
        }
    )
    return ns
