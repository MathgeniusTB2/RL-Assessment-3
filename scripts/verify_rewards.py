"""Identity test: the dense training reward must equal the metric, per term.

`PotentialRewards` is only worth using if, on a real rollout,

    sum_t r_t  ==  the score `DefaultRewards` would have paid

for every agent and every penalty term -- including the cases greedy never
produces (cancellation, overdue-and-unserved stops, agents still on the map at
the horizon).  This script runs exactly those trajectories and compares.

Usage:  python scripts/verify_rewards.py [--scenario ...] [--max-steps N]
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nb_harness  # noqa: E402
from metric_rewards import PotentialRewards  # noqa: E402
from flatland.envs.rewards import BaseDefaultRewards  # noqa: E402
from flatland.envs.step_utils.states import TrainState  # noqa: E402
from flatland.envs.rail_env_action import RailEnvActions as A  # noqa: E402

# collision_factor is 0 in these scenarios, so the transition data is unused,
# but the metric still touches these two fields inside the MOVING->STOPPED path.
DUMMY_TRANSITION = types.SimpleNamespace(
    speed=0,
    state_transition_signal=types.SimpleNamespace(
        stop_action_given=False, new_speed_zero=False, movement_allowed=True
    ),
)


def make_env(nb, seed):
    """Prebuilt env with the metric potential installed as its reward object."""
    env = nb["make_env"](scenario=nb["SCENARIO"], condition="delayed", seed=seed)
    metric = env.rewards  # the DefaultRewards instance the builder installed
    potential = PotentialRewards(
        cancellation_factor=metric.cancellation_factor,
        cancellation_time_buffer=metric.cancellation_time_buffer,
        target_not_reached_minimum_penalty=metric.target_not_reached_minimum_penalty,
        intermediate_not_served_penalty=metric.intermediate_not_served_penalty,
        intermediate_late_arrival_penalty_factor=metric.intermediate_late_arrival_penalty_factor,
        intermediate_early_departure_penalty_factor=metric.intermediate_early_departure_penalty_factor,
        collision_factor=metric.collision_factor,
    )
    env.rewards = potential
    original = env.reset

    def reset(*args, **kwargs):
        out = original(*args, **kwargs)
        potential.begin_episode(env)
        return out

    env.reset = reset
    return env, potential


def policy_actions(nb, kind, env, rng):
    handles = env.get_agent_handles()
    if kind == "greedy":
        return {h: int(v) for h, v in nb["greedy_actions"](env).items()}
    if kind == "never_depart":
        return {h: int(A.STOP_MOVING.value) for h in handles}
    if kind == "random":
        return {h: int(rng.integers(0, 5)) for h in handles}
    if kind == "mixed":  # some move, some sit: unserved stops, cancellations, stragglers
        return {
            h: int(A.MOVE_FORWARD.value) if rng.random() < 0.6 else int(A.STOP_MOVING.value)
            for h in handles
        }
    raise ValueError(kind)


def rollout(nb, kind, seed, max_steps=None):
    env, potential = make_env(nb, seed)
    ref = BaseDefaultRewards(collision_factor=0.0, intermediate_not_served_penalty=1.0)
    handles = env.get_agent_handles()
    env.reset(random_seed=seed)
    T = int(max_steps or env._max_episode_steps)
    rng = np.random.default_rng(seed)

    emitted = {h: 0.0 for h in handles}      # sum of what the trainer would see
    reference = {h: 0.0 for h in handles}    # the metric, driven on the same states
    ref_terms: dict[str, float] = defaultdict(float)

    step = 0
    for step in range(1, T + 1):
        actions = policy_actions(nb, kind, env, rng)
        _, rewards, dones, _ = env.step(actions)
        for h in handles:
            emitted[h] += float(rewards[h])
        for h in handles:
            terms = ref.step_reward(env.agents[h], DUMMY_TRANSITION, env.distance_map, step)
            for k, v in terms.items():
                ref_terms[k] += v
            reference[h] += sum(terms.values())
        if dones.get("__all__"):
            break
    for h in handles:
        terms = ref.end_of_episode_reward(env.agents[h], env.distance_map, step)
        for k, v in terms.items():
            ref_terms[k] += v
        reference[h] += sum(terms.values())

    return {
        "steps": step,
        "emitted": emitted,
        "reference": reference,
        "potential_terms": potential.term_table(),
        "reference_terms": dict(ref_terms),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    nb = nb_harness.load()
    if args.scenario:
        nb["SCENARIO"] = args.scenario
    print(f"scenario={nb['SCENARIO']}  condition=delayed")

    header = f"{'policy':<14}{'seed':>6}{'steps':>6}{'agents':>7}{'sum P':>11}{'sum R':>11}{'max|d|':>9}{'ok':>6}"
    print(header)
    ok = True
    term_rows = []
    for kind in ("greedy", "random", "mixed", "never_depart"):
        for seed in (2000, 2001):
            t0 = time.time()
            out = rollout(nb, kind, seed, args.max_steps)
            h = list(out["emitted"])
            dp = np.array([out["emitted"][x] for x in h])
            dr = np.array([out["reference"][x] for x in h])
            delta = float(np.abs(dp - dr).max())
            same = delta < 1e-9
            ok &= same
            print(
                f"{kind:<14}{seed:>6}{out['steps']:>6}{len(h):>7}"
                f"{dp.sum():>11.1f}{dr.sum():>11.1f}{delta:>9.1e}{str(same):>6}"
                f"   ({time.time() - t0:.0f}s)"
            )
            term_rows.append((kind, seed, out["potential_terms"], out["reference_terms"]))
            if not same:
                worst = np.argsort(-np.abs(dp - dr))[:4]
                for i in worst:
                    print(f"    agent {h[i]}: potential={dp[i]:.3f} metric={dr[i]:.3f}")

    print("\nper-term comparison (sum over agents):")
    keys = sorted({k for _, _, p, _ in term_rows for k in p} |
                  {k for _, _, _, r in term_rows for k in r})
    for k in keys:
        diffs = [abs(p.get(k, 0.0) - r.get(k, 0.0)) for _, _, p, r in term_rows]
        print(f"  {k:<32} max |diff| = {max(diffs):.3e}")

    print("\nIDENTITY", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
