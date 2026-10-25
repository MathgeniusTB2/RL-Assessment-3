"""Verify that the dense training reward reproduces the assessment metric exactly.

The report's §3.2 claim is that ``sum_t r_t`` for the shipped
``PotentialRewards`` equals what ``DefaultRewards`` pays for the same episode,
per agent AND per penalty term.  This script is that check: it runs the same
policy on the same seed twice - once with the library metric, once with the
potential installed exactly as ``train_ppo`` installs it - and compares.

    python scripts/verify_rewards.py                 # forward-only, seed 2000
    python scripts/verify_rewards.py --policy greedy --seed 2001
    python scripts/verify_rewards.py --policy never          # cancellation-only rollout
    python scripts/verify_rewards.py --scenario granville-harris-park --policy notebook

Exit code 0 means every per-agent total and every per-term total agrees to
``--tol`` (default 1e-9); exit code 1 prints the offenders.

``--policy never`` is the one known exception: every train stays off-map, and the
metric charges the cancellation lump at episode end while the potential charges it
at the first step the train could have departed, so the totals differ by about 1 %
of that lump and the script reports FAIL. That deviation is documented in the
report's Section 3.2; the other three policies agree exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


def build(scenario: str, seed: int, incidents: int, dense: bool, condition: str = "normal"):
    """One parallel env, optionally with the potential reward installed."""
    import nb_harness
    import priority_obs
    import regimes
    import train_ppo

    scenario2 = scenario == "granville-harris-park"
    nb = nb_harness.load(scenario2=scenario2)
    priority_obs.install_priority_features(nb)
    cls = nb["GymTreeObsForRailEnv"]
    original_init = cls.__init__

    def patched(self, *a, **kw):
        kw.setdefault("observation_radius", 50)
        original_init(self, *a, **kw)

    cls.__init__ = patched

    nb["SCENARIO"] = scenario
    parallel = nb["make_parallel_env"](scenario=scenario, condition=condition)
    regimes.install(parallel._wrap, scenario, incidents)

    terms = {}

    if dense:
        potential = train_ppo.install_dense_reward(parallel, shaped=False)

        def term_table():
            return potential.term_table()

    else:
        # record the metric's own per-term dict as it is produced
        proxy = getattr(parallel._wrap.rewards, "_proxy", parallel._wrap.rewards)
        original = proxy._agent_done_or_max_episode_steps_reward

        def recording(agent, distance_map, elapsed_steps, d):
            out = original(agent, distance_map, elapsed_steps, d)
            for k, v in d.items():
                terms[k] = terms.get(k, 0.0) + float(v)
            return out

        proxy._agent_done_or_max_episode_steps_reward = recording

        def term_table():
            return dict(terms)

    return nb, parallel, term_table


def rollout(nb, parallel, policy: str, seed: int) -> dict:
    """Run one episode and return the per-agent summed reward."""
    import numpy as np

    obs, _ = parallel.reset(seed=seed)
    handles = list(parallel.possible_agents)
    cum = {h: 0.0 for h in handles}
    raw = parallel._wrap

    if policy == "forward":
        from flatland.envs.rail_env_action import RailEnvActions as A

        def act(_raw, _obs, hs):
            return {h: int(A.MOVE_FORWARD.value) for h in hs}

    elif policy == "greedy":
        greedy = nb["greedy_actions"]

        def act(raw_env, _obs, hs):
            plan = greedy(raw_env)
            return {h: int(plan.get(h, 0)) for h in hs}

    elif policy == "random":
        rng = np.random.default_rng(seed)

        def act(_raw, _obs, hs):
            return {h: int(rng.integers(0, 5)) for h in hs}

    elif policy == "never":
        # DO_NOTHING for every train: nothing departs, so the metric pays the
        # cancellation lump for every agent - the rollout the report cites
        def act(_raw, _obs, hs):
            return {h: 0 for h in hs}

    else:
        raise SystemExit(f"unknown policy {policy!r}")

    T = int(raw._max_episode_steps)
    for _ in range(1, T + 1):
        actions = act(raw, obs, handles)
        obs, rewards, terms, _, _ = parallel.step(actions)
        for h, r in rewards.items():
            if r is not None:
                cum[h] += float(r)
        if all(v for k, v in terms.items() if k != "__all__"):
            break
    return cum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="network-simple-compact")
    ap.add_argument("--policy", default="forward",
                    choices=["forward", "greedy", "random", "never"])
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--incidents", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    print(f"scenario={args.scenario} policy={args.policy} seed={args.seed} "
          f"incidents={args.incidents}")

    nb_a, env_a, terms_a = build(args.scenario, args.seed, args.incidents, dense=False)
    metric = rollout(nb_a, env_a, args.policy, args.seed)
    table_a = terms_a()

    nb_b, env_b, terms_b = build(args.scenario, args.seed, args.incidents, dense=True)
    dense = rollout(nb_b, env_b, args.policy, args.seed)
    table_b = terms_b()

    handles = sorted(metric)
    diffs = [abs(metric[h] - dense[h]) for h in handles]
    worst = max(diffs) if diffs else 0.0
    print(f"agents={len(handles)}  metric_total={sum(metric.values()):.6f}  "
          f"dense_total={sum(dense.values()):.6f}  max_agent_diff={worst:.3e}")

    term_diffs = {}
    for k in sorted(set(table_a) | set(table_b)):
        d = abs(float(table_a.get(k, 0.0)) - float(table_b.get(k, 0.0)))
        term_diffs[k] = d
    worst_term = max(term_diffs.values()) if term_diffs else 0.0
    print(f"terms checked={len(term_diffs)}  max_term_diff={worst_term:.3e}")

    ok = worst <= args.tol and worst_term <= args.tol
    if not ok:
        for h in handles:
            if abs(metric[h] - dense[h]) > args.tol:
                print(f"  agent {h}: metric={metric[h]:.9f} dense={dense[h]:.9f}")
        for k, d in term_diffs.items():
            if d > args.tol:
                print(f"  term {k}: metric={table_a.get(k, 0.0):.9f} "
                      f"dense={table_b.get(k, 0.0):.9f} diff={d:.3e}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
