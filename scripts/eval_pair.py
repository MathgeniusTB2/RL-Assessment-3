"""Paired evaluation: greedy baseline vs a trained PPO checkpoint.

Both policies run the *same* episode seeds on the *same* env, so the comparison
is paired per episode rather than two independent averages.  Scoring always
uses the env's own `DefaultRewards` (never the training reward).

Usage
  python scripts/eval_pair.py --model runs/s1_dense_a/best_model.zip \
      --seeds 20 --conditions delayed,normal --out results/pair_s1_a.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import secrets
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import priority_obs  # noqa: E402
import nb_harness  # noqa: E402

from flatland.envs.step_utils.states import TrainState  # noqa: E402


class Episode:
    """One env, reused across policies and seeds (deadline calibration is expensive)."""

    def __init__(self, nb, scenario, condition, disruption="poisson", incidents=1):
        self.nb = nb
        self.disruption = disruption
        self.incidents = incidents
        # `condition` is what the caller asked for. The deterministic regime is
        # layered on a NORMAL env, so that path still overrides it - but the
        # poisson path must honour the request, otherwise `--conditions
        # delayed,normal` silently runs the same condition twice.
        base_condition = condition if disruption == "poisson" else "normal"
        self.condition = base_condition
        self.parallel = nb["make_parallel_env"](scenario=scenario, condition=base_condition)
        if disruption == "few_long":
            import regimes

            regimes.install(self.parallel._wrap, scenario, incidents)
        self.raw = self.parallel._wrap
        self.handles = list(self.parallel.possible_agents)
        self.T = int(self.raw._max_episode_steps)

    def run(self, seed, policy_fn) -> dict:
        raw, T = self.raw, self.T
        obs, _ = self.parallel.reset(seed=seed)
        cum = {h: 0.0 for h in self.handles}
        reached, done_step = {h: False for h in self.handles}, {}
        inter_cells = {
            h: {
                tuple(int(x) for x in wp.position)
                for grp in raw.agents[h].waypoints[1:-1]
                for wp in grp
            }
            for h in self.handles
        }
        served = {h: set() for h in self.handles}
        step = 0
        for step in range(1, T + 1):
            actions = policy_fn(raw, obs, self.handles)
            obs, rewards, terms, _, infos = self.parallel.step(actions)
            for h, r in rewards.items():
                if r is not None:
                    cum[h] += float(r)
            for h in self.handles:
                if infos[h].get("state") == TrainState.DONE:
                    reached[h] = True
                    done_step.setdefault(h, step)
                if infos[h].get("state") == TrainState.STOPPED:
                    cfg = raw.agents[h].current_configuration
                    if cfg is not None and cfg[0] is not None:
                        pos = tuple(int(x) for x in cfg[0])
                        if pos in inter_cells[h]:
                            served[h].add(pos)
            if all(v for k, v in terms.items() if k != "__all__"):
                break

        rewards_arr = np.array([cum[h] for h in self.handles])
        score = float(
            raw.rewards.normalize(
                *rewards_arr, num_agents=len(self.handles), max_episode_steps=T
            )
        ) - 1.0
        scheduled = [raw.agents[h].waypoints_latest_arrival[-1] for h in self.handles]
        delays = [
            done_step[h] - scheduled[h]
            for h in self.handles
            if h in done_step and scheduled[h] is not None
        ]
        delay_all = float(np.sum([
            (done_step[h] if h in done_step else T) - scheduled[h]
            for h in self.handles
            if scheduled[h] is not None
        ]))
        total_inter = sum(len(inter_cells[h]) for h in self.handles)
        return {
            "score": score,
            "delay_all": delay_all,
            "completion": sum(reached.values()) / len(self.handles),
            "mean_return": float(np.mean(rewards_arr)),
            "total_delay": float(np.sum(delays)) if delays else float("nan"),
            "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "on_time": float(np.mean([d <= 0 for d in delays])) if delays else float("nan"),
            "stops_served": (
                sum(len(served[h]) for h in self.handles) / total_inter
                if total_inter
                else float("nan")
            ),
            "episode_steps": step,
        }


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact binomial p-value for wins vs losses (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to a saved SB3 model")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"],
                    help="which SB3 class saved --model (they are not interchangeable)")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--conditions", default="delayed")
    ap.add_argument("--seeds", default="8", help="integer count (fresh entropy seeds) or comma list")
    ap.add_argument("--out", default="results/pair.csv")
    ap.add_argument("--disruption", default="poisson", choices=["poisson", "few_long"])
    ap.add_argument("--incidents", type=int, default=1)
    ap.add_argument("--baseline", default="notebook",
                    choices=["forward", "notebook"],
                    help="which greedy baseline to compare against")
    ap.add_argument("--priority", action="store_true")
    ap.add_argument("--observation-radius", type=int, default=None)
    args = ap.parse_args()

    scenario2 = args.scenario == "granville-harris-park"
    nb = nb_harness.load(scenario2=scenario2)
    if args.priority:
        priority_obs.install_priority_features(nb)
    if args.observation_radius is not None:
        cls = nb["GymTreeObsForRailEnv"]
        original_init = cls.__init__

        def patched(self, *a, **kw):
            kw.setdefault("observation_radius", args.observation_radius)
            original_init(self, *a, **kw)

        cls.__init__ = patched
    if args.scenario:
        nb["SCENARIO"] = args.scenario
    scenario = nb["SCENARIO"]

    if "," in args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        n = int(args.seeds)
        seeds = [secrets.randbits(31) for _ in range(n)]
    print(f"scenario={scenario} seeds={seeds}")

    if args.baseline == "forward":
        from flatland.envs.rail_env_action import RailEnvActions

        def greedy(raw, obs, handles):
            return {h: int(RailEnvActions.MOVE_FORWARD.value) for h in handles}
    else:
        def greedy(raw, obs, handles):
            plan = nb["greedy_actions"](raw)
            return {h: int(plan.get(h, 0)) for h in handles}

    model = None
    if args.model:
        from stable_baselines3 import A2C, DQN, PPO

        loader = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo]
        model = loader.load(args.model, device="cpu")

        def ppo(raw, obs, handles):
            batch = np.stack([obs[h] for h in handles])
            chosen = model.predict(batch, deterministic=True)[0]
            return {h: int(a) for h, a in zip(handles, chosen, strict=True)}
    else:
        ppo = None

    rows = []
    for condition in args.conditions.split(","):
        ep = Episode(nb, scenario, condition, args.disruption, args.incidents)
        for seed in seeds:
            g = ep.run(seed, greedy)
            rows.append({"condition": condition, "policy": "greedy", "seed": seed, **g})
            if ppo is not None:
                p = ep.run(seed, ppo)
                rows.append({"condition": condition, "policy": args.algo, "seed": seed, **p})
                print(
                    f"  {condition} seed={seed:<12} greedy={g['score']:+.4f} "
                    f"{args.algo}={p['score']:+.4f} "
                    f"delta={p['score'] - g['score']:+.4f} "
                    f"(compl {g['completion']:.3f} -> {p['completion']:.3f})",
                    flush=True,
                )
            else:
                print(f"  {condition} seed={seed:<12} greedy={g['score']:+.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print("\nsummary")
    for condition in args.conditions.split(","):
        sub = [r for r in rows if r["condition"] == condition]
        g = [r for r in sub if r["policy"] == "greedy"]
        print(
            f"  {condition:<8} greedy score={np.mean([r['score'] for r in g]):+.4f} "
            f"compl={np.mean([r['completion'] for r in g]):.3f} "
            f"delay={np.mean([r['total_delay'] for r in g]):.0f}"
        )
        if ppo is None:
            continue
        p = [r for r in sub if r["policy"] == args.algo]
        deltas = np.array([b["score"] - a["score"] for a, b in zip(g, p, strict=True)])
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        print(
            f"  {condition:<8} ppo    score={np.mean([r['score'] for r in p]):+.4f} "
            f"compl={np.mean([r['completion'] for r in p]):.3f} "
            f"delay={np.mean([r['total_delay'] for r in p]):.0f}"
        )
        print(
            f"  {condition:<8} paired delta={deltas.mean():+.4f} "
            f"median={np.median(deltas):+.4f} wins={wins}/{len(deltas)} "
            f"sign-test p={sign_test_p(wins, losses):.4f}"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
