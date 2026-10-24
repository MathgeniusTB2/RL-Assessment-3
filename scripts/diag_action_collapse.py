"""Is a trained policy degenerate? Print its action distribution over an episode.

A correct-but-unlearned policy and a broken implementation look the same on the
score line, so this separates them: a healthy policy uses several actions across
observations, a collapsed one emits the same action for every agent at every step.

Also scores forward-only and a random policy on the same seed, which proves the
evaluator actually discriminates between policies.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--a2c", default=None, help="optional A2C checkpoint to audit")
    ap.add_argument("--dqn", default=None, help="optional DQN checkpoint to audit")
    ap.add_argument("--ppo", default="runs/xb_b/best_model.zip",
                    help="the shipped corridor PPO checkpoint")
    ap.add_argument("--scenario", default="network-simple-compact")
    ap.add_argument("--incidents", type=int, default=2)
    ap.add_argument("--variant", default="block",
                    help="regime variant: block (reported), crossing, band or none")
    ap.add_argument("--seed", type=int, default=9001)
    args = ap.parse_args()

    from flatland.envs.step_utils.states import TrainState
    from stable_baselines3 import A2C, DQN, PPO

    import nb_harness
    import priority_obs
    import regimes

    nb = nb_harness.load()
    priority_obs.install_priority_features(nb)
    cls = nb["GymTreeObsForRailEnv"]
    original = cls.__init__

    def patched(self, *a, **kw):
        kw.setdefault("observation_radius", 50)
        original(self, *a, **kw)

    cls.__init__ = patched

    def episode(policy_fn, seed=None):
        seed = args.seed if seed is None else seed
        p = nb["make_parallel_env"](scenario=args.scenario,
                                    condition="normal", seed=seed)
        raw = p._wrap
        if args.variant not in (None, "", "none"):
            regimes.install(raw, args.scenario, args.incidents, variant=args.variant)
        obs, _ = p.reset(seed=seed)
        handles = list(p.possible_agents)
        hist = collections.Counter()
        T = int(raw._max_episode_steps)
        cum = {h: 0.0 for h in handles}
        for _ in range(1, T + 1):
            acts = policy_fn(obs, handles)
            for a in acts.values():
                hist[a] += 1
            obs, rew, _, _, _ = p.step(acts)
            for h, r in rew.items():
                if r is not None:
                    cum[h] += float(r)
            if all(raw.agents[h].state == TrainState.DONE for h in handles):
                break
        score = float(raw.rewards.normalize(*[cum[h] for h in handles],
                                            num_agents=len(handles),
                                            max_episode_steps=T)) - 1.0
        return score, hist

    def from_model(model):
        def fn(obs, handles):
            batch = np.stack([obs[h] for h in handles])
            chosen = model.predict(batch, deterministic=True)[0]
            return {h: int(a) for h, a in zip(handles, chosen)}
        return fn

    rng = np.random.default_rng(0)
    policies = {
        "forward-only": lambda o, hs: {h: 2 for h in hs},
        "random": lambda o, hs: {h: int(rng.integers(0, 5)) for h in hs},
    }
    for name, path in ((n, p) for n, p in (("a2c", args.a2c), ("dqn", args.dqn),
                                           ("ppo", args.ppo)) if p):
        p = Path(path)
        if not p.exists():
            print(f"{name.upper():<13} skipped: {path} not found")
            continue
        loader = {"a2c": A2C, "dqn": DQN, "ppo": PPO}[name].load
        policies[name.upper()] = from_model(loader(str(p), device="cpu"))

    for name, fn in policies.items():
        score, hist = episode(fn)
        total = sum(hist.values())
        top = ", ".join(f"{a}:{100 * v / total:.0f}%" for a, v in sorted(hist.items()))
        print(f"{name:<13} score {score:+.4f}  actions used: {len(hist)}/5  [{top}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
