"""Train A2C or DQN on the compact scenario, per-agent env.

Why this exists: `train_ppo.py` builds a JOINT action space (MultiDiscrete over
all agents via concat_vec_envs), which is why its PPO reaches -0.064. DQN does not
support MultiDiscrete at all, so A2C and DQN run on the other wiring in the
project - `pettingzoo_env_to_vec_env_v1` WITHOUT concat, i.e. one env slot per
agent with Discrete(5). The scenario, disruption regime, reward, observation and
evaluator are identical to the PPO run, so each algorithm is comparable to the
forward-only baseline on the same seeds.

    python scripts/train_algo.py --algo a2c --timesteps 2000000 --out runs/a2c_compact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def prepare_nb(radius: int = 50, priority: bool = True, scenario2: bool = False):
    import nb_harness
    import priority_obs

    nb = nb_harness.load(scenario2=scenario2)
    if priority:
        priority_obs.install_priority_features(nb)
    cls = nb["GymTreeObsForRailEnv"]
    original = cls.__init__

    def patched(self, *a, **kw):
        kw.setdefault("observation_radius", radius)
        original(self, *a, **kw)

    cls.__init__ = patched
    return nb


def build_per_agent_env(nb, scenario: str, incidents: int, scale: float = 100.0):
    """One env slot per agent -> action_space Discrete(5), which DQN supports."""
    import supersuit as ss

    import regimes
    import train_ppo as T

    parallel = nb["make_parallel_env"](scenario=scenario, condition="normal")
    regimes.install(parallel._wrap, scenario, incidents)
    T.install_dense_reward(parallel, shaped=True, arrival_bonus=50.0, stuck_penalty=1.0)
    T.install_random_episodes(parallel)
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

    vec = ss.pettingzoo_env_to_vec_env_v1(parallel)
    # SB3VecEnvWrapper unpacks supersuit's (obs, info) reset and hides infos,
    # which is what the A2C notebook does too
    vec = SB3VecEnvWrapper(vec)
    vec.seed = lambda seed=None: [None] * vec.num_envs
    return T.make_scale_wrapper(scale)(vec)


def evaluate(nb, model, scenario: str, incidents: int, seeds) -> dict:
    import train_ppo as T

    ev = T.MetricEvaluator(nb, seeds=seeds, disruption="few_long", incidents=incidents,
                           scenario=scenario)
    res = ev.score(lambda batch: model.predict(batch, deterministic=True)[0])
    return {k: res[k] for k in ("score", "completion", "total_delay", "delay_all")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["a2c", "dqn"])
    ap.add_argument("--scenario", default="network-simple-compact")
    ap.add_argument("--incidents", type=int, default=3)
    ap.add_argument("--timesteps", type=int, default=1_497_600,
                    help="agent-steps; matches the reported A2C/DQN budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--n-steps", type=int, default=5,
                    help="A2C rollout length per env slot (SB3 default 5)")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--eval-seeds", default="9001,9002")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)

    scenario2 = args.scenario == "granville-harris-park"
    nb = prepare_nb(scenario2=scenario2)
    vec = build_per_agent_env(nb, args.scenario, args.incidents)
    print(f"vec envs {vec.num_envs}, obs {vec.observation_space.shape}, "
          f"actions {vec.action_space}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]

    if args.algo == "a2c":
        from stable_baselines3 import A2C

        model = A2C("MlpPolicy", vec, learning_rate=args.lr, gamma=args.gamma,
                    n_steps=args.n_steps, ent_coef=args.ent_coef,
                    vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=dict(net_arch=[args.hidden, args.hidden]),
                    verbose=0, device="cpu", seed=args.seed)
    else:
        from stable_baselines3 import DQN

        model = DQN("MlpPolicy", vec, learning_rate=1e-4, gamma=args.gamma,
                    buffer_size=200_000, learning_starts=10_000, batch_size=64,
                    train_freq=4, gradient_steps=1, target_update_interval=1000,
                    exploration_fraction=0.2, exploration_final_eps=0.02,
                    policy_kwargs=dict(net_arch=[args.hidden, args.hidden]),
                    verbose=0, device="cpu", seed=args.seed)

    from stable_baselines3.common.callbacks import BaseCallback

    rows = []
    state = {"next": args.eval_every, "best": -np.inf}

    class Cb(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps < state["next"]:
                return
            state["next"] += args.eval_every
            res = evaluate(nb, self.model, args.scenario, args.incidents, eval_seeds)
            rows.append({"timesteps": self.num_timesteps, **res})
            star = ""
            if res["score"] > state["best"]:
                state["best"] = res["score"]
                self.model.save(out / "best_model")
                star = " *best"
            print(f"[eval {self.num_timesteps}] score={res['score']:+.4f} "
                  f"compl={res['completion']:.3f} delay_all={res['delay_all']:.0f}{star}",
                  flush=True)

    model.learn(total_timesteps=args.timesteps, callback=Cb(), log_interval=50)
    model.save(out / "final_model")
    (out / "done.json").write_text(json.dumps({"rows": rows,
                                               "best": state["best"]}, indent=1))
    print(f"done: best {state['best']:+.4f} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
