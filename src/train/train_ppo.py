#!/usr/bin/env python3
"""Train an approximate PPO baseline on Flatland (paper: arXiv:2012.05893).

Modern stack: current ``flatland-rl`` + PettingZoo/SuperSuit + Stable-Baselines3
PPO with a shared policy across agents. Logs periodic evaluation
(normalized score, completion rate) to CSV for comparison with the paper.

Usage:
    python -m src.train.train_ppo --seed 0 --total-timesteps 15000000
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from src.envs.flatland_env import make_parallel_env
from src.train.callbacks import FlatlandEvalCallback

# SuperSuit's subprocess workers must use `fork`. On macOS the default `spawn`
# re-imports the entry module and deadlocks; `fork` works and is the Linux
# default. Must be set before SuperSuit spawns any workers.
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass


def build_vec_env(seed: int, n_envs: int, n_cpus: int, max_depth: int, paper_reward: bool = True):
    """PettingZoo parallel env -> SuperSuit VecEnv for SB3.

    ``pettingzoo_env_to_vec_env_v1`` makes one sub-env per agent (shared policy),
    ``concat_vec_envs_v1`` replicates them for parallel data collection.
    """
    env = make_parallel_env(seed=seed, max_depth=max_depth, paper_reward=paper_reward)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, n_envs, num_cpus=n_cpus, base_class="stable_baselines3"
    )
    # SB3 2.9 calls `env.seed()` in set_random_seed, but SuperSuit's in-process
    # ConcatVecEnv (gymnasium VectorEnv) has no `seed`. Seeding is not needed:
    # Flatland regenerates rail/schedule from its own RNG on every reset.
    env.seed = lambda seed=None: [None] * env.num_envs
    return env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=15_000_000)
    p.add_argument("--n-envs", type=int, default=8, help="replicas concatenated by SuperSuit")
    p.add_argument(
        "--n-cpus",
        type=int,
        default=0,
        help="SuperSuit subprocess workers; 0 = in-process, "
        ">0 = multiprocessing (the script forces the `fork` start method)",
    )
    p.add_argument("--eval-freq", type=int, default=100_000)
    p.add_argument("--n-eval-episodes", type=int, default=50)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--net-arch", type=int, nargs=2, default=[256, 256])
    p.add_argument(
        "--norm-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="normalize/clip rewards with VecNormalize (recommended for Flatland)",
    )
    p.add_argument(
        "--paper-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the paper's -1/step, 0-on-arrival reward instead of Flatland defaults",
    )
    p.add_argument("--clip-reward", type=float, default=10.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--tensorboard", action="store_true", help="enable SB3 TensorBoard logging")
    p.add_argument("--out-dir", type=Path, default=Path("results/ppo"))
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_name = args.run_name or f"ppo_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval.csv"

    print(f"[train] run={run_name} out={out_dir}", flush=True)
    print(
        f"[train] env: 25x25, 5 agents, tree depth {args.max_depth}; "
        f"vec replicas={args.n_envs}, cpus={args.n_cpus}",
        flush=True,
    )

    train_env = build_vec_env(
        args.seed, args.n_envs, args.n_cpus, args.max_depth, args.paper_reward
    )
    if args.norm_reward:
        # Flatland returns are large and negatively scaled; normalizing/clipping
        # rewards keeps value targets well-conditioned (paper used clip_rewards).
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=args.clip_reward,
            gamma=args.gamma,
        )
    eval_env = make_parallel_env(
        seed=args.seed + 1000, max_depth=args.max_depth, paper_reward=args.paper_reward
    )
    eval_seeds = [10_000 + i for i in range(args.n_eval_episodes)]

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        policy_kwargs=dict(net_arch=list(args.net_arch)),
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(out_dir / "tb") if args.tensorboard else None,
    )

    callback = FlatlandEvalCallback(
        eval_env,
        csv_path=csv_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        eval_seeds=eval_seeds,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    dt = time.time() - t0
    model.save(out_dir / "final_model")
    print(
        f"[train] done: {args.total_timesteps} steps in {dt:.1f}s "
        f"({args.total_timesteps / max(dt, 1e-9):.0f} steps/s)",
        flush=True,
    )

    train_env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
