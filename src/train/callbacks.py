"""Evaluation callback for the Flatland PPO baseline.

Runs deterministic evaluation episodes on held-out seeds and appends
``step, normalized_score, completion_rate`` rows to a CSV, matching the schema
used for the paper's W&B baseline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from flatland.envs.step_utils.states import TrainState


class FlatlandEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        csv_path: str | Path,
        eval_freq: int = 100_000,
        n_eval_episodes: int = 50,
        eval_seeds: list[int] | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = Path(csv_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seeds = eval_seeds
        self._next_eval = 0

    def _on_training_start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "normalized_score", "completion_rate"])

    def _run_episode(self, seed: int | None) -> tuple[float, float]:
        raw = self.eval_env._wrap
        n_agents = raw.get_num_agents()
        agents = list(self.eval_env.possible_agents)

        obs, _ = self.eval_env.reset(seed=seed)
        max_steps = raw._max_episode_steps
        cum = {a: 0.0 for a in agents}
        done = {a: False for a in agents}
        reached = {a: False for a in agents}

        for _ in range(max_steps):
            actions = {
                a: (0 if done[a] else int(self.model.predict(obs[a], deterministic=True)[0]))
                for a in agents
            }
            obs, rewards, terminations, _, infos = self.eval_env.step(actions)
            for a, r in rewards.items():
                if r is not None:
                    cum[a] += float(r)
            for a in agents:
                if terminations.get(a):
                    done[a] = True
                if infos[a].get("state") == TrainState.DONE:
                    reached[a] = True
            if all(done.values()):
                break

        rewards_arr = np.array([cum[a] for a in agents])
        # Flatland's normalize() returns the [0, 1] normalized reward; the paper
        # (and its W&B metric `episode_score_normalized`) reports the same value
        # without the +1 shift, i.e. in (-1, 0]. Subtract 1 to match that scale.
        normalized = float(
            raw.rewards.normalize(*rewards_arr, num_agents=n_agents, max_episode_steps=max_steps)
        ) - 1.0
        completion = sum(reached.values()) / n_agents
        return normalized, completion

    def _evaluate(self) -> tuple[float, float]:
        scores, comps = [], []
        for ep in range(self.n_eval_episodes):
            seed = None if self.eval_seeds is None else self.eval_seeds[ep % len(self.eval_seeds)]
            s, c = self._run_episode(seed)
            scores.append(s)
            comps.append(c)
        return float(np.mean(scores)), float(np.mean(comps))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            mean_score, mean_comp = self._evaluate()
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, mean_score, mean_comp])
            if self.verbose:
                print(
                    f"[eval] step={self.num_timesteps} "
                    f"normalized_score={mean_score:.3f} completion={mean_comp:.3f}",
                    flush=True,
                )
        return True
