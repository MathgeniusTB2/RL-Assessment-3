#!/usr/bin/env python3
"""Plot Flatland PPO training curves from ``eval.csv`` files.

Reads the per-seed evaluation CSVs produced by ``train_ppo`` and renders the
mean normalized score and completion rate versus training steps, with each run
shown as a fine line and the mean in bold.

Usage:
    python -m src.eval.plot_curves \
        --train-glob 'results/ppo/*/eval.csv' \
        --out results/ppo_training_curves.png
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TRAIN_GLOB = "results/ppo/*/eval.csv"
DEFAULT_OUT = Path("results/ppo_training_curves.png")

PANELS = [
    ("normalized_score", "Mean normalized score"),
    ("completion_rate", "Completion rate"),
]


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def smooth(y: np.ndarray, sigma_idx: float) -> np.ndarray:
    if sigma_idx <= 0:
        return y
    kernel = gaussian_kernel(sigma_idx)
    pad = len(kernel) // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def load_curves(pattern: str, grid: np.ndarray, sigma_idx: float) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path).dropna(subset=["step"]).sort_values("step")
        if df.empty:
            continue
        entry = {}
        for key, _ in PANELS:
            if key not in df:
                continue
            y = np.interp(grid, df["step"].values, df[key].values)
            entry[key] = smooth(y, sigma_idx)
        out[Path(path).parent.name if Path(path).name == "eval.csv" else Path(path).stem] = entry
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-glob", default=DEFAULT_TRAIN_GLOB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--smooth-sigma", type=float, default=250_000.0)
    p.add_argument("--max-step", type=float, default=15_000_000.0)
    args = p.parse_args()

    grid = np.linspace(0.0, args.max_step, 600)
    sigma_idx = args.smooth_sigma / (grid[1] - grid[0])

    train = load_curves(args.train_glob, grid, sigma_idx)
    if not train:
        print(f"No curves found for {args.train_glob}")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    steps_m = grid / 1e6

    for ax, (key, title) in zip(axes, PANELS):
        for name, series in train.items():
            if key in series:
                ax.plot(steps_m, series[key], color="tab:red", alpha=0.45, linewidth=0.9)
        stack = np.vstack([s[key] for s in train.values() if key in s])
        ax.plot(
            steps_m, stack.mean(axis=0), color="tab:red", linewidth=2.4,
            label=f"mean (n={len(stack)})",
        )
        ax.set_title(title)
        ax.set_xlabel("training steps (millions)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=False)
        if key == "completion_rate":
            ax.set_ylim(0, 1.05)

    fig.suptitle("Flatland PPO training curves", fontsize=12)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
