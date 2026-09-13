#!/usr/bin/env python3
"""Plot the Flatland-RL paper's PPO baseline curves (Figure 6 style).

Reads the per-run CSVs produced by ``fetch_wandb_baseline.py`` and renders the
mean normalized score and completion rate versus training steps, with each run
shown as a fine line and the mean as a bold line, matching Figure 6 of
arXiv:2012.05893.

Usage:
    python -m src.baselines.plot_baseline
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

DEFAULT_DATA_DIR = Path("data/baselines/flatland_paper_ppo")
DEFAULT_OUT = Path("results/baselines/ppo_baseline_curves.png")

PANELS = [
    ("normalized_score", "Mean normalized score", "normalized score"),
    ("completion_rate", "Completion rate", "completion rate"),
]


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return y
    kernel = gaussian_kernel(sigma)
    pad = len(kernel) // 2
    # Edge-pad so the convolution does not drag the endpoints toward zero.
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def load_runs(data_dir: Path, min_step: float) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(str(data_dir / "ppo_*.csv"))):
        df = pd.read_csv(path)
        df = df[df["step"] >= min_step]
        df = df.groupby("step", as_index=False).mean(numeric_only=True)
        runs[Path(path).stem] = df.sort_values("step")
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=250_000.0,
        help="Gaussian smoothing sigma in training steps",
    )
    parser.add_argument(
        "--min-step",
        type=float,
        default=0.0,
        help="drop points before this many training steps",
    )
    parser.add_argument("--max-step", type=float, default=None)
    args = parser.parse_args()

    runs = load_runs(args.data_dir, args.min_step)
    if not runs:
        print(f"No ppo_*.csv found in {args.data_dir}")
        return 1

    max_step = args.max_step or max(df["step"].max() for df in runs.values())
    grid = np.linspace(0.0, max_step, 600)
    # Convert the smoothing sigma from training steps to grid-index units.
    sigma_idx = args.smooth_sigma / (grid[1] - grid[0])

    smoothed: dict[str, dict[str, np.ndarray]] = {}
    for name, df in runs.items():
        smoothed[name] = {}
        for key, _, _ in PANELS:
            y = np.interp(grid, df["step"].values, df[key].values)
            smoothed[name][key] = smooth(y, sigma_idx)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    steps_m = grid / 1e6

    for ax, (key, title, ylabel) in zip(axes, PANELS):
        stack = np.vstack([smoothed[name][key] for name in smoothed])
        for name in smoothed:
            ax.plot(
                steps_m,
                smoothed[name][key],
                color="tab:blue",
                alpha=0.25,
                linewidth=0.8,
            )
        ax.plot(steps_m, stack.mean(axis=0), color="tab:blue", linewidth=2.2,
                label=f"mean (n={len(smoothed)})")
        ax.set_title(title)
        ax.set_xlabel("training steps (millions)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=False)
        if key == "completion_rate":
            ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Flatland-RL paper baseline: standard PPO (arXiv:2012.05893, Fig. 6)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # Summary stats over the final 1M steps, to sanity-check against Table 1.
    print("\nFinal-window (last 1M steps) per run:")
    for name, df in runs.items():
        tail = df[df["step"] >= max_step - 1_000_000]
        print(
            f"  {name:34s} completion={tail['completion_rate'].mean():.3f} "
            f"score={tail['normalized_score'].mean():.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
