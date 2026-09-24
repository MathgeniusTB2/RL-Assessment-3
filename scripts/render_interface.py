#!/usr/bin/env python3
"""Render the Flatland interface figure for the Part-B report.

Builds a 24x24 sparse Flatland scene with three trains, steps a few times so the
agents are mid-simulation, and writes one headless PIL frame to
``docs/interface_flatland_simulation.png`` (Figure 1 in the Part-B plan).

Usage:
    python scripts/render_interface.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from flatland.env_generation.env_generator import env_generator
from flatland.utils.rendertools import RenderTool

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "interface_flatland_simulation.png"


def build_env(seed: int, n_agents: int, size: int):
    env, _, _ = env_generator(
        n_agents=n_agents,
        x_dim=size,
        y_dim=size,
        n_cities=2,
        grid_mode=False,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=1,
        seed=seed,
    )
    return env


def render(
    out_path: Path,
    seed: int,
    n_agents: int,
    size: int,
    warmup_steps: int,
    screen_width: int,
) -> None:
    env = build_env(seed=seed, n_agents=n_agents, size=size)
    env.reset(random_seed=seed)
    for _ in range(warmup_steps):
        env.step({h: 2 for h in env.get_agent_handles()})  # 2 = MOVE_FORWARD

    renderer = RenderTool(
        env, gl="PIL", screen_width=screen_width, screen_height=screen_width
    )
    img = renderer.render_env(
        show=False, show_agents=True, show_observations=False, return_image=True
    )
    renderer.close_window()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    print(f"wrote {out_path} ({img.shape[1]}x{img.shape[0]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-agents", type=int, default=3)
    ap.add_argument("--size", type=int, default=24)
    ap.add_argument("--warmup-steps", type=int, default=15)
    ap.add_argument("--screen-width", type=int, default=800)
    args = ap.parse_args()

    render(
        args.out,
        args.seed,
        args.n_agents,
        args.size,
        args.warmup_steps,
        args.screen_width,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
