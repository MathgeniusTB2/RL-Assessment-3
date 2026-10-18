#!/usr/bin/env python3
"""Adopt the hand-built Scenario-2 level and repair the Down Suburban.

Takes the level you painted in the RailEnv Editor (`~/Downloads/network.mpk`),
makes the Down Suburban (row 7) straight through by swapping the two cells that
break it, and saves it as the scenario level.

Only two cells change:
    (7,10)  turn N (2064)   -> switch WR (1097)   straight E-W + north crossover
    (7,11)  switch SL (32872) -> diamond  (33825)  row 7 crosses col 11

Usage:
    python scripts/adopt_scenario2_level.py [--src ~/Downloads/network.mpk]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import msgpack
import numpy as np

from flatland.envs.persistence import RailEnvPersister

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path.home() / "Downloads" / "network.mpk"
DEFAULT_OUT = REPO_ROOT / "data" / "levels" / "scenario2_env.mpk"

# the repaired Down Suburban cells
PATCH = {(7, 10): 1097, (7, 11): 33825, (7, 12): 17411}


def adopt(src: Path, out: Path) -> None:
    env_dict = RailEnvPersister.load_env_dict(str(src))
    grid = np.array(env_dict["grid"], dtype=env_dict["grid"].dtype)
    for (r, c), value in PATCH.items():
        grid[r, c] = value
    env_dict["grid"] = grid
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(msgpack.packb(env_dict))
    print(f"wrote {out} from {src}")
    _report(grid)


def _report(grid: np.ndarray) -> None:
    N, E, S, W = 0, 1, 2, 3

    def outs(v, f):
        return [o for o in range(4) if v >> (4 * (3 - f) + (3 - o)) & 1]

    def run(row, start, d, step):
        p = [start, start]
        p[1] = start
        col = start
        for _ in range(40):
            o = outs(int(grid[row, col]), d)
            if d not in o:
                break
            col += step
            if not (0 <= col < grid.shape[1]):
                break
        return col

    print("  through rows (MOVE_FORWARD only):")
    for r in (4, 5, 6, 7):
        e = run(r, 1, E, 1)
        w = run(r, 18, W, -1)
        ok = "OK" if (e >= 18 and w <= 1) else f"GAP east->{e} west->{w}"
        print(f"    row {r}: {ok}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")
    adopt(args.src, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
