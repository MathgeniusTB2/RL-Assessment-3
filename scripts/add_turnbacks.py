#!/usr/bin/env python3
"""Add dead-end turnbacks to Central platform cells of a Flatland .mpk.

A dead-end cell encodes a single reverse transition (E->W for the eastbound
platforms here), so a train that arrives facing east turns 180 degrees and runs
back west when it moves forward (see RailEnvActions / the Flatland docs).

Usage:
    python scripts/add_turnbacks.py --src "network (5).mpk" \
        --out data/levels/central_strathfield.mpk --rows 0,2,4,6,8,10,12,14,16,18,20,22,24,26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import msgpack
import numpy as np
from flatland.core.grid.grid4 import fast_grid4_set_transitions
from flatland.envs.persistence import RailEnvPersister


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--col", type=int, default=84, help="Central platform column")
    ap.add_argument("--rows", default="28,30,32,34", help="comma-separated platform rows")
    ap.add_argument(
        "--both-ways",
        action="store_true",
        help="allow reversal from both facings (W->E as well); default E->W only",
    )
    args = ap.parse_args()

    # Edit the raw env dict and re-pack it. `RailEnvPersister.save` on an already
    # loaded env writes a `malfunction` tuple that `load_new` cannot read back,
    # so go through `load_env_dict`/msgpack instead.
    env_dict = RailEnvPersister.load_env_dict(str(args.src))
    grid = np.array(env_dict["grid"])  # msgpack arrays are read-only; take a copy
    env_dict["grid"] = grid
    rows = [int(r) for r in args.rows.split(",") if r.strip()]
    # E->W (facing East, moving West): the corridor runs west->east.
    value = fast_grid4_set_transitions(0, 1, (0, 0, 0, 1))
    if args.both_ways:
        value = fast_grid4_set_transitions(value, 3, (0, 1, 0, 0))
    for r in rows:
        grid[r][args.col] = value
        print(f"  ({r},{args.col}) -> {value}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        fh.write(msgpack.packb(env_dict))
    print(f"[turnbacks] wrote {args.out}: rows {rows} now dead-ends at col {args.col}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
