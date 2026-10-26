"""Evidence readers shared by the report builders.

Every number in the report comes through here, from the CSVs in results/, so the
Markdown and the .docx cannot drift from the measurements.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = RES / "report_figures"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pair import sign_test_p  # noqa: E402

def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def paired(path: Path, arm: str = "ppo", base: str = "greedy") -> dict:
    """Paired statistics from a results/pair*.csv written by eval_pair.py."""
    data = rows(path)
    a = {int(r["seed"]): r for r in data if r["policy"] == arm}
    b = {int(r["seed"]): r for r in data if r["policy"] == base}
    seeds = sorted(set(a) & set(b))
    da = [float(a[s]["score"]) for s in seeds]
    db = [float(b[s]["score"]) for s in seeds]
    diff = [x - y for x, y in zip(da, db, strict=True)]
    wins = sum(d > 0 for d in diff)
    losses = sum(d < 0 for d in diff)
    return {
        "n": len(seeds),
        "base": statistics.fmean(db),
        "arm": statistics.fmean(da),
        "delta": statistics.fmean(diff),
        "median": statistics.median(diff),
        "wins": wins,
        "losses": losses,
        "ties": len(seeds) - wins - losses,
        "p": sign_test_p(wins, losses),
        "compl_base": statistics.fmean(float(b[s]["completion"]) for s in seeds),
        "compl_arm": statistics.fmean(float(a[s]["completion"]) for s in seeds),
        "delay_base": statistics.fmean(float(b[s]["delay_all"]) for s in seeds),
        "delay_arm": statistics.fmean(float(a[s]["delay_all"]) for s in seeds),
        "on_time_base": statistics.fmean(float(b[s]["on_time"]) for s in seeds),
        "on_time_arm": statistics.fmean(float(a[s]["on_time"]) for s in seeds),
    }


def best_checkpoint(name: str) -> tuple[str, float, float]:
    """The checkpoint selection rule: completion first, then score."""
    data = rows(RES / "curves" / f"{name}_corridor_block.csv")
    best = max(data, key=lambda r: (float(r["completion"]), float(r["score"])))
    return f"{int(best['timesteps']) // 1000}k", float(best["score"]), float(best["completion"])


def audit_table() -> tuple[list[list[str]], int, int, int, int]:
    data = rows(RES / "crossing_audit_block.csv")
    cells: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for r in data:
        key = (int(r["row"]), int(r["col"]))
        cells.setdefault(key, {})[int(r["blocked"])] = (int(r["visits"]), int(r["turns"]))
    tracks = {2: "up_suburban", 4: "up_local", 5: "down_local"}
    body, tot_b, tot_bt, tot_o, tot_ot = [], 0, 0, 0, 0
    for (row, col) in sorted(cells):
        bv, bt = cells[(row, col)].get(1, (0, 0))
        ov, ot = cells[(row, col)].get(0, (0, 0))
        tot_b += bv
        tot_bt += bt
        tot_o += ov
        tot_ot += ot
        body.append([f"({row},{col})", tracks.get(row, "?"), str(bv), str(bt), str(ov), str(ot)])
    body.append(["Total", "", str(tot_b), str(tot_bt), str(tot_o), str(tot_ot)])
    return body, tot_b, tot_bt, tot_o, tot_ot


def pct(x: float) -> str:
    """Scores and deltas to the four decimals the evidence supports."""
    return f"{x:.4f}"


def comp(x: float) -> str:
    """Completion shares to three decimals."""
    return f"{x:.3f}"


