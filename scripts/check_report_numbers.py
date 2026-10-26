"""Check that the numbers in report/part-e.md still match the shipped evidence.

The report is hand-edited Markdown, so this guards the other direction: every
figure that was derived from `results/` is recomputed here and must appear in the
text verbatim. Run it after editing the report and before submitting.

    python scripts/check_report_numbers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_data as rd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MD = REPO / "report" / "part-e.md"
RES = REPO / "results"


def expected() -> list[tuple[str, str, str]]:
    """(value, where it must appear, which file backs it)."""
    a = rd.paired(RES / "pair_block_ppo.csv")
    clean = rd.paired(RES / "clean_normal.csv")
    tight = rd.paired(RES / "clean_delayed_none.csv")
    jc = rd.paired(RES / "junction_current.csv")
    audit_rows, tb, tb_turns, to, to_turns = rd.audit_table()
    _ck, _sc, _cp = rd.best_checkpoint("dqn")

    rows = rd.rows(RES / "sensitivity_short.csv")
    short = {p: [float(r["score"]) for r in rows if r["policy"] == p] for p in ("greedy", "ppo")}
    short_tie = abs(sum(short["greedy"]) - sum(short["ppo"])) < 1e-12

    return [
        (rd.pct(a["base"]), "corridor baseline", "pair_block_ppo.csv"),
        (rd.pct(a["arm"]), "corridor arm", "pair_block_ppo.csv"),
        (f"{a['delta']:+.4f}", "corridor delta", "pair_block_ppo.csv"),
        (f"{a['median']:+.4f}", "corridor median", "pair_block_ppo.csv"),
        (f"{a['wins']} of {a['n']}", "corridor wins", "pair_block_ppo.csv"),
        (f"{tb}", "audit blocked visits", "crossing_audit_block.csv"),
        (f"{tb_turns}", "audit blocked turns", "crossing_audit_block.csv"),
        (f"{to}", "audit open visits", "crossing_audit_block.csv"),
        (f"{to_turns}", "audit open turns", "crossing_audit_block.csv"),
        (rd.pct(clean["arm"]), "clean arm score", "clean_normal.csv"),
        (rd.pct(tight["arm"]), "malfunction control arm", "clean_delayed_none.csv"),
        (rd.pct(tight["base"]), "malfunction control baseline", "clean_delayed_none.csv"),
        (f"p = {tight['p']:.2f}", "malfunction control p", "clean_delayed_none.csv"),
        (rd.pct(jc["base"]), "junction baseline", "junction_current.csv"),
        (rd.pct(jc["arm"]), "junction arm", "junction_current.csv"),
        (f"{jc['compl_arm'] * 126:.0f} of 126", "junction completions", "junction_current.csv"),
        ("546", "clean audit visits", "clean_audit.csv"),
        ("0.0005", "teacher score", "all_arms.md"),
        ("3000 and 3001", "audit seed pair", "crossing_audit_block.csv"),
    ], {"short_tie": short_tie, "dqn_completion": _cp}


def main() -> int:
    text = MD.read_text()
    checks, extras = expected()
    problems = []

    for value, label, source in checks:
        if value not in text:
            problems.append(f"{label}: {value!r} (from {source}) is not in the report")

    if not extras["short_tie"]:
        problems.append("sensitivity_short.csv no longer shows an exact tie between the arms")
    if extras["dqn_completion"] > 0.6:
        problems.append(f"DQN best-checkpoint completion is {extras['dqn_completion']:.3f}, "
                        "so 'parks half the fleet' no longer holds")

    # the audit table in the report must carry the same per-crossover counts
    for row in rd.rows(RES / "crossing_audit_block.csv"):
        cell = f"({row['row']},{row['col']})"
        if row["blocked"] == "1" and row["turns"] != "0" and cell not in text:
            problems.append(f"audit row {cell} missing from the report's Table 4")

    if problems:
        print("MISMATCH")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK: {len(checks)} figures in report/part-e.md match results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
