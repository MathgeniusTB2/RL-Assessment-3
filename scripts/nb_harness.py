"""Load the deliverable notebook's environment cells into a module namespace.

The notebook is the graded artifact and the single source of truth for how the
environment is built, so training and evaluation exec its cells instead of
maintaining a fork.  Only cells that define things are executed: no TODO stubs,
no plotting, no Colab installs beyond the guarded one in cell 2.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

# Cells the training/eval scripts need, by index in the deliverable notebook.
# If you add or remove a cell above these, update the tuple (the comment lists
# what each one provides).
# 4  imports + constants   7  ScenarioSpec     9  GymTreeObsForRailEnv
# 11 registry/conditions  13 env factory     15 greedy + force-stop + deadline calibration
# 22 vec-env wrappers     31 evaluate()
CELLS = (4, 7, 9, 11, 13, 15, 22, 31)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load(notebook: str | Path | None = None, scenario2: bool = False) -> Dict[str, Any]:
    root = repo_root()
    # the setup cell resolves data/ from the working directory; the notebook is run
    # from the repo root, so mirror that.
    os.chdir(root)
    path = Path(notebook) if notebook else root / "notebooks" / "train_schedule_optimisation.ipynb"
    cells = json.loads(path.read_text())["cells"]
    ns: Dict[str, Any] = {"__name__": "nb_harness", "__file__": str(path)}
    for i in CELLS:
        source = "".join(cells[i]["source"])
        exec(compile(source, f"{path.name}[cell {i}]", "exec"), ns)
    if scenario2:
        import s2_env

        ns = s2_env.load(ns)

    if scenario2:
        # s2_env.load() returns its own namespace: no SCENARIOS registry, so the
        # extra-scenario registration below only applies to the S1 notebook
        return ns

    # the corridor every result uses (four through tracks, ladder rows dropped)
    compact_level = root / "data" / "levels" / "network_simple_compact.pkl"
    if compact_level.exists():
        ns["SCENARIOS"]["network-simple-compact"] = ns["ScenarioSpec"](
            name="network-simple-compact",
            level_path=compact_level,
            schedule_path=root / "data" / "schedules" / "network_simple_compact_timetable.json",
            calibrate=False,
        )
    return ns
