#!/usr/bin/env python3
"""Fetch the Flatland-RL paper's published PPO learning curves from public W&B.

Reproduces Figure 6 of arXiv:2012.05893 ("Flatland-RL: Multi-Agent Reinforcement
Learning on Trains") from the Weights & Biases runs logged by the authors.

Uses the public, unauthenticated W&B GraphQL endpoint, so no API key is needed.
The original RLlib training stack (ray==0.8.5, tensorflow==2.1) is *not* required
to reproduce the curves -- only the logged run history is.

Usage:
    python -m src.baselines.fetch_wandb --discover
    python -m src.baselines.fetch_wandb --fetch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GRAPHQL_URL = "https://api.wandb.ai/graphql"

# Public projects that hold the paper's runs. The standard PPO runs live in the
# older `masterscrat/flatland` project; the newer `aicrowd/flatland-paper`
# project holds the follow-up experiments (masking/skip/IL).
PROJECTS: list[tuple[str, str]] = [
    ("aicrowd", "flatland-paper"),
    ("masterscrat", "flatland"),
]

# Preferred evaluation metrics (paper's Figure 6 y-axes), with training-log
# fallbacks for runs that did not log a separate evaluation series.
NORMALIZED_SCORE_KEYS = [
    "evaluation/custom_metrics/episode_score_normalized_mean",
    "custom_metrics/episode_score_normalized_mean",
]
COMPLETION_RATE_KEYS = [
    "evaluation/custom_metrics/percentage_complete_mean",
    "custom_metrics/percentage_complete_mean",
]
STEP_KEYS = ["timesteps_total", "_step"]

# Standard PPO runs are identified by tag; variants carry extra tags.
PPO_TAG = "ppo"
EXCLUDE_TAGS = {"mask", "skip", "imitation", "ccppo", "apex", "marwil", "dqn"}
# The paper trains pure RL for 15M steps; require a near-complete run.
MIN_PPO_STEPS = 14_000_000

DEFAULT_OUT = Path("data/baselines/flatland_paper_ppo")

RUNS_QUERY = """
query ProjectRuns($entity: String!, $project: String!, $first: Int!) {
  project(name: $project, entityName: $entity) {
    runs(first: $first) {
      edges {
        node { name displayName state createdAt config summaryMetrics tags }
      }
    }
  }
}
"""

HISTORY_QUERY = """
query RunHistory($entity: String!, $project: String!, $run: String!) {
  project(name: $project, entityName: $entity) {
    run(name: $run) { name displayName history }
  }
}
"""


def gql(query: str, variables: dict[str, Any], retries: int = 3) -> dict[str, Any]:
    """POST a GraphQL query to the public W&B API with simple retries."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("errors"):
                raise RuntimeError(f"GraphQL errors: {payload['errors']}")
            return payload["data"]
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GraphQL request failed after {retries} attempts: {last_err}")


def _unwrap(value: Any) -> Any:
    """W&B sometimes stores config entries as {"value": ..., "desc": ...}."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def cfg_get(config: dict[str, Any], *path: str) -> Any:
    cur: Any = config
    for key in path:
        cur = _unwrap(cur)
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return _unwrap(cur)


def parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}


def list_runs(entity: str, project: str, first: int = 800) -> list[dict[str, Any]]:
    data = gql(RUNS_QUERY, {"entity": entity, "project": project, "first": first})
    proj = data.get("project")
    if not proj:
        return []
    runs = []
    for edge in proj["runs"]["edges"]:
        node = edge["node"]
        node["_entity"] = entity
        node["_project"] = project
        runs.append(node)
    return runs


def looks_like_ppo(node: dict[str, Any]) -> bool:
    """Identify the paper's standard PPO runs.

    The `masterscrat/flatland` project holds hundreds of runs with opaque
    names and no algorithm name in `config.run`. The reliable marker is the
    ``ppo`` tag combined with the stock tree observation on the small sparse
    grid and the full ~15M-step training budget. Variants (action masking,
    frame skipping, imitation, centralized critic, Ape-X) carry extra tags and
    are excluded.
    """
    tags = {str(t).lower() for t in (node.get("tags") or [])}
    if PPO_TAG not in tags or tags & EXCLUDE_TAGS:
        return False

    cfg = parse_json(node.get("config"))
    if cfg_get(cfg, "env_config", "observation") != "tree":
        return False
    env = cfg_get(cfg, "env")
    if env is not None and "sparse" not in str(env).lower():
        return False

    steps = parse_json(node.get("summaryMetrics")).get("timesteps_total")
    if steps is None or float(steps) < MIN_PPO_STEPS:
        return False
    return True


def discover(projects: list[tuple[str, str]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for entity, project in projects:
        runs = list_runs(entity, project)
        print(f"[discover] {entity}/{project}: {len(runs)} runs", file=sys.stderr)
        for node in runs:
            if looks_like_ppo(node):
                cfg = parse_json(node.get("config"))
                summary = parse_json(node.get("summaryMetrics"))
                candidates.append(
                    {
                        "entity": entity,
                        "project": project,
                        "run_id": node["name"],
                        "display_name": node.get("displayName"),
                        "state": node.get("state"),
                        "created_at": node.get("createdAt"),
                        "run": cfg_get(cfg, "run"),
                        "env": cfg_get(cfg, "env"),
                        "observation": cfg_get(cfg, "env_config", "observation"),
                        "fcnet_hiddens": cfg_get(cfg, "model", "fcnet_hiddens"),
                        "steps": summary.get("timesteps_total"),
                        "has_eval_completion": any(
                            k in summary for k in COMPLETION_RATE_KEYS
                        ),
                        "tags": node.get("tags"),
                    }
                )
    return candidates


def fetch_history(
    entity: str, project: str, run_id: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    data = gql(
        HISTORY_QUERY,
        {"entity": entity, "project": project, "run": run_id},
    )
    run = data.get("project", {}).get("run")
    if not run:
        raise RuntimeError(f"Run not found: {entity}/{project}/{run_id}")

    records = [json.loads(row) for row in (run.get("history") or [])]
    rows: list[dict[str, Any]] = []
    for rec in records:
        step = next((rec[k] for k in STEP_KEYS if rec.get(k) is not None), None)
        norm = next(
            (rec[k] for k in NORMALIZED_SCORE_KEYS if rec.get(k) is not None), None
        )
        comp = next(
            (rec[k] for k in COMPLETION_RATE_KEYS if rec.get(k) is not None), None
        )
        if step is None or (norm is None and comp is None):
            continue
        rows.append(
            {"step": float(step), "normalized_score": norm, "completion_rate": comp}
        )

    df = pd.DataFrame(rows).dropna(subset=["step"]).sort_values("step")
    meta = {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "display_name": run.get("displayName"),
        "n_points": int(len(df)),
        "max_step": float(df["step"].max()) if len(df) else None,
    }
    return meta, df


def fetch_all(candidates: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        meta, df = fetch_history(cand["entity"], cand["project"], cand["run_id"])
        slug = (meta.get("display_name") or cand["run_id"]).replace("/", "_")
        csv_path = out_dir / f"ppo_{i:02d}_{slug}.csv"
        df.to_csv(csv_path, index=False)
        meta.update({"index": i, "csv": str(csv_path)})
        manifest.append(meta)
        print(
            f"[fetch] {i:02d} {meta['display_name']} "
            f"({meta['n_points']} pts, max_step={meta['max_step']}) -> {csv_path}",
            file=sys.stderr,
        )
        time.sleep(1)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[fetch] wrote manifest {manifest_path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--discover", action="store_true", help="list PPO candidates")
    group.add_argument("--fetch", action="store_true", help="download histories")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=None,
        help="explicit entity/project/run_id to fetch (repeatable); skips discovery",
    )
    args = parser.parse_args()

    if args.discover:
        candidates = discover(PROJECTS)
        print(json.dumps(candidates, indent=2))
        return 0

    if args.run_id:
        candidates = []
        for spec in args.run_id:
            entity, project, run_id = spec.split("/", 2)
            candidates.append(
                {"entity": entity, "project": project, "run_id": run_id}
            )
    else:
        candidates = discover(PROJECTS)

    if not candidates:
        print("[fetch] no PPO candidates found", file=sys.stderr)
        return 1
    fetch_all(candidates, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
