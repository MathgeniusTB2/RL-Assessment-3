"""Crossing diagnostics for the corridor: score, decisions, exposure.

Three questions, one entry point:

  score      what does a policy score, how many incidents fired, and how many
             row changes were responsive (made while the track ahead was blocked)
  decisions  at each commit crossover, how often did the policy turn when the
             track ahead was blocked versus open
  liveness   for the regime design: how often a blockage placed d cells past a
             crossover would still have a train on the approach side

Usage:
  RL_A3_REGIME=block python scripts/crossing_audit.py score --seeds 3000,3001,3002 \
      --incidents 2 --model runs/xb_b/best_model.zip
  RL_A3_REGIME=block python scripts/crossing_audit.py decisions --seeds 3000,3001
  python scripts/crossing_audit.py liveness --seeds 3000 --incidents 0
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

MOVE_FORWARD = 2


# ------------------------------------------------------------------ env helper
def build(scenario: str, incidents: int, variant: str | None):
    """Parallel env with priority features at radius 50 and the regime installed."""
    import nb_harness
    import priority_obs
    import regimes

    nb = nb_harness.load(scenario2=scenario != "network-simple-compact")
    priority_obs.install_priority_features(nb)
    cls = nb["GymTreeObsForRailEnv"]
    original_init = cls.__init__

    def patched(self, *a, **kw):
        kw.setdefault("observation_radius", 50)
        original_init(self, *a, **kw)

    cls.__init__ = patched
    par = nb["make_parallel_env"](scenario=scenario, condition="normal")
    raw = par._wrap
    if incidents:
        state = regimes.install(raw, scenario, incidents, variant=variant)
    else:
        state = {"fired": [], "approach": {}}
    return par, raw, state, list(par.possible_agents)


def position_of(agent):
    cfg = agent.current_configuration
    if cfg is None or cfg[0] is None:
        return None
    return int(cfg[0][0]), int(cfg[0][1]), int(cfg[1])


def make_policy(args, raw, handles):
    """forward | detour | a checkpoint."""
    if args.policy == "detour":
        from bc_warmstart import make_detour_policy

        return make_detour_policy(raw)
    if args.model:
        from stable_baselines3 import A2C, DQN, PPO

        model = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo].load(args.model, device="cpu")

        def policy(obs_map=None, actions=None):
            batch = np.stack([obs_map[h] for h in handles])
            chosen = model.predict(batch, deterministic=True)[0]
            return {h: int(a) for h, a in zip(handles, chosen)}

        return policy
    return None


def hazards(raw):
    """Cells no train may enter right now (blocked cell or malfunctioning train)."""
    cells = set(getattr(raw, "_blocked_cells", ()) or ())
    for agent in raw.agents:
        cfg = agent.current_configuration
        if cfg is not None and cfg[0] is not None and agent.malfunction_handler.in_malfunction:
            cells.add((int(cfg[0][0]), int(cfg[0][1])))
    return cells


def blocked_ahead(handles, raw, hazard_cells):
    """Handles that have an impassable cell ahead on their own row within 50 cells."""
    out = set()
    for handle in handles:
        agent = raw.agents[handle]
        point = position_of(agent)
        if point is None or not agent.state.is_on_map_state():
            continue
        row, col, heading = point
        step = 1 if heading == 1 else -1 if heading == 3 else 0
        if not step:
            continue
        for (hr, hc) in hazard_cells:
            if hr == row and 0 < (hc - col) * step <= 50:
                out.add(handle)
                break
    return out


# ---------------------------------------------------------------------- score
def run_score(args) -> int:
    from flatland.envs.step_utils.states import TrainState

    par, raw, state, handles = build(args.scenario, args.incidents, args.variant)
    model_fn = make_policy(args, raw, handles)
    T = int(raw._max_episode_steps)
    print(f"regime={args.variant} incidents={args.incidents} T={T}")

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, _ = par.reset(seed=seed)
        cum = {h: 0.0 for h in handles}
        done_step, prev_row = {}, {}
        crossings = responsive = 0
        for step in range(1, T + 1):
            ahead = blocked_ahead(handles, raw, hazards(raw))
            if model_fn is None:
                acts = {h: MOVE_FORWARD for h in handles}
            else:
                acts = model_fn(obs, None)
            obs, rewards, terms, _, infos = par.step(acts)
            for h, r in rewards.items():
                if r is not None:
                    cum[h] += float(r)
            for h in handles:
                agent = raw.agents[h]
                if infos[h].get("state") == TrainState.DONE and h not in done_step:
                    done_step[h] = step
                point = position_of(agent)
                if point is None:
                    continue
                if h in prev_row and prev_row[h] != point[0]:
                    crossings += 1
                    if h in ahead:
                        responsive += 1
                prev_row[h] = point[0]
            if all(v for k, v in terms.items() if k != "__all__"):
                break

        rewards_arr = np.array([cum[h] for h in handles])
        score = float(raw.rewards.normalize(
            *rewards_arr, num_agents=len(handles), max_episode_steps=T)) - 1.0
        scheduled = [raw.agents[h].waypoints_latest_arrival[-1] for h in handles]
        delay_all = float(np.sum([(done_step.get(h, step) - scheduled[h])
                                  for h in handles if scheduled[h] is not None]))
        approach = sum(
            1 for f in state["fired"]
            if state.get("approach", {}).get((f[1], f[2]), (None, None))[0] is not None
        )
        rows.append(dict(seed=seed, score=score, completion=len(done_step) / len(handles),
                         delay_all=delay_all, fired=len(state["fired"]),
                         crossings=crossings, responsive=responsive,
                         incidents_with_escape=approach))
        print(f"  seed={seed} score={score:+.4f} compl={rows[-1]['completion']:.3f} "
              f"delay_all={delay_all:7.0f} fired={len(state['fired'])} "
              f"crossings={crossings} responsive={responsive}", flush=True)

    print("\nmean  " + "  ".join(
        f"{key}={np.mean([r[key] for r in rows]):.3f}"
        for key in ("score", "completion", "delay_all", "fired", "crossings", "responsive")))
    return 0


# ------------------------------------------------------------------ decisions
def run_decisions(args) -> int:
    par, raw, state, handles = build(args.scenario, args.incidents, args.variant)
    model_fn = make_policy(args, raw, handles)
    commits = __import__("regimes").COMMIT_CELLS
    T = int(raw._max_episode_steps)

    stats = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, _ = par.reset(seed=seed)
        for step in range(1, T + 1):
            cells = hazards(raw)
            if model_fn is None:
                acts = {h: MOVE_FORWARD for h in handles}
            else:
                acts = model_fn(obs, None)
            for h in handles:
                point = position_of(raw.agents[h])
                if point is None or (point[0], point[1]) not in commits:
                    continue
                row, col, heading = point
                step_dir = 1 if heading == 1 else -1 if heading == 3 else 0
                ahead = any(hr == row and 0 < (hc - col) * step_dir <= 60
                            for (hr, hc) in cells) if step_dir else False
                rec = stats.setdefault(((row, col), ahead), [0, 0])
                rec[0] += 1
                if acts[h] in (1, 3):
                    rec[1] += 1
            obs, _, terms, _, _ = par.step(acts)
            if all(v for k, v in terms.items() if k != "__all__"):
                break

    print(f"\nvariant={args.variant} incidents={args.incidents} model={args.model}")
    print("cell          blocked  visits  turns  turn-rate")
    for ((row, col), ahead), (visits, turns) in sorted(stats.items()):
        print(f"({row},{col})".ljust(14) + f"{str(ahead):<9}{visits:<8}{turns:<7}"
              f"{turns / visits:.3f}")
    for flag in (True, False):
        visits = sum(v[0] for (cell, blocked), v in stats.items() if blocked == flag)
        turns = sum(v[1] for (cell, blocked), v in stats.items() if blocked == flag)
        label = "blocked" if flag else "open   "
        print(f"\n{label} visits={visits} turns={turns}")

    if getattr(args, "out", None):
        import csv
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "col", "blocked", "visits", "turns", "seeds", "model"])
            for ((row, col), blocked), (visits, turns) in sorted(stats.items()):
                w.writerow([row, col, int(blocked), visits, turns, args.seeds, args.model or "forward"])
        print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------- liveness
def run_liveness(args) -> int:
    """How often is a blockage placed d cells past a crossover still avoidable?"""
    from flatland.envs.rail_env_action import RailEnvActions

    par, raw, state, handles = build(args.scenario, 0, None)
    dists = [int(d) for d in args.distances.split(",")]
    print(f"scenario={args.scenario} T={int(raw._max_episode_steps)} lead={dists}")

    # segments: (row, commit column, direction, station)
    import json

    from regimes import COMPACT_UP_ROWS, REPO

    geometry = json.loads((REPO / "results" / "station_access_compact.json").read_text())
    segments = []
    for entry in geometry:
        if not entry.get("alt_available") or entry.get("commit_from_col") is None:
            continue
        row, station_col = int(entry["row"]), int(entry["station_col"])
        commit = int(entry["commit_from_col"])
        segments.append((row, commit, "up" if row in COMPACT_UP_ROWS else "down",
                         entry["station"], max(1, abs(station_col - commit)) - 1))

    stats = {(row, commit): {d: [0, 0] for d in dists} for row, commit, *_ in segments}
    for seed in [int(s) for s in args.seeds.split(",")]:
        par.reset(seed=seed)
        for _ in range(int(raw._max_episode_steps)):
            obs, _, terms, _, _ = par.step(
                {h: int(RailEnvActions.MOVE_FORWARD.value) for h in handles})
            rows = {}
            for agent in raw.agents:
                point = position_of(agent)
                if point:
                    rows.setdefault(point[0], []).append(point[1])
            for (row, commit, direction, _station, _band) in segments:
                cols = rows.get(row, [])
                approach = (any(c < commit for c in cols) if direction == "up"
                            else any(c > commit for c in cols))
                for d in dists:
                    trigger = (any(commit < c <= commit + d for c in cols) if direction == "up"
                               else any(commit - d <= c < commit for c in cols))
                    if not trigger:
                        continue
                    stats[(row, commit)][d][0] += 1
                    if approach:
                        stats[(row, commit)][d][1] += 1
            if all(v for k, v in terms.items() if k != "__all__"):
                break

    print("\nlive trigger steps / total trigger steps per segment")
    for (row, commit, direction, station, band) in segments:
        cells = " ".join(f"d={d}:{stats[(row, commit)][d][1]}/"
                         f"{stats[(row, commit)][d][0]}" for d in dists)
        print(f"  row {row} commit {commit:>3} {direction:<5} {station:<12} "
              f"band {band:>2}  {cells}")
    return 0


# ----------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    for name, help_text in (("score", "score, fired count and responsive crossings"),
                            ("decisions", "per-crossover blocked vs open turn audit"),
                            ("liveness", "regime exposure at lead distances")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--seeds", default="3000,3001,3002")
        p.add_argument("--scenario", default="network-simple-compact")
        p.add_argument("--incidents", type=int, default=2)
        p.add_argument("--variant", default="block")
        p.add_argument("--policy", default=None, choices=[None, "forward", "detour"])
        p.add_argument("--model", default=None)
        if name == "decisions":
            p.add_argument("--out", default=None, help="write the per-crossover table as CSV")
            p.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"])
        if name == "liveness":
            p.add_argument("--distances", default="2,4,6,8,12")

    args = ap.parse_args()
    if args.command == "score":
        return run_score(args)
    if args.command == "decisions":
        return run_decisions(args)
    return run_liveness(args)


if __name__ == "__main__":
    raise SystemExit(main())
