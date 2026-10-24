"""Detour teacher, behaviour-cloning labels and the warm-start checkpoint.

The corridor policy is warm-started, because from-scratch PPO converges to the
forward reflex and, once blockages are present, slides into a parking basin:

1. ``collect`` drives a hand-written teacher (forward everywhere, but turn at a
   commit crossover when the track ahead on my own row is blocked before my next
   waypoint) and stores (observation, action) pairs;
2. ``train`` fits the shared PPO policy to those labels, producing a normal SB3
   checkpoint that ``train_ppo.py --pretrain`` can load.

Usage:
  python scripts/bc_warmstart.py collect --episodes 20 --out runs/bc_block/data.npz
  python scripts/bc_warmstart.py train --data runs/bc_block/data.npz \
      --out runs/bc_block/bc3.zip --epochs 8 --turn-weight 20
"""
from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import numpy as np
from flatland.envs.rail_env_action import RailEnvActions

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import regimes  # noqa: E402  (owns the map geometry: COMMIT_CELLS)

MOVE_FORWARD = int(RailEnvActions.MOVE_FORWARD.value)
MOVE_LEFT = int(RailEnvActions.MOVE_LEFT.value)
MOVE_RIGHT = int(RailEnvActions.MOVE_RIGHT.value)


# --------------------------------------------------------------------- teacher
def configuration_of(agent):
    cfg = agent.current_configuration
    if cfg is None or cfg[0] is None:
        return None
    return (int(cfg[0][0]), int(cfg[0][1])), int(cfg[1])


def lateral_action(rail, position, direction):
    """MOVE_LEFT/MOVE_RIGHT if that transition exists at this cell, else None."""
    transitions = rail.get_transitions((position, direction))
    left = (direction - 1) % 4
    right = (direction + 1) % 4
    if transitions[left]:
        return MOVE_LEFT
    if transitions[right]:
        return MOVE_RIGHT
    return None


def next_waypoint_distance(agent, row: int, column: int, heading: int):
    """Distance to the nearest waypoint cell ahead of me on my own row.

    Geometric rather than indexed: Flatland keeps the full waypoint list, so
    ``waypoints[1]`` is still Burwood when the train is already past Redfern.
    """
    step = 1 if heading == 1 else -1 if heading == 3 else 0
    if step == 0:
        return None
    best = None
    for group in agent.waypoints:
        for waypoint in group:
            position = getattr(waypoint, "position", None)
            if position is None or int(position[0]) != row:
                continue
            gap = (int(position[1]) - column) * step
            if gap > 0 and (best is None or gap < best):
                best = gap
    return best


def make_detour_policy(raw):
    """env -> actions: turn onto the parallel track when the track ahead is blocked."""
    rail = raw.rail

    def policy(handles):
        hazards = set(getattr(raw, "_blocked_cells", ()) or ())
        for agent in raw.agents:
            cfg = configuration_of(agent)
            if cfg is not None and agent.malfunction_handler.in_malfunction:
                hazards.add(cfg[0])
        acts = {}
        for handle in handles:
            agent = raw.agents[handle]
            acts[handle] = MOVE_FORWARD
            cfg = configuration_of(agent)
            if cfg is None or not agent.state.is_on_map_state():
                continue
            (row, column), heading = cfg
            if (row, column) not in regimes.COMMIT_CELLS:
                continue
            step = 1 if heading == 1 else -1 if heading == 3 else 0
            if step == 0:
                continue
            stop = next_waypoint_distance(agent, row, column, heading)
            limit = 60 if stop is None else min(60, max(1, stop - 1))
            blocked_ahead = any(
                hr == row and 0 < (hc - column) * step <= limit for (hr, hc) in hazards
            )
            if not blocked_ahead:
                continue
            action = lateral_action(rail, (row, column), heading)
            if action is not None:
                acts[handle] = action
        return acts

    return policy


# ------------------------------------------------------------------ env helper
def _priority_env(scenario: str, incidents: int, variant: str | None, install_regime: bool):
    """Parallel env with the reported observation width (tree + 12 features)."""
    import nb_harness
    import priority_obs

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
    state = None
    if install_regime:
        state = regimes.install(raw, scenario, incidents, variant=variant)
    return nb, par, raw, state


# ---------------------------------------------------------------------- collect
def collect(args) -> int:
    _, par, raw, _ = _priority_env(args.scenario, args.incidents, args.variant, True)
    policy = make_detour_policy(raw)
    handles = list(par.possible_agents)
    T = int(raw._max_episode_steps)
    print(f"regime={regimes.selected_variant(args.variant)} T={T}")

    obs_buf, act_buf = [], []
    for episode in range(args.episodes):
        seed = secrets.randbits(31)
        obs, _ = par.reset(seed=seed)
        turns = steps = 0
        for step in range(1, T + 1):
            acts = policy(handles)
            keep = step % args.keep_every == 0 or any(a != MOVE_FORWARD for a in acts.values())
            if keep:
                obs_buf.append(np.stack([obs[h] for h in handles]).astype(np.float16))
                act_buf.append(np.array([acts[h] for h in handles], dtype=np.int8))
                steps += 1
                turns += sum(1 for a in acts.values() if a != MOVE_FORWARD)
            obs, _, terms, _, _ = par.step(acts)
            if all(v for k, v in terms.items() if k != "__all__"):
                break
        print(f"  episode {episode + 1}/{args.episodes} steps={step} kept={steps} "
              f"turns={turns}", flush=True)

    obs_arr = np.concatenate(obs_buf, axis=0)
    act_arr = np.concatenate(act_buf, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, obs=obs_arr, acts=act_arr)
    print(f"wrote {out}: obs={obs_arr.shape} acts={act_arr.shape} "
          f"turn_share={float((act_arr != MOVE_FORWARD).mean()):.4f}")
    return 0


# ------------------------------------------------------------------------ train
def train(args) -> int:
    import supersuit as ss
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(args.torch_threads)

    parts = [np.load(name) for name in args.data.split(",")]
    obs_all = np.concatenate([part["obs"] for part in parts]).astype(np.float32)
    act_all = np.concatenate([part["acts"] for part in parts]).astype(np.int64)
    print(f"data obs={obs_all.shape} acts={act_all.shape} "
          f"turn_share={float((act_all != MOVE_FORWARD).mean()):.4f}")

    _, par, _, _ = _priority_env(args.scenario, args.incidents, args.variant, False)
    vec = ss.pettingzoo_env_to_vec_env_v1(par)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=0, base_class="stable_baselines3")
    model = PPO("MlpPolicy", vec, n_steps=512, batch_size=2048, learning_rate=3e-4,
                gamma=0.999, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001,
                vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]), device="cpu", verbose=0)
    print("obs space", model.observation_space.shape, "act space", model.action_space)

    weights = np.ones(len(act_all), dtype=np.float32)
    turn = act_all != MOVE_FORWARD
    weights[turn] = float(args.turn_weight)
    print(f"turn weight {args.turn_weight}, turn share {float(turn.mean()):.4f}")

    obs_t = torch.as_tensor(obs_all)
    act_t = torch.as_tensor(act_all)
    w_t = torch.as_tensor(weights)
    opt = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
    n = len(act_t)
    rng = np.random.default_rng(0)
    t0 = time.time()

    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = correct = 0
        loss_sum = 0.0
        for start in range(0, n, args.batch_size):
            idx = torch.as_tensor(order[start:start + args.batch_size])
            obs_b, act_b = obs_t[idx], act_t[idx]
            _, log_prob, _ = model.policy.evaluate_actions(obs_b, act_b)
            loss = -(log_prob * w_t[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                pred = model.policy.get_distribution(obs_b).get_actions(deterministic=True)
                correct += int((pred == act_b).sum())
            total += len(idx)
            loss_sum += float(loss) * len(idx)
        with torch.no_grad():
            pred_turn = model.policy.get_distribution(
                obs_t[turn][:20000]).get_actions(deterministic=True)
            recall = float((pred_turn == act_t[turn][:20000]).float().mean())
        print(f"epoch {epoch + 1}/{args.epochs} loss={loss_sum / total:.4f} "
              f"acc={correct / total:.4f} turn_recall={recall:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"wrote {args.out}")
    return 0


# ------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    col = sub.add_parser("collect", help="label episodes with the detour teacher")
    col.add_argument("--scenario", default="network-simple-compact")
    col.add_argument("--incidents", type=int, default=2)
    col.add_argument("--episodes", type=int, default=20)
    col.add_argument("--keep-every", type=int, default=3, help="background step stride")
    col.add_argument("--variant", default="block")
    col.add_argument("--out", default="runs/bc_block/data.npz")

    tr = sub.add_parser("train", help="behaviour-clone the labels into a PPO checkpoint")
    tr.add_argument("--data", default="runs/bc_block/data.npz",
                    help="one or more comma-separated .npz files from collect")
    tr.add_argument("--out", default="runs/bc_block/bc3.zip")
    tr.add_argument("--scenario", default="network-simple-compact")
    tr.add_argument("--incidents", type=int, default=2)
    tr.add_argument("--variant", default="block")
    tr.add_argument("--epochs", type=int, default=8)
    tr.add_argument("--batch-size", type=int, default=4096)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--turn-weight", type=float, default=20.0,
                    help="extra weight on turning labels (~0.2%% of the data)")
    tr.add_argument("--torch-threads", type=int, default=8)

    args = ap.parse_args()
    return collect(args) if args.command == "collect" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
