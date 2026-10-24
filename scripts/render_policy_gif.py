"""Render one episode of a trained policy to a GIF, under the designed regime.

    python scripts/render_policy_gif.py --algo ppo \
        --model runs/v2_ppo_compact/best_model.zip \
        --scenario network-simple-compact --seed 3000 \
        --out results/figures/policy_ppo_corridor.gif

The notebook displays these; the checkpoints themselves are not committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "dqn"])
    ap.add_argument("--model", default=None,
                    help="checkpoint to render; omit with --policy forward for the baseline")
    ap.add_argument("--policy", default=None, choices=["forward"],
                    help="render the forward-only reflex instead of a checkpoint")
    ap.add_argument("--scenario", default="network-simple-compact")
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--incidents", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=12, help="render every Nth step")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=72)
    ap.add_argument("--crop-cols", default=None,
                    help="a,b - keep only map columns a..b (the decision stretch)")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--switch-hold", type=int, default=12,
                    help="frames to keep a green marker on a cell where a train changed row")
    ap.add_argument("--markers", default="block", choices=["block", "all"],
                    help="block = red X on impassable cells only (report figures); "
                         "all = also outline crossovers and mark row changes in green")
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    import nb_harness
    import priority_obs
    import regimes

    from flatland.envs.step_utils.states import TrainState
    from flatland.utils.rendertools import RenderTool

    from regimes import COMMIT_CELLS

    from stable_baselines3 import A2C, DQN, PPO

    s2 = args.scenario != "network-simple-compact"
    nb = nb_harness.load(scenario2=s2)
    priority_obs.install_priority_features(nb)
    cls = nb["GymTreeObsForRailEnv"]
    original_init = cls.__init__

    def patched(self, *a, **kw):
        kw.setdefault("observation_radius", 50)
        original_init(self, *a, **kw)

    cls.__init__ = patched

    env = nb["make_env"](scenario=args.scenario, condition="normal", seed=args.seed)
    state = regimes.install(env, args.scenario, args.incidents)
    handles = env.get_agent_handles()
    T = int(env._max_episode_steps)

    if args.policy == "forward":
        from flatland.envs.rail_env_action import RailEnvActions

        model = None

        def policy(raw):
            return {h: int(RailEnvActions.MOVE_FORWARD.value) for h in handles}
    else:
        if not args.model:
            raise SystemExit("--model is required unless --policy forward")
        model = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo].load(args.model, device="cpu")

        def policy(raw):
            obs = raw.obs_builder.get_many(list(range(raw.get_num_agents())))
            batch = np.stack([np.asarray(obs[h], dtype=np.float64) for h in handles])
            chosen = model.predict(batch, deterministic=True)[0]
            return {h: int(a) for h, a in zip(handles, chosen)}

    env.reset(random_seed=args.seed)
    height = max(120, int(args.width * env.height / max(env.width, 1)))
    renderer = RenderTool(env, gl="PILSVG", screen_width=args.width, screen_height=height)
    cell_px = int(renderer.renderer.gl.nPixCell)

    frames, reached = [], {h: False for h in handles}
    switches = []          # (row, column, expiry_step, handle)
    prev_row = {}
    try:
        label_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        label_font = ImageFont.load_default()

    def grab(step: int, caption: str) -> None:
        img = renderer.render_env(show=False, show_agents=True,
                                  show_observations=False, return_image=True)
        im = Image.fromarray(img)
        first = 0
        if args.crop_cols:
            first, last = (int(x) for x in args.crop_cols.split(","))
            x0, x1 = first * cell_px, min(im.width, last * cell_px)
            im = im.crop((max(0, x0), 0, min(im.width, x1), im.height))
        if args.zoom != 1.0:
            im = im.resize((int(im.width * args.zoom), int(im.height * args.zoom)),
                           Image.LANCZOS)
        draw = ImageDraw.Draw(im, "RGBA")
        px = cell_px * args.zoom
        # where a train can still change track (diagnostic overlay, off by default)
        for (row, col) in (COMMIT_CELLS if args.markers == "all" else ()):
            if col < first or col * px - first * px > im.width:
                continue
            x, y = (col - first) * px, row * px
            if 0 <= y <= im.height - px:
                draw.rectangle([x + 1, y + 1, x + px - 1, y + px - 1],
                               outline=(40, 90, 220, 255), width=max(2, int(px * 0.06)))
        now = int(env._elapsed_steps)
        # green: this cell is where a train changed track just now (off by default)
        for (srow, scol, expiry, handle) in (switches if args.markers == "all" else ()):
            if expiry < now or scol < first:
                continue
            x, y = (scol - first) * px, srow * px
            if not (0 <= y <= im.height - px):
                continue
            draw.rectangle([x, y, x + px, y + px], fill=(0, 200, 70, 110))
            draw.rectangle([x + 1, y + 1, x + px - 1, y + px - 1],
                           outline=(0, 150, 40, 255), width=max(2, int(px * 0.10)))
            draw.text((x + 1, y + px - 15), f"S{handle}", fill=(0, 110, 30),
                      font=label_font)
        # the active blockage: big red X, labelled with the release step
        for (row, col), info in state.get("active", {}).items():
            if col < first:
                continue
            x, y = (col - first) * px, row * px
            if not (0 <= y <= im.height - px):
                continue
            width = max(3, int(px * 0.16))
            draw.line([(x + 2, y + 2), (x + px - 2, y + px - 2)],
                      fill=(220, 20, 20, 255), width=width)
            draw.line([(x + 2, y + px - 2), (x + px - 2, y + 2)],
                      fill=(220, 20, 20, 255), width=width)
            left = max(0, int(info["release"]) - now)
            draw.text((x, y - 16), f"BLOCKED {left}", fill=(160, 0, 0),
                      font=label_font)
        draw.rectangle([0, 0, im.width, 24], fill=(255, 255, 255))
        draw.text((4, 3), caption, fill=(0, 0, 0), font=label_font)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))

    done = 0
    for step in range(0, T + 1):
        if step % args.stride == 0:
            name = "FORWARD" if args.policy == "forward" else args.algo.upper()
            grab(step, f"{name} step {step}  done {done}/{len(handles)}  "
                       f"fired {len(state['fired'])}  switches {len(switches)}")
        if step == T:
            break
        _, _, dones, _ = env.step(policy(env))
        done = sum(1 for h in handles if env.agents[h].state == TrainState.DONE)
        now = int(env._elapsed_steps)
        for h in handles:
            cfg = env.agents[h].current_configuration
            if cfg is None or cfg[0] is None:
                continue
            row, col = int(cfg[0][0]), int(cfg[0][1])
            if h in prev_row and prev_row[h] != row:
                switches.append((row, col, now + args.switch_hold, h))
            prev_row[h] = row
        switches = [item for item in switches if item[2] >= now]
        if dones.get("__all__"):
            name = "FORWARD" if args.policy == "forward" else args.algo.upper()
            grab(step + 1, f"{name} step {step + 1}  done {done}/{len(handles)}")
            break

    renderer.close_window()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(args.fps, 1)), loop=0, optimize=True)
    size_mb = out.stat().st_size / 1e6
    print(f"{out}: {len(frames)} frames, {size_mb:.2f} MB, "
          f"done {done}/{len(handles)}, fired {len(state['fired'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
