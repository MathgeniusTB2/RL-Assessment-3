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
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", default="network-simple-compact")
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--incidents", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=12, help="render every Nth step")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=72)
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    import nb_harness
    import priority_obs
    import regimes

    from flatland.envs.step_utils.states import TrainState
    from flatland.utils.rendertools import RenderTool

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

    model = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[args.algo].load(args.model, device="cpu")

    def policy(raw):
        obs = raw.obs_builder.get_many(list(range(raw.get_num_agents())))
        batch = np.stack([np.asarray(obs[h], dtype=np.float64) for h in handles])
        chosen = model.predict(batch, deterministic=True)[0]
        return {h: int(a) for h, a in zip(handles, chosen)}

    env.reset(random_seed=args.seed)
    height = max(120, int(args.width * env.height / max(env.width, 1)))
    renderer = RenderTool(env, gl="PILSVG", screen_width=args.width, screen_height=height)

    frames, reached = [], {h: False for h in handles}
    label_font = ImageFont.load_default()

    def grab(step: int, caption: str) -> None:
        img = renderer.render_env(show=False, show_agents=True,
                                  show_observations=False, return_image=True)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        draw.rectangle([0, 0, im.width, 16], fill=(255, 255, 255))
        draw.text((4, 2), caption, fill=(0, 0, 0), font=label_font)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))

    done = 0
    for step in range(0, T + 1):
        if step % args.stride == 0:
            grab(step, f"{args.algo.upper()} step {step}  done {done}/{len(handles)}  "
                       f"fired {len(state['fired'])}")
        if step == T:
            break
        _, _, dones, _ = env.step(policy(env))
        done = sum(1 for h in handles if env.agents[h].state == TrainState.DONE)
        if dones.get("__all__"):
            grab(step + 1, f"{args.algo.upper()} step {step + 1}  done {done}/{len(handles)}")
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
