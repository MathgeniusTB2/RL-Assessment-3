"""Animated GIF: the trained agent racing the no-crossover baseline.

Both policies run the SAME episode (same seed, same disruption regime) in
lockstep, so the only difference on screen is the decision they make.  Frames are
rendered with the project's headless PILSVG renderer and stacked: agent on top,
baseline underneath.

The view is cropped to the stretch of line where the failures actually fire, so
the blockage and the queue behind it are legible on an 87-wide corridor.

    python scripts/render_race_gif.py --model /tmp/opencode/race/best_model.zip \
        --seed 2002 --stride 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import regimes  # noqa: E402

RED = (210, 0, 0)
ORANGE = (235, 130, 0)
TOP = (240, 244, 248)


def cell(agent):
    cfg = agent.current_configuration
    if cfg is None or cfg[0] is None:
        return None
    return int(cfg[0][0]), int(cfg[0][1])


_NB = {}
_PREPARED = set()


def prepare_nb(scen: str, radius: int | None, priority: bool):
    """Load the notebook namespace once and apply the obs patches the checkpoint
    was trained with (priority features + observation radius)."""
    import nb_harness

    key = "s2" if regimes.is_s2(scen) else "s1"
    if key not in _NB:
        _NB[key] = nb_harness.load(scenario2=(key == "s2"))
    nb = _NB[key]
    if key in _PREPARED:
        return nb
    if priority:
        import priority_obs

        priority_obs.install_priority_features(nb)
    if radius is not None:
        cls = nb["GymTreeObsForRailEnv"]
        original = cls.__init__

        def patched(self, *a, **kw):
            kw.setdefault("observation_radius", radius)
            original(self, *a, **kw)

        cls.__init__ = patched
    _PREPARED.add(key)
    return nb


def build(scen: str, seed: int, incidents: int, seed_plan: int | None = None):
    """Fresh env + regime. seed_plan overrides the disruption plan seed."""
    nb = _NB["s2" if regimes.is_s2(scen) else "s1"]
    parallel = nb["make_parallel_env"](scenario=scen, condition="normal", seed=seed)
    raw = parallel._wrap
    state = (regimes.install(raw, scen, incidents) if incidents > 0
             else {"fired": [], "fire_cells": None})
    parallel.reset(seed=seed if seed_plan is None else seed_plan)
    return parallel, raw, list(parallel.possible_agents), state, nb


def act(policy, raw, obs, handles, nb, model):
    if policy == "agent":
        batch = np.stack([obs[h] for h in handles])
        chosen = model.predict(batch, deterministic=True)[0]
        return {h: int(a) for h, a in zip(handles, chosen)}
    if policy == "forward":
        return {h: 2 for h in handles}
    plan = nb["greedy_actions"](raw)
    return {h: int(plan.get(h, 0)) for h in handles}


def late_count(raw, handles):
    from flatland.envs.step_utils.states import TrainState

    n = 0
    for h in handles:
        a = raw.agents[h]
        if a.state in (TrainState.DONE, TrainState.WAITING):
            continue
        sched = a.waypoints_latest_arrival[-1]
        if sched is not None and raw._elapsed_steps > sched:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="network-simple-compact")
    ap.add_argument("--seed", type=int, default=2002)
    ap.add_argument("--incidents", type=int, default=3,
                    help="0 = no delays at all (normal condition)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline", default="forward", choices=["forward", "notebook"])
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--duration", type=int, default=90,
                    help="ms per frame in the GIF (higher = slower)")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--height", type=int, default=320,
                    help="renderer screen height; sets the cell size")
    ap.add_argument("--pad", type=int, default=8, help="cells either side of the failures")
    ap.add_argument("--observation-radius", type=int, default=50)
    ap.add_argument("--no-priority", action="store_true")
    ap.add_argument("--out", default=str(REPO / "docs" / "race_compact.gif"))
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from flatland.envs.step_utils.states import TrainState
    from flatland.utils.rendertools import RenderTool

    model = PPO.load(args.model, device="cpu")
    prepare_nb(args.scenario, args.observation_radius, not args.no_priority)
    _nb = _NB["s2" if regimes.is_s2(args.scenario) else "s1"]
    _probe = _nb["make_env"](scenario=args.scenario, condition="normal", seed=0)
    _obs, _ = _probe.reset(random_seed=0)
    _k = next(iter(_obs))
    print(f"agents {len(_obs)}, per-agent obs {_obs[_k].shape}")

    # ---- lockstep rollouts with rendering
    envs = {}
    for pol in ("agent", args.baseline):
        parallel, raw, handles, state, nb = build(args.scenario, args.seed, args.incidents)
        renderer = RenderTool(raw, gl="PILSVG", screen_width=args.width,
                              screen_height=max(args.height,
                                                int(args.width * raw.height / raw.width)))
        envs[pol] = dict(parallel=parallel, raw=raw, handles=handles, state=state, nb=nb,
                         renderer=renderer, cum={h: 0.0 for h in handles},
                         done={})

    # crop to the cells this episode's failures were scheduled on (captured
    # before any of them fire, because firing removes the cell from the set)
    if regimes.is_s2(args.scenario):
        # the junction needs the whole 20x20 map: cropping to the fire band would
        # cut off Harris Park, Granville and the Merrylands branch
        cols = [0, envs["agent"]["raw"].width - 1]
        args.pad = 0
    else:
        planned = set(envs["agent"]["state"]["fire_cells"] or ())
        cols = [c for (_r, c) in planned] or [0, 40]
    lo = max(0, min(cols) - args.pad)
    hi = min(envs["agent"]["raw"].width, max(cols) + args.pad)

    frames = []
    font = ImageFont.load_default()
    T = int(envs["agent"]["raw"]._max_episode_steps)

    obs = {}
    for pol, e in envs.items():
        obs[pol], _ = e["parallel"].reset(seed=args.seed)

    alive = {pol: True for pol in envs}
    for step in range(1, T + 1):
        for pol, e in envs.items():
            if not alive[pol]:
                continue
            acts = act(pol, e["raw"], obs[pol], e["handles"], e["nb"], model)
            obs[pol], rew, terms, _, _ = e["parallel"].step(acts)
            for h, r in rew.items():
                if r is not None:
                    e["cum"][h] += float(r)
            for h in e["handles"]:
                if e["raw"].agents[h].state == TrainState.DONE:
                    e["done"].setdefault(h, step)
            if (terms or {}).get("__all__") or len(e["done"]) == len(e["handles"]):
                alive[pol] = False
        if not any(alive.values()):
            break

        if step % args.stride == 0 or step == T:
            panels = []
            for pol, e in envs.items():
                arr = e["renderer"].render_env(show=False, show_agents=True,
                                               show_observations=False, return_image=True)
                S = int(e["renderer"].renderer.gl.nPixCell)
                img = Image.fromarray(arr).convert("RGB").crop((lo * S, 0, hi * S, arr.shape[0]))
                d = ImageDraw.Draw(img, "RGBA")
                for a in e["raw"].agents:
                    c = cell(a)
                    if c is None or a.state in (TrainState.DONE, TrainState.WAITING):
                        continue
                    x = (c[1] - lo) * S
                    y = c[0] * S
                    if a.malfunction_handler.malfunction_down_counter > 0:
                        d.rectangle([x, y, x + S, y + S], fill=RED + (80,),
                                    outline=RED + (255,), width=2)
                    elif a.state == TrainState.STOPPED:
                        d.rectangle([x + 2, y + 2, x + S - 2, y + S - 2],
                                    outline=ORANGE + (230,), width=1)
                head = 16
                strip = Image.new("RGB", (img.width, img.height + head), TOP)
                strip.paste(img, (0, head))
                dd = ImageDraw.Draw(strip)
                done = len(e["done"])
                label = ("AGENT (PPO)" if pol == "agent" else f"BASELINE ({args.baseline})")
                dd.text((6, 3), f"{label}   step {step}   finished {done}/{len(e['handles'])}   "
                                f"late now {late_count(e['raw'], e['handles'])}",
                        fill=(20, 20, 20), font=font)
                panels.append(strip)
            sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels) + 4), (255, 255, 255))
            y = 0
            for p in panels:
                sheet.paste(p, (0, y))
                y += p.height + 4
            frames.append(sheet)

        if all(len(e["done"]) == len(e["handles"]) for e in envs.values()):
            break

    for e in envs.values():
        e["renderer"].close_window()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.duration, loop=0, optimize=True)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(frames)} frames, "
          f"{frames[0].width}x{frames[0].height})")

    for pol, e in envs.items():
        score = float(e["raw"].rewards.normalize(*[e["cum"][h] for h in e["handles"]],
                                                 num_agents=len(e["handles"]),
                                                 max_episode_steps=T)) - 1.0
        print(f"  {pol:<8} finished {len(e['done'])}/{len(e['handles'])}  score {score:+.4f}")


if __name__ == "__main__":
    main()
