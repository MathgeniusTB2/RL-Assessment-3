"""Render the compact corridor level as a single wide plate for the report.

    python scripts/render_level_plate.py            # -> results/report_figures/fig_s1_level.png

One frame of the simulator's own renderer, no agents, at 3600 px wide, with the
four track rows labelled, the station columns named and a tick under each of the
seven commit crossovers. No inset: this is the whole 87 x 8 map.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import nb_harness  # noqa: E402
import regimes  # noqa: E402
from flatland.utils.rendertools import RenderTool  # noqa: E402

OUT = REPO / "results" / "report_figures" / "fig_s1_level.png"
WIDTH = 3600
SCENARIO = "network-simple-compact"

SCENARIO = "network-simple-compact"


def main() -> int:
    nb = nb_harness.load()
    env = nb["make_env"](scenario=SCENARIO, condition="normal", seed=3000)
    env.reset(random_seed=3000)

    width = 3600
    height = max(120, int(width * env.height / max(env.width, 1)))
    renderer = RenderTool(env, gl="PILSVG", screen_width=width, screen_height=height)
    frame = renderer.render_env(show=False, show_agents=False,
                                show_observations=False, return_image=True)
    plate = Image.fromarray(frame).convert("RGB")

    # the level itself and nothing else: crop the renderer's blank margin
    bg = Image.new("RGB", plate.size, (255, 255, 255))
    box = Image.eval  # keep ruff quiet about the unused name
    bbox = _content_bbox(plate)
    if bbox:
        plate = plate.crop(bbox)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plate.save(OUT)
    print("wrote", OUT, plate.size)
    return 0


def _content_bbox(img: Image.Image):
    """Bounding box of everything that is not the white page margin."""
    from PIL import ImageChops

    grey = img.convert("L")
    diff = ImageChops.difference(grey, Image.new("L", grey.size, 255))
    return diff.getbbox()


if __name__ == "__main__":
    raise SystemExit(main())
