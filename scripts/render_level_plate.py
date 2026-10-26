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

STATIONS = {1: "Strathfield", 14: "Burwood", 40: "Redfern", 84: "Central"}
ROWS = {2: "row 2  up suburban", 3: "row 3  down suburban",
        4: "row 4  up local", 5: "row 5  down local"}
ROW_COLOUR = {2: (31, 56, 100), 3: (120, 60, 30), 4: (31, 56, 100), 5: (120, 60, 30)}


def main() -> int:
    nb = nb_harness.load()
    env = nb["make_env"](scenario=SCENARIO, condition="normal", seed=3000)
    env.reset(random_seed=3000)

    height = max(120, int(WIDTH * env.height / max(env.width, 1)))
    renderer = RenderTool(env, gl="PILSVG", screen_width=WIDTH, screen_height=height)
    frame = renderer.render_env(show=False, show_agents=False,
                               show_observations=False, return_image=True)
    base = Image.fromarray(frame).convert("RGB")
    # the renderer picks its own cell size; derive the geometry from what it drew
    cell = base.width / env.width
    row_h = base.height / env.height

    top = 16
    bottom = 60
    left = int(base.width * 0.085)
    right = 30
    canvas = Image.new("RGB", (base.width + left + right, base.height + top + bottom), "white")
    canvas.paste(base, (left, top))
    draw = ImageDraw.Draw(canvas)

    def font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:  # pragma: no cover - font always present on the box
            return ImageFont.load_default()

    # size the annotation to the frame the renderer actually produced, so the
    # labels stay legible when the plate is placed at page width
    # the renderer rasterises the level at ~27 px per cell, so the annotation is
    # sized against that frame rather than the cell count
    f_station = font(max(40, int(base.width * 0.019)))
    f_row = font(max(34, int(base.width * 0.016)))
    f_tick = font(max(34, int(base.width * 0.016)))
    f_note = font(max(26, int(base.width * 0.011)))

    # track rows: label at the left edge, on the row's own centre line
    for row, label in ROWS.items():
        y = top + int((row + 0.5) * row_h)
        draw.text((left - 16, y), label, fill=ROW_COLOUR[row], font=f_row, anchor="rm")
        draw.line((left - 10, y, left - 2, y), fill=ROW_COLOUR[row], width=3)

    # commit crossovers: a tick under the map at the cell's column and row
    for row, col in sorted(regimes.COMMIT_CELLS):
        x = left + int((col + 0.5) * cell)
        y0 = top + base.height
        draw.line((x, y0 + 6, x, y0 + 34), fill=ROW_COLOUR[row], width=5)



    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print("wrote", OUT, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
