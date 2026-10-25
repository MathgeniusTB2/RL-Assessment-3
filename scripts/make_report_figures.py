"""Regenerate every figure used by the report into results/report_figures/.

Inputs are all shipped: the six GIFs in results/figures/, the paired CSV, the three
training curves and the junction frame. The two map plates and the workflow diagram
are static assets committed alongside (maps extracted from the previous report,
workflow drawn by this script).

    python scripts/make_report_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "results" / "figures"
OUT = REPO / "results" / "report_figures"
NAVY, FILL, EDGE = "#1f3864", "#eef3fa", "#8ea9c9"

OUT.mkdir(parents=True, exist_ok=True)
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
except Exception:
    FONT = ImageFont.load_default()


def frame(name: str, index: int) -> Image.Image:
    im = Image.open(FIGDIR / name)
    im.seek(min(index, im.n_frames - 1))
    return im.convert("RGB")


# ------------------------------------------------------------------ workflow
def workflow() -> None:
    steps = [
        (0.855, 0.105, "TfNSW GTFS static feed", "1.47 GB: trips, stop_times, shapes"),
        (0.735, 0.105, "Pre-processing", "Thursday 15 Oct 2026, 16:00-19:00\n"
                                        "link traversal from shapes (<= 250 m)"),
        (0.605, 0.115, "Scenario build", "corridor 87 x 8, 52 movements\n"
                                        "junction 20 x 20, 126 movements"),
        (0.445, 0.155, "MDP", "observation: tree (depth 3, radius 50)\n"
                              "+ 12 conflict/geometry features (1032-d)\n"
                              "action: Discrete(5) per train\n"
                              "reward: metric potential r(t) = R(t) - R(t-1)\n"
                              "transition: Flatland 4.3 + 2 x 400-step exogenous blockages"),
        (0.315, 0.105, "Vectorised environment", "2 copies, one slot per agent (52 / 126)\n"
                                                "shared policy parameters"),
        (0.185, 0.115, "Training and checkpoint selection", "PPO warm start (BC + critic warm-up)\n"
                                                           "A2C, DQN per agent\n"
                                                           "best of 2-4 unseen seeds by (completion, score)"),
        (0.055, 0.105, "Paired evaluation and reporting", "seeds 3000-3029 vs baseline, exact sign test\n"
                                                          "score, delay_all, completion, crossings"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for y, h, title, text in steps:
        ax.add_patch(FancyBboxPatch((0.06, y), 0.88, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                    linewidth=1.0, facecolor=FILL, edgecolor=EDGE))
        title_y = y + h - 0.035 if h > 0.13 else y + h * 0.68
        body_y = y + h * 0.36 if h > 0.13 else y + h * 0.30
        ax.text(0.5, title_y, title, ha="center", va="center", fontsize=9.5,
                color=NAVY, fontweight="bold")
        ax.text(0.5, body_y, text, ha="center", va="center", fontsize=8.5)
    for i in range(len(steps) - 1):
        ax.annotate("", xy=(0.5, steps[i + 1][0] + steps[i + 1][1] + 0.004),
                    xytext=(0.5, steps[i][0]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#404040"))
    ax.annotate("", xy=(0.945, 0.52), xytext=(0.945, 0.30),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#bf9000",
                                linestyle=(0, (4, 3))))
    ax.text(0.955, 0.41, "design\nchanges", rotation=90, fontsize=7.5, color="#bf9000",
            va="center")
    ax.set_title("End-to-end workflow: each stage consumes the previous stage's artefact",
                 fontsize=10, color=NAVY, pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ curves
def curves() -> None:
    ppo = pd.read_csv(REPO / "results" / "curves" / "ppo_corridor_block.csv")
    series = [("PPO (warm start)", ppo.timesteps / 1e6, ppo.score, ppo.completion, "#1f4e79")]
    for label, name, colour in (("A2C", "a2c_corridor_block.csv", "#c55a11"),
                                ("DQN", "dqn_corridor_block.csv", "#7f7f7f")):
        df = pd.read_csv(REPO / "results" / "curves" / name)
        series.append((label, df.timesteps / 1e6, df.score, df.completion, colour))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2.1, 1]})
    for label, x, score, completion, colour in series:
        ax1.plot(x, score, "o-", color=colour, label=label)
        ax2.plot(x, completion, "o-", color=colour)
    ax1.axhline(-0.0662, ls="--", lw=1, color="black")
    ax1.text(0.02, -0.0662, " forward-only (30-seed mean)", va="bottom", fontsize=7.5)
    ax1.set_ylabel("normalised score")
    ax1.set_ylim(-0.62, 0.02)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("Corridor arms on the block regime (internal evaluation, 2 or 4 seeds)")
    ax2.set_ylabel("completion")
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel("training steps (millions)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curve.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ corridor arms
def corridor_arms() -> None:
    """2 x 2 grid, each panel cropped to the stretch where the blockages land."""
    arms = [
        ("forward-only   -0.082   1.000", "corridor_forward_block.gif", 232),
        ("PPO   -0.001   1.000  (crosses)", "corridor_ppo_block.gif", 232),
        ("A2C   -0.082   1.000  (no crossing)", "corridor_a2c_block.gif", 232),
        ("DQN   -0.377   0.500", "corridor_dqn_block.gif", 300),
    ]
    panels = []
    for label, name, idx in arms:
        f = frame(name, idx)
        cell = f.width / 87
        f = f.crop((int(cell * 13), 0, int(cell * 41), f.height))
        panels.append((label, f.resize((660, max(1, int(f.height * 660 / f.width))), Image.LANCZOS)))
    w, h = 660, max(f.height for _, f in panels)
    sheet = Image.new("RGB", (2 * w + 30, 2 * (h + 56) + 16), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    for i, (label, f) in enumerate(panels):
        x = 8 + (i % 2) * (w + 14)
        y = 8 + (i // 2) * (h + 56)
        draw.text((x, y), label, fill=(20, 20, 60), font=FONT)
        sheet.paste(f, (x, y + 46))
    sheet.save(OUT / "fig_corridor_arms.gif.png" if False else OUT / "fig_corridor_arms.png")


# ------------------------------------------------------------------ junction pair
def junction_pair() -> None:
    panels = []
    for label, name, idx in (
        ("route follower   -0.182   126/126", "junction_route_follower.gif", 156),
        ("PPO   -0.240   0.468  (jams)", "junction_ppo.gif", 156),
    ):
        f = frame(name, idx)
        panels.append((label, f.resize((600, int(f.height * 600 / f.width)), Image.LANCZOS)))
    w, h = 600, panels[0][1].height
    sheet = Image.new("RGB", (2 * w + 30, h + 60), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    for i, (label, f) in enumerate(panels):
        x = 8 + i * (w + 14)
        draw.text((x, 8), label, fill=(20, 20, 60), font=FONT)
        sheet.paste(f, (x, 52))
    sheet.save(OUT / "fig_junction_pair.png")


# ------------------------------------------------------------------ paired delta
def delta() -> None:
    d = pd.read_csv(REPO / "results" / "pair_block_ppo.csv")
    g = d[d.policy == "greedy"].set_index("seed")
    p = d[d.policy == "ppo"].set_index("seed")
    diff = (p.score - g.score).dropna()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.hist(diff, bins=12, color="#2e75b6", edgecolor="white")
    ax.axvline(diff.mean(), color="#c00000", lw=2, label=f"mean {diff.mean():+.4f}")
    ax.set_xlabel("paired score difference (PPO - forward-only)")
    ax.set_ylabel("episodes (of 30)")
    ax.set_title(f"30 held-out seeds: {int((diff > 0).sum())}/30 wins")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_delta.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ audit
def audit() -> None:
    # from scripts/crossing_audit.py decisions --seeds 3000,3001 (Table 4)
    rows = [("(2,6)", 22, 22), ("(2,25)", 2, 2), ("(4,51)", 18, 18),
            ("(5,22)", 19, 19), ("(5,59)", 40, 20)]
    labels = [r[0] for r in rows]
    rates = [100 * r[2] / r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bars = ax.bar(labels, rates, color="#2e75b6", width=0.62)
    for bar, rate in zip(bars, rates, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 3, f"{rate:.0f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("turns on blocked visits (%)")
    ax.set_xlabel("commit crossover (row, column)")
    ax.set_title("Open visits in the same audit: 202, of which 0 turned")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_audit.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    workflow()
    curves()
    corridor_arms()
    junction_pair()
    delta()
    audit()
    print("figures written to", OUT)
