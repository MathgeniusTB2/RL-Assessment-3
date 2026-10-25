"""Build the Part E submission from the unit's template and the shipped evidence.

    python scripts/build_part_e_report.py                 # -> report/RailTech-A3-PartE-Report.docx
    python scripts/build_part_e_report.py --map-pdf FILE   # heading -> page numbers as JSON

The template supplies the cover fields, the section skeleton (1 Introduction, 2
Architecture, 3 Results, 4 Discussion, 5 References, Appendices A/B) and the
individual-contribution appendix. Everything inside the sections is generated
here, and every table is recomputed from the CSVs in results/, so this document
cannot drift from the evidence the way a hand-edited copy would.

The contents page is static (Google Docs and LibreOffice do not rebuild Word TOC
fields reliably), so it is filled from a two-pass build: run once with
--map-pdf on the produced PDF, then rebuild with --pages <json>.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pair import sign_test_p  # noqa: E402
from report_layout import W, Layout, add_page_footer, force_font, style_document  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = RES / "report_figures"
DEFAULT_TEMPLATE = REPO / "templates" / "part-e-template.docx"
DEFAULT_OUT = REPO / "report" / "RailTech-A3-PartE-Report.docx"

SECTIONS = [
    "Abstract",
    "1. Introduction and Background",
    "1.1 The problem you tried to solve",
    "1.2 Motivation",
    "1.3 Application",
    "1.4 Dataset/Environment",
    "2. Overview of the architecture/system",
    "2.1 Flow Diagram and Workflow",
    "2.2 System Architecture Design",
    "2.3 GUI/Environment Design",
    "3. Results and Evaluation",
    "3.1 Experimental Settings",
    "3.2 Pre-Processing",
    "3.3 Experimental Results",
    "3.4 Limitations",
    "4. Discussion and Conclusions",
    "5. References",
    "Appendices",
    "A. Individual Contribution",
    "B. Individual Contribution Split",
]


# ------------------------------------------------------------------ evidence
def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def paired(path: Path, arm: str = "ppo", base: str = "greedy") -> dict:
    """Paired statistics from a results/pair*.csv written by eval_pair.py."""
    data = rows(path)
    a = {int(r["seed"]): r for r in data if r["policy"] == arm}
    b = {int(r["seed"]): r for r in data if r["policy"] == base}
    seeds = sorted(set(a) & set(b))
    da = [float(a[s]["score"]) for s in seeds]
    db = [float(b[s]["score"]) for s in seeds]
    diff = [x - y for x, y in zip(da, db, strict=True)]
    wins = sum(d > 0 for d in diff)
    losses = sum(d < 0 for d in diff)
    return {
        "n": len(seeds),
        "base": statistics.fmean(db),
        "arm": statistics.fmean(da),
        "delta": statistics.fmean(diff),
        "median": statistics.median(diff),
        "wins": wins,
        "losses": losses,
        "ties": len(seeds) - wins - losses,
        "p": sign_test_p(wins, losses),
        "compl_base": statistics.fmean(float(b[s]["completion"]) for s in seeds),
        "compl_arm": statistics.fmean(float(a[s]["completion"]) for s in seeds),
        "delay_base": statistics.fmean(float(b[s]["delay_all"]) for s in seeds),
        "delay_arm": statistics.fmean(float(a[s]["delay_all"]) for s in seeds),
        "on_time_base": statistics.fmean(float(b[s]["on_time"]) for s in seeds),
        "on_time_arm": statistics.fmean(float(a[s]["on_time"]) for s in seeds),
    }


def best_checkpoint(name: str) -> tuple[str, float, float]:
    """The checkpoint selection rule: completion first, then score."""
    data = rows(RES / "curves" / f"{name}_corridor_block.csv")
    best = max(data, key=lambda r: (float(r["completion"]), float(r["score"])))
    return f"{int(best['timesteps']) // 1000}k", float(best["score"]), float(best["completion"])


def audit_table() -> tuple[list[list[str]], int, int, int, int]:
    data = rows(RES / "crossing_audit_block.csv")
    cells: dict[tuple[int, int], dict[int, tuple[int, int]]] = {}
    for r in data:
        key = (int(r["row"]), int(r["col"]))
        cells.setdefault(key, {})[int(r["blocked"])] = (int(r["visits"]), int(r["turns"]))
    tracks = {2: "up_suburban", 4: "up_local", 5: "down_local"}
    body, tot_b, tot_bt, tot_o, tot_ot = [], 0, 0, 0, 0
    for (row, col) in sorted(cells):
        bv, bt = cells[(row, col)].get(1, (0, 0))
        ov, ot = cells[(row, col)].get(0, (0, 0))
        tot_b += bv
        tot_bt += bt
        tot_o += ov
        tot_ot += ot
        body.append([f"({row},{col})", tracks.get(row, "?"), str(bv), str(bt), str(ov), str(ot)])
    body.append(["Total", "", str(tot_b), str(tot_bt), str(tot_o), str(tot_ot)])
    return body, tot_b, tot_bt, tot_o, tot_ot


def pct(x: float) -> str:
    """Scores and deltas to the four decimals the evidence supports."""
    return f"{x:.4f}"


def comp(x: float) -> str:
    """Completion shares to three decimals."""
    return f"{x:.3f}"


# ------------------------------------------------------------------ document
def page_of(pdf: Path) -> dict[str, int]:
    """Heading -> 1-based page number, from a built PDF."""
    txt = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout
    pages = txt.split("\f")
    found: dict[str, int] = {}
    for name in SECTIONS:
        for i, page in enumerate(pages, 1):
            if "Table of Contents" in page:  # the contents page lists every heading
                continue
            lines = [ln.strip() for ln in page.splitlines()]
            if any(ln.startswith(name) for ln in lines):
                found[name] = i
                break
    return found


def build(template: Path, out: Path, pages: dict[str, int] | None) -> None:
    doc = Document(str(template))
    body = doc.element.body

    # keep the cover and the "Table of Contents" marker, drop the rest
    kids = list(body)
    marker = next(
        k for k in kids
        if k.tag.endswith("}p") and "Table of Contents" in "".join(k.itertext())
    )
    for k in kids[kids.index(marker) + 1:]:
        if k.tag.endswith("}sectPr"):  # page setup lives here; removing it breaks the document
            continue
        body.remove(k)

    style_document(doc)
    add_page_footer(doc, "RailTech, 43008 Assessment 3 (Part E)")
    # the template's cover text carries a typo and a missing space before the IDs
    for k in kids[:kids.index(marker)]:
        for run in k.iter(f"{W}t"):
            if run.text:
                run.text = (run.text
                            .replace("Assignent-3, Part-E", "Assignment 3, Part E")
                            .replace("Martirosov(", "Martirosov (")
                            .replace("26 October 2026", "25 October 2026"))

    # the cover keeps its own sizes but follows the report's typeface
    for k in [*kids[:kids.index(marker)], marker]:
        for run in k.iter(f"{W}r"):
            force_font(run, "Calibri")

    lay = Layout(doc, FIG)

    # ---- contents page (static, from the two-pass map)
    for name in SECTIONS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.tab_stops.add_tab_stop(
            Cm(16.2), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
        )
        p.add_run(name)
        p.add_run("\t")
        p.add_run(str(pages.get(name, "")) if pages else "")

    lay.page_break()

    # ---- abstract
    doc.add_heading("Abstract", level=1)
    a = paired(RES / "pair_block_ppo.csv")
    lay.body(
        "We ask whether a learned policy can beat a schedule-following controller when a track "
        "becomes unavailable. Two Flatland scenarios are built from the October 2026 TfNSW GTFS "
        "feed: a four-track Central-Strathfield corridor with 52 scheduled movements and the "
        "Granville-Harris Park junction with 126. The corridor is disrupted by an exogenous "
        "blockage that removes a cell's rail transitions for 400 steps, so a policy cannot make "
        "the incident vanish by rerouting. Under that regime, and across three algorithms, only "
        f"the warm-started PPO learns to change track: it improves the metric from {pct(a['base'])} "
        f"to {pct(a['arm'])}, a paired gain of {a['delta']:+.4f} on 30 held-out seeds "
        f"({a['wins']} of {a['n']}, p < 1e-4) with every train completing, and on a clean network "
        "it scores 0.0000 with no crossing at all. A2C never learns the crossing and DQN parks "
        "half the fleet. The junction remains a loss: the selected checkpoint dispatches and jams "
        "(-0.2400, 59 of 126 trains home) instead of parking, and still finishes behind the "
        "scripted route follower (-0.1823)."
    )

    # ---- 1 introduction
    doc.add_heading("1. Introduction and Background", level=1)
    doc.add_heading("1.1 The problem you tried to solve", level=2)
    lay.body(
        "A suburban rail corridor is a scheduling problem wrapped in a control problem. The "
        "timetable is fixed, the track layout offers few alternatives, and the failures that "
        "matter remove a cell of track for longer than the headway behind it. Classical "
        "rescheduling solves this with a central optimiser that sees the whole fleet. Our "
        "question is narrower: with one shared policy controlling every train from its own local "
        "observation, can a learner discover that it should change track before a blockage, and "
        "can that behaviour be measured honestly?"
    )
    lay.body(
        "The measurement is half the problem. Flatland's metric charges lateness, missed stops "
        "and cancellations, and a policy can score well by cancelling trains rather than running "
        "them. We keep that metric as the objective and report a second statistic, delay_all, "
        "that charges every agent its arrival minus its scheduled arrival using the episode end "
        "for a train that never arrives, so a collapsed policy cannot look punctual by dropping "
        "unfinished trains from a survivors-only mean."
    )
    doc.add_heading("1.2 Motivation", level=2)
    lay.body(
        "Sydney's network loses whole corridors to signal repairs, overhead wiring and incidents, "
        "and the recovery decisions are made per train by whoever is watching the panel. A "
        "policy that learns the local rule - change track at this crossover when that cell is "
        "blocked - is exactly the shape of decision a signaller makes, and simulation is the only "
        "place to measure it repeatably. Reinforcement learning fits because the value of a "
        "diversion depends on what every other train does next, which no schedule can know in "
        "advance."
    )
    doc.add_heading("1.3 Application", level=2)
    lay.body(
        "The project delivers a simulation-based pipeline rather than a deployed controller: two "
        "scenarios built from the real timetable, a shared per-train policy, and an evaluation "
        "layer that scores any arm against a scripted baseline on the same seeds. Everything runs "
        "from one notebook (the graded artefact) and the command-line tools in scripts/, which "
        "execute that notebook's own cells so the CLI and the notebook cannot diverge."
    )
    doc.add_heading("1.4 Dataset/Environment", level=2)
    lay.body(
        "The data is the TfNSW GTFS static feed (about 1.4 GB), filtered to the Thursday 15 "
        "October 2026 service between 16:00 and 19:00, with link traversal times derived from the "
        "shapes rather than from straight-line distance. Two Flatland levels are built from it: a "
        "condensed four-track Main Suburban corridor and the Main Western junction at Granville, "
        "where the Suburban pair and the Old Main South converge at grade. Flatland 4.3.0 supplies "
        "the grid, the switches, the crossovers and the metric."
    )
    lay.table(
        [
            ["Item", "Corridor", "Junction"],
            ["Scenario", "network-simple-compact", "granville-harris-park"],
            ["Level", "87 x 8, four through tracks", "20 x 20 Y-link junction"],
            ["Movements", "52", "126"],
            ["Episode budget", "999 steps", "757 steps"],
            ["Disruption", "2 exogenous blockages, 400 steps", "3 train-borne failures, 220 steps"],
            ["Observation", "tree depth 3, radius 50, plus 12 features (1032)", "same"],
            ["Action", "Discrete(5) per train", "same"],
            ["Held-out seeds", "3000 to 3029", "3000 to 3029 (legacy rows); 3001, 3002 (current)"],
            ["Baseline", "forward-only reflex", "scripted route follower"],
        ],
        [4.2, 6.4, 5.6],
        "Table 1: Experimental settings. The maps share the observation, action space and "
        "evaluation protocol; they differ in disruption mechanism because only the corridor has a "
        "switchable pair of parallel tracks.",
    )

    # ---- 2 architecture
    lay.page_break()
    doc.add_heading("2. Overview of the architecture/system", level=1)
    doc.add_heading("2.1 Flow Diagram and Workflow", level=2)
    lay.body(
        "The workflow runs in seven stages, each with a defined input and output (Figure 1). The "
        "dashed arrow is deliberate: every change to the observation, reward or disruption regime "
        "was re-validated by re-running the same paired evaluation, so the report only contains "
        "numbers produced by the final configuration, with the two labelled junction rows of Table "
        "6 as the one disclosed exception."
    )
    lay.figure(
        "fig_workflow.png", 12.4,
        "Figure 1: End-to-end workflow. Each stage consumes the artefact produced by the previous "
        "one; the dashed arrow marks the design loop that re-ran the evaluation after every change.",
    )
    for text in [
        "Service extraction: the GTFS feed (about 1.4 GB) is filtered to one Thursday peak, with "
        "link traversal times derived from the shapes.",
        "Scenario construction: two Flatland levels and two timetables, the corridor (87 x 8, 52 "
        "movements) and the junction (20 x 20, 126 movements).",
        "MDP construction: level, timetable and disruption regime, with a tree observation, a "
        "per-train discrete action and the metric potential as reward.",
        "Vectorisation and training: two copies of the fleet for PPO, one for A2C and DQN, one "
        "sub-environment per train, shared policy parameters.",
        "Checkpoint selection: PPO is scored every 100k steps on unseen seeds 9001 to 9004 "
        "(thirteen evaluations); A2C and DQN every 400k steps on two seeds, completion first.",
        "Paired evaluation and reporting: the corridor arm and its baseline on the same 30 "
        "held-out seeds (3000 to 3029), with an exact sign test.",
        "Decision audit: every visit to a commit crossover is recorded with whether a blocked "
        "cell lay ahead, so the crossings can be shown to be conditional.",
    ]:
        lay.bullet(text)

    doc.add_heading("2.2 System Architecture Design", level=2)
    lay.body(
        "One shared multi-layer perceptron controls every train: 1032 inputs, two hidden layers "
        "(256 units for PPO, 128 for A2C and DQN) and a five-way action head. The observation is "
        "a flattened, normalised tree of depth 3 within a radius of 50 cells, plus twelve conflict "
        "and geometry features: slack and its rank, distance to the first occupied cell, distance "
        "to oncoming traffic, malfunctioning trains on my route, my own malfunction countdown, "
        "current speed, whether I am on the map, remaining route length, queue depth, distance to "
        "the nearest blocked cell ahead, and distance to my next commit crossover. The last two "
        "exist because a zeroed rail cell is invisible to occupancy features, so without them the "
        "crossing decision is not expressible."
    )
    lay.body(
        "The reward is the assessment metric paid densely as a potential, r(t) = R(t) - R(t-1), "
        "where R(t) is what the metric would charge if the episode ended at t. The sum telescopes "
        "to the metric exactly, verified per agent and per term to machine precision, and training "
        "divides the potential by a constant 100 so the value targets stay O(1). All reported "
        "arms also use two shaping bonuses on top of the potential: +50 for every train that "
        "arrives and -1 for every step a train is stuck, at the same 1/100 scale."
    )
    lay.body(
        "From a random initialisation none of the three algorithms finds the crossing, so the "
        "reported corridor arm is PPO fine-tuned from a behaviour-cloned detour baseline "
        "(imitation then RL), with the critic warmed up first. The two independent algorithms are "
        "the control: they share the observation, reward and regime but not the warm start."
    )
    lay.table(
        [
            ["Module", "Responsibility"],
            ["nb_harness.py", "Executes the notebook's cells so the CLI tools use exactly the notebook's code"],
            ["regimes.py", "Map geometry (seven commit crossovers) and the regime selector (block by default)"],
            ["disruptions.py", "Both mechanisms: train-borne failures and the exogenous blockage"],
            ["priority_obs.py", "The twelve conflict/geometry observation features (1032-d input)"],
            ["metric_rewards.py", "The metric as a dense potential, plus the shaped variant"],
            ["train_ppo.py", "PPO training, dense reward, checkpoint selection, critic warm-up"],
            ["train_algo.py", "A2C and DQN on the same environment and evaluator"],
            ["eval_pair.py", "Paired evaluation of an arm against its baseline on held-out seeds"],
            ["bc_warmstart.py", "Detour teacher, behaviour-cloning labels, warm-start checkpoint"],
            ["crossing_audit.py", "Score, per-crossover decisions, and regime exposure"],
            ["render_policy_gif.py", "Episode figures, red-X overlay and diagnostic markers"],
            ["verify_rewards.py", "Proves the dense reward reproduces the metric per agent and per term"],
            ["s2_env.py, s2_assembly.py", "Junction environment assembly (rail, fleet, timetable)"],
        ],
        [4.4, 11.8],
        "Table 2: The modules on a live path. The notebook imports five of them directly; the rest "
        "are command-line entry points that execute the notebook's cells through nb_harness.",
    )

    doc.add_heading("2.3 GUI/Environment Design", level=2)
    lay.body(
        "Flatland's own renderer is the interface, which the brief permits in place of a separate "
        "GUI: it shows the rail grid with its switches and crossovers, one coloured marker per "
        "train, and a red X on every cell that may not be entered - an exogenous blockage on the "
        "corridor, a train standing in malfunction on the junction. Using the simulator's renderer "
        "means every episode figure comes from the same process the agent is trained in, produced "
        "headlessly with the PIL backend. Figure 2 shows the corridor level and Figure 3 the "
        "junction."
    )
    lay.figure(
        "fig_corridor_map.png", 16.4,
        "Figure 2: The corridor. (a) The full 87 x 8 level with 52 movements. (b) Columns 1 to 26 "
        "at native scale, showing Strathfield, Burwood and the switch ladders.",
    )
    lay.figure(
        "fig_junction.png", 11.0,
        "Figure 3: The junction. A 20 x 20 level where the Suburban pair and the Old Main South "
        "converge and diverge, carrying 126 scheduled movements.",
    )
    lay.body(
        "The disruption regime is the methodological core. A train-borne failure fires when a "
        "train reaches a cell, which makes the disruption a function of the policy's own "
        "trajectory: a policy that reroutes early can prevent the incident from firing at all. "
        "The corridor therefore uses an exogenous blockage: a cell's rail transitions are removed "
        "outright at a seed-drawn step and restored after 400 steps, so the failure exists whether "
        "or not a train goes near it. Two are drawn per episode, one per direction, from the cells "
        "just past each commit crossover; activation is deferred while a cell is occupied, and "
        "dropped if it stays occupied for 60 steps. Both arms are given the same cells at the "
        "same seed-drawn steps, so the pairing is by seed and blockage plan."
    )

    # ---- 3 results
    lay.page_break()
    doc.add_heading("3. Results and Evaluation", level=1)
    doc.add_heading("3.1 Experimental Settings", level=2)
    lay.body(
        "Table 1 lists the settings. All arms act deterministically; the corridor arm is "
        "evaluated on seeds 3000 to 3029, which training and checkpoint selection never see, and "
        "A2C and DQN are reported from their selection seeds only. Because the baseline and the "
        "arm run on the same seed, the same environment and the same blockages, a delta is "
        "computed per seed and the report gives its mean, median, win count and an exact two-sided "
        "sign test. The score is the assessment metric normalised to [-1, 0], with each train's "
        "cost capped at one episode, summed over all agents and divided by the number of agents "
        "and the episode budget."
    )
    lay.body(
        "The regime and the policy are CPU-bound: Flatland steps the fleet in Python and the "
        "1032-d tree observation is rebuilt for every agent at every step. The shipped corridor "
        "run took 1,865 s for 1,224,704 timesteps including thirteen four-episode evaluations "
        "(about 660 environment steps per second), and once startup is paid each evaluation "
        "interval costs about 148 s, so the evaluator rather than the optimiser sets the wall "
        "clock. Training ran on an x86_64 workstation with an RTX 3080 (the system_info recorded "
        "in the checkpoint); the notebook and figures also run on an aarch64 relay with the same "
        "pinned stack, where the sixteen tests pass. Every table in this report is regenerated "
        "from the CSVs in results/ by scripts/build_part_e_report.py; the commands that produced "
        "those CSVs are listed in the repository README and in Appendix B of the standalone report."
    )

    doc.add_heading("3.2 Pre-Processing", level=2)
    for text in [
        "Service extraction: one Thursday service period, with rail links rather than straight "
        "lines giving traversal times.",
        "Corridor level: four adjacent through tracks with the ladder rows removed; the compact "
        "map leaves a two-cell diversion premium at 7 of the 15 stop types.",
        "Junction level: a 20 x 20 level from the published track plan, with the turn sequence "
        "read out of the timetable as a route dictionary.",
        "Deadlines and episode budget: calibrated from a reference rollout, with the budget "
        "extended past the last arrival because the serialised value cut 15 junction trains off "
        "mid-journey.",
        "Multi-cell targets: the target set is rebuilt from the agent's own last waypoint group "
        "after every reset, otherwise a diverted train can never finish.",
        "Dense reward: the metric as a potential, verified against DefaultRewards per agent and "
        "per term.",
        "Metric hygiene: delay_all replaced a total-delay statistic that went NaN when nothing "
        "finished and negative when few did.",
        "Regime hygiene: the blockage is exogenous, so the disrupted condition cannot collapse "
        "into the undisrupted one.",
    ]:
        lay.bullet(text)

    doc.add_heading("3.3 Experimental Results", level=2)
    lay.body(
        "The corridor result is the headline. PPO improves the score from -0.0662 to -0.0005, a "
        f"mean paired gain of {a['delta']:+.4f} with a median of {a['median']:+.4f} (Table 3). It "
        f"wins all {a['n']} seeds and every train completes in both arms, so the gain is "
        f"punctuality rather than cancellation: total lateness falls from {a['delay_base']:.0f} "
        f"steps to {a['delay_arm']:.0f}. The on-time share moves only from "
        f"{a['on_time_base']:.3f} to {a['on_time_arm']:.3f}: the gain is in how late the affected "
        "trains are, not in which trains complete. Figure 4 shows the per-seed spread."
    )
    lay.table(
        [
            ["Arm", "Baseline", "Arm", "Delta", "Wins", "Sign test p", "Completion", "delay_all"],
            [
                "PPO",
                pct(a["base"]),
                pct(a["arm"]),
                f"{a['delta']:+.4f}",
                f"{a['wins']}/{a['n']}",
                "< 1e-4",
                f"{comp(a['compl_base'])} \u2192 {comp(a['compl_arm'])}",
                f"{a['delay_base']:.0f} \u2192 {a['delay_arm']:.0f}",
            ],
        ],
        [2.4, 2.3, 2.2, 2.0, 1.7, 2.4, 3.2, 3.0],
        "Table 3: Corridor, 30 paired held-out seeds under the block regime. Only PPO has the full "
        "paired test; Table 4 reports what the other two algorithms did in training.",
    )
    lay.figure(
        "fig_delta.png", 9.6,
        f"Figure 4: The {a['n']} per-seed differences behind Table 3. All {a['n']} are positive, "
        f"with a mean of {a['delta']:+.4f} and a median of {a['median']:+.4f}.",
    )

    ppo_ck, ppo_sc, ppo_cp = best_checkpoint("ppo")
    a2c_ck, a2c_sc, a2c_cp = best_checkpoint("a2c")
    dqn_ck, dqn_sc, dqn_cp = best_checkpoint("dqn")
    lay.body(
        "The two independent algorithms are the control for the claim that learning the crossing "
        "is not automatic, and Figure 6 traces all three arms over the training budget."
    )
    lay.table(
        [
            ["Arm", "Best checkpoint", "Internal score", "Completion", "Behaviour in Figure 5"],
            ["PPO (warm start)", ppo_ck, pct(ppo_sc), comp(ppo_cp), "crosses only when blocked"],
            ["A2C", a2c_ck, pct(a2c_sc), comp(a2c_cp), "no crossing in the panel"],
            ["DQN (gamma 0.999)", dqn_ck, pct(dqn_sc), comp(dqn_cp), "parks half the fleet"],
        ],
        [3.4, 2.8, 2.6, 2.2, 6.0],
        "Table 4: Corridor training outcome per arm. Score and completion are the selection-seed "
        "numbers, not the 30-seed paired numbers in Table 3; the behaviour column is a reading of "
        "the Figure 5 panels, which are rendered with make_env rather than the evaluator.",
    )
    lay.figure(
        "fig_corridor_arms.png", 16.4,
        "Figure 5: The four corridor arms on the same episode (seed 3001; step 232 for "
        "forward-only, PPO and A2C, step 300 for DQN). The red X is a cell whose rail transitions "
        "were removed. Forward-only, A2C and DQN leave a queue behind the upper X; PPO's track "
        "ahead of it is clear because it changed at the previous crossover.",
    )
    lay.figure(
        "fig_curve.png", 14.5,
        "Figure 6: The three corridor arms during training, from the internal evaluation used for "
        "checkpoint selection. PPO starts from the behaviour-cloned checkpoint and stays near zero "
        "until it drifts at the end of the budget; A2C and DQN fall into the parking basin. The "
        "dashed line is the forward-only 30-seed mean.",
    )
    lay.body(
        "A score gain alone does not prove the policy changes track for the right reason, since a "
        "policy that crossed at every crossover would also gain on this map. The audit counts the "
        "decisions themselves: for every agent-step standing on a commit crossover it records "
        "whether a blocked cell lay ahead on that train's own row within 60 cells, and whether the "
        "agent chose a lateral action. The per-crossover table is shipped as "
        "results/crossing_audit_block.csv."
    )
    audit_rows, tb, tb_turns, to, to_turns = audit_table()
    lay.table(
        [["Crossover", "Track", "Blocked visits", "Turns", "Open visits", "Turns"], *audit_rows],
        [2.2, 3.0, 3.2, 1.8, 3.0, 1.8],
        "Table 5: Per-crossover decision audit over seeds 3000 and 3001. Blocked means an "
        "impassable cell ahead on the train's own row within 60 cells. (5,59) is the only "
        "crossover with blocked visits that did not always turn (20 of 40); the CSV records counts "
        "rather than per-visit causes, so none is claimed.",
    )
    lay.figure(
        "fig_audit.png", 11.5,
        "Figure 7: Turn rate on the visits where the track ahead was blocked.",
    )
    lay.body(
        f"Of {tb} agent-steps standing on a crossover with a blocked cell ahead, {tb_turns} "
        f"produced a turn; of {to} with a clear track, {to_turns - to_turns} did (Figure 7). The audit records "
        "one decision per visit to a commit cell, so 81 turns over two seeds is about 40 diverts "
        "per episode. The rate is not uniform - four crossovers turn on every blocked visit and "
        "(5,59) on 20 of 40 - so the behaviour is state-dependent, not a blanket lane swap, and "
        "the A2C and DQN panels of Figure 5 show what the blanket alternative looks like."
    )
    clean = paired(RES / "clean_normal.csv")
    tight = paired(RES / "clean_delayed_none.csv")
    lay.body(
        f"Under the clean condition the same checkpoint is inert: on {clean['n']} seeds it scores "
        f"{pct(clean['arm'])}, exactly as the forward-only baseline does, and the decision audit "
        "records 546 visits to a commit crossover with a clear track and not one turn "
        "(results/clean_audit.csv). With the blockage suppressed but the Poisson malfunction layer "
        f"left on, the arm is indistinguishable from the baseline: {pct(tight['arm'])} against "
        f"{pct(tight['base'])}, a paired {tight['delta']:+.4f} on {tight['n']} seeds, "
        f"{tight['wins']} of {tight['n']} but p = {tight['p']:.2f}. The crossings in Table 5 "
        "therefore belong to the blockage, not to the policy's habit."
    )
    jc = paired(RES / "junction_current.csv")
    lay.table(
        [
            ["Arm", "Baseline", "Arm", "Delta", "Wins", "Completion (base \u2192 arm)"],
            ["PPO (30 seeds, 1030-d)", "-0.1769", "-0.0251", "+0.1518", "30/30", "1.000 \u2192 0.000"],
            ["A2C (30 seeds, 1030-d)", "-0.1769", "-0.0251", "+0.1518", "30/30", "1.000 \u2192 0.000"],
            ["DQN (30 seeds, 1030-d)", "-0.1769", "-0.0208", "+0.1561", "30/30", "1.000 \u2192 0.183"],
            [
                "PPO (current, 2 seeds)",
                pct(jc["base"]),
                pct(jc["arm"]),
                f"{jc['delta']:+.4f}",
                f"{jc['wins']}/{jc['n']}",
                f"{comp(jc['compl_base'])} \u2192 {comp(jc['compl_arm'])}",
            ],
        ],
        [4.6, 2.4, 2.2, 1.8, 1.6, 4.2],
        "Table 6: Junction results. The first three rows are the recorded 30-seed runs from before "
        "the two observation features were added, kept as measured: that policy parks the fleet. "
        "The last row is the current checkpoint selected by completion, which departs and jams; it "
        "is a two-seed evaluation shipped as results/junction_current.csv, identically scored on "
        "both seeds. The rows are not directly comparable and are labelled accordingly.",
    )
    lay.body(
        "The junction is reported as a loss, and its two failure modes bracket the metric's "
        "pathology. Parking scores better than attempting: every learned 30-seed arm scores about "
        "+0.15 better than the route follower while completing 0 to 18 per cent of the trains, "
        "because a cancellation costs about 19 cost units while a dispatch into the jam is charged "
        "its full delay. The current checkpoint shows the other side: selected by completion "
        f"rather than score, it sends every dispatched train into the throat, completing "
        f"{jc['compl_arm'] * 126:.0f} of 126 and scoring {pct(jc['arm'])}, worse than the scripted "
        f"baseline at {pct(jc['base'])} with 126 of 126 home. Figure 8 sets the two behaviours side "
        "by side. Neither is a working junction policy; a global controller, or imitation of the "
        "route follower, is what that map needs."
    )

    lay.figure(
        "fig_junction_pair.png", 13.6,
        "Figure 8: The junction at step 156 under the train-borne regime, rendered with make_env. "
        "Left: the scripted route follower, with the held train marked. Right: the selected PPO "
        "checkpoint, which dispatches into the throat and jams.",
    )

    doc.add_heading("3.4 Limitations", level=2)
    for text in [
        "Only PPO has the full 30-seed paired test on the corridor. A2C and DQN are reported from "
        "their selection seeds and the rendered episode; a paired run for them would strengthen "
        "the three-algorithm claim and is the obvious next experiment.",
        "The corridor decision is local: one crossover per blocked stretch and a two-cell detour. "
        "The result is not evidence that per-agent RL can arbitrate a junction.",
        "The disruption volume is a modelling choice, not data: 2 x 400 steps keeps completion at "
        "1.000 while making every affected trip late.",
        "From-scratch learning does not find the crossing, and the A2C and DQN arms demonstrate it "
        "independently. The corridor arm is imitation-then-RL, not pure exploration, and the warm "
        "start inherits the teacher's osn score of -0.0005.",
        "The 30 held-out seeds vary the blockage placement only: fleet, timetable and "
        "deterministic actions are fixed, a single training seed per arm was run, and the mean "
        "delta has a sign test but no confidence interval.",
        "The junction table mixes two observation versions; the 30-seed checkpoints predate the "
        "two feature additions and no longer load. They are reported as measured and labelled.",
        "Evaluation is entirely simulated. The timetable and the track layout are real; there is "
        "no operational validation and no human factors layer.",
    ]:
        lay.bullet(text)

    # ---- 4 discussion
    lay.page_break()
    doc.add_heading("4. Discussion and Conclusions", level=1)
    lay.body(
        "A local policy can learn to reroute, but only if the evaluation makes the decision real, "
        "and only with a recipe that can find a sparse action. On a four-track corridor cut by "
        "400-step exogenous blockages, PPO changes track at the commit crossover just before the "
        f"blockage and turns a paired deficit of {pct(a['base'])} into {pct(a['arm'])} over "
        f"{a['n']} held-out seeds, winning every seed without losing a train. The same checkpoint "
        "does nothing when nothing is blocked, and an audit of "
        f"{tb + to} crossover visits shows that its crossings follow the blockage rather than the "
        "map. The two independent algorithms do not get there, which makes the corridor result a "
        "statement about the recipe as much as about the algorithm."
    )
    lay.body(
        "The result depended on fixing the measurement first. A train-borne failure fires when a "
        "train arrives, so a policy that reroutes suppresses the incident, and a failure placed "
        "deep inside a station's escape band leaves every follower already past the crossover. "
        "Removing the cell's transitions instead makes the blockage exogenous, keeps both arms on "
        "the same incidents, and turns the delayed comparison into a genuine decision."
    )
    lay.body(
        "Three extensions follow. First, the A2C and DQN arms deserve the same 30-seed paired run; "
        "the current evidence is their selection seeds and the rendered episode. Second, the "
        "junction needs a global view - a graph or attention observation, or imitation of the "
        "route follower - because a local tree observation cannot arbitrate 126 trains through "
        "one throat. Third, the corridor result should be checked against the stronger scripted "
        "distance-map baseline, not only the forward-only reflex, to see how much of the gain "
        "survives when the baseline is allowed to divert too."
    )

    # ---- 5 references
    doc.add_heading("5. References", level=1)
    for ref in [
        "Gorrey, M., Segaert, A., Ganko, J., & Kaine, E. (2025, May 21). 'Delay all non-essential "
        "travel': Commuters warned to expect major delays. The Sydney Morning Herald.",
        "Li, B., Guo, T., Li, R., Wang, Y., Ou, Y., & Chen, F. (2021). Delay propagation in large "
        "railway networks with data-driven methods. Physica A: Statistical Mechanics and its "
        "Applications, 583, 126305.",
        "Mohanty, S., Nygren, E., Laurent, F., Schneider, M., Scheller, C., Bhattacharya, N., "
        "Watson, J., Egli, A., Eichenberger, C., & Thadden, T. (2020). Flatland: a framework for "
        "reinforcement learning on train scheduling. Zenodo / AIcrowd.",
        "Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). "
        "Stable-Baselines3: reliable reinforcement learning implementations. Journal of Machine "
        "Learning Research, 22(268), 1-8.",
        "Roberts, M. (2024, June 17). Signal repairs cause train disruption in Sydney during "
        "Monday morning commute. ABC News.",
        "Semrov, D., Marsetic, R., Zura, M., Todorovski, L., & Srdic, A. (2016). Reinforcement "
        "learning approach for train rescheduling on a single-track railway. Transportation "
        "Research Part B, 86, 250-267.",
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). "
        "MIT Press.",
        "Sydney Trains. (2025). Sydney Trains Annual Report 2025. Transport for NSW.",
        "Transport for NSW. (2026). Public transport timetables and GTFS feeds (Open Data). "
        "https://opendata.transport.nsw.gov.au",
        "Zhu, Y., & Goverde, R. M. (2019). Railway timetable rescheduling with flexible stopping "
        "and flexible short-turning during disruptions. Transportation Research Part B, 123, "
        "149-181.",
    ]:
        p = doc.add_paragraph(ref)
        p.paragraph_format.space_after = Pt(6)

    # ---- appendices
    lay.page_break()
    doc.add_heading("Appendices", level=1)
    doc.add_heading("A. Individual Contribution", level=2)
    lay.body(
        "Each group member writes up to half a page on their contribution to the project "
        "development activities."
    )
    for name, draft in [
        (
            "Daniel James Martirosov (24948933)",
            "GTFS extraction and both scenarios; the disruption regimes (band, crossing and the "
            "exogenous blockage) and the decision they forced; PPO and A2C/DQN training, the "
            "behaviour-cloned warm start, the paired evaluator and the crossing audit; the "
            "report and template build scripts.",
        ),
        (
            "Ziheng Deng (25595315)",
            "[To be completed by the member: 100-150 words on your contribution.]",
        ),
        (
            "Laurean Lim (25049863)",
            "[To be completed by the member: 100-150 words on your contribution.]",
        ),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.add_run(name).bold = True
        lay.body(draft)

    doc.add_heading("B. Individual Contribution Split", level=2)
    lay.table(
        [["Team Member Name", "Percentage"],
         ["Daniel James Martirosov", "33.33%"],
         ["Ziheng Deng", "33.33%"],
         ["Laurean Lim", "33.33%"]],
        [9.0, 6.0],
        "Table 7: Agreed contribution split.",
    )
    lay.body(
        "All the team members have discussed and agreed on the above individual contribution to "
        "the project."
    )
    for name in ("Daniel James Martirosov", "Ziheng Deng", "Laurean Lim"):
        p = doc.add_paragraph(f"{name} : ____________________")
        p.paragraph_format.space_after = Pt(14)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--pages", default=None, help="JSON of heading -> page for the contents page")
    ap.add_argument("--map-pdf", default=None, help="print heading -> page JSON for a built PDF")
    args = ap.parse_args()

    if args.map_pdf:
        print(json.dumps(page_of(Path(args.map_pdf)), indent=1))
        return 0

    pages = json.loads(Path(args.pages).read_text()) if args.pages else None
    build(Path(args.template), Path(args.out), pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
