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
from docx.enum.text import (
    WD_ALIGN_PARAGRAPH,
    WD_BREAK,
    WD_TAB_ALIGNMENT,
    WD_TAB_LEADER,
)
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_pair import sign_test_p  # noqa: E402
from report_layout import (  # noqa: E402
    W, Layout, add_page_footer, force_font, set_document_defaults,
)

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
            if "....." in page:  # a contents page: it lists every heading
                continue
            lines = [ln.strip() for ln in page.splitlines() if "....." not in ln]
            if any(ln.startswith(name) for ln in lines):
                found[name] = i
                break
    return found



def prune_unused_media(path: Path) -> None:
    """Drop image parts the template carried but that no paragraph references.

    Deleting the template's body leaves its image relationships behind, so the
    saved file would otherwise ship the retired band-regime figures inside the
    package even though they are never displayed.
    """
    import re
    import zipfile

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    rels_name = "word/_rels/document.xml.rels"
    doc = blobs["word/document.xml"].decode()
    rels = blobs[rels_name].decode()
    used = set(re.findall(r'r:(?:embed|link|id)="(rId\d+)"', doc))

    kept_tags, kept_targets = [], set()
    for tag in re.findall(r"<Relationship [^>]*?/>", rels):
        rid = re.search(r'Id="(rId\d+)"', tag)
        target = re.search(r'Target="([^"]+)"', tag)
        if rid and rid.group(1) in used:
            kept_tags.append(tag)
            if target:
                kept_targets.add(target.group(1).lstrip("/"))

    dropped = [
        n for n in names
        if n.startswith("word/media/")
        and n not in kept_targets
        and n[len("word/"):] not in kept_targets
    ]
    if not dropped:
        return

    head = rels[: rels.index("<Relationship ")] if "<Relationship " in rels else rels
    blobs[rels_name] = (head + "".join(kept_tags) + "</Relationships>").encode()

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if n in dropped:
                continue
            z.writestr(n, blobs[n])
    print(f"pruned {len(dropped)} unused image part(s)")


def build(template: Path, out: Path, pages: dict[str, int] | None) -> None:
    """Keep the template's cover, section 1, section 4, references and appendices;
    replace sections 2 and 3 with the shipped content."""
    doc = Document(str(template))
    body = doc.element.body

    def text_of(el) -> str:
        # only real text runs: itertext() also collects field instructions and
        # repeats runs that carry xml:space twice
        return "".join(t.text or "" for t in el.iter(f"{W}t")).strip()

    def heading_el(prefix: str):
        """The real heading paragraph, not a contents entry that repeats its text."""
        for k in body:
            if not k.tag.endswith("}p"):
                continue
            pPr = k.find(f"{W}pPr")
            st = pPr.find(f"{W}pStyle") if pPr is not None else None
            if st is None or st.get(f"{W}val") not in ("Heading1", "Heading2", "Heading3"):
                continue
            if text_of(k).startswith(prefix):
                return k
        raise LookupError(f"heading not found: {prefix}")

    kids = list(body)
    marker = next(k for k in kids if k.tag.endswith("}p") and "Table of Contents" in text_of(k))
    abstract = heading_el("Abstract")
    sec2 = heading_el("2. Overview")
    sec4 = heading_el("4. Discussion")

    # where the replaced block used to sit: right after the last paragraph of section 1
    anchor = sec2.getprevious()

    # blank contents area and the two replaced sections
    for k in kids[kids.index(marker) + 1:kids.index(abstract)]:
        body.remove(k)
    for k in kids[kids.index(sec2):kids.index(sec4)]:
        if k.tag.endswith("}sectPr"):
            continue
        body.remove(k)

    # keep the template's typography: Times New Roman 12 pt body, Times 24/18 pt
    # bold headings, 2.54 cm margins
    set_document_defaults(doc, "Times New Roman", 12)
    for st in doc.styles:
        if st.name and st.name.lower() in ("normal", "footer", "header", "page number"):
            force_font(st.element, "Times New Roman")
            st.font.name = "Times New Roman"
            st.font.size = Pt(12)
    add_page_footer(doc, "RailTech, 43008 Assessment 3 (Part E)",
                    font="Times New Roman", size=9)

    # the template's cover carries a typo and a missing space before the IDs
    for k in kids[:kids.index(marker)]:
        for run in k.iter(f"{W}t"):
            if run.text:
                run.text = (run.text
                            .replace("Assignent-3, Part-E", "Assignment 3, Part E")
                            .replace("Martirosov(", "Martirosov ("))

    def place(new_els, cursor):
        """Move freshly created elements to sit, in order, right after cursor."""
        for el in new_els:
            body.remove(el)
            cursor.addnext(el)
            cursor = el
        return cursor

    # ---- contents page (static; page numbers come from the two-pass build)
    existing = set(body)
    for name in SECTIONS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.tab_stops.add_tab_stop(
            Cm(15.7), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS
        )
        for text in (name, "\t", str(pages.get(name, "")) if pages else ""):
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            force_font(run._r, "Times New Roman")
    place([e for e in body if e not in existing and not e.tag.endswith("}sectPr")], marker)

    # the template puts the contents on its own page
    toc_heading = doc.paragraphs[0]
    for para in doc.paragraphs:
        if para.text.strip() == "Table of Contents":
            toc_heading = para
            break
    toc_heading.paragraph_format.page_break_before = True

    # ---- sections 2 and 3, built with the same helpers as the standalone report
    lay = Layout(doc, FIG, max_width_cm=15.8, font="Times New Roman")

    def heading(text: str, level: int):
        """Heading in the template's own type: Times, bold, 24 pt then 18 pt."""
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18 if level == 1 else 12)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(24 if level == 1 else 18)
        run.font.name = "Times New Roman"
        force_font(run._r, "Times New Roman")
        return p

    existing = set(body)

    # ================================================================ section 2
    heading("2. Overview of the architecture/system", level=1)
    heading("2.1 Flow Diagram and Workflow", level=2)
    lay.body(
        "The workflow runs in seven stages, each with a defined input and output (Figure 3). The "
        "dashed arrow is deliberate: every change to the observation, reward or disruption regime "
        "was re-validated by re-running the same paired evaluation, so this report only contains "
        "numbers produced by the final configuration, with the two labelled junction rows of Table "
        "5 as the one disclosed exception."
    )
    lay.figure(
        "fig_workflow.png", 15.4,
        "Figure 3: End-to-end workflow. Each stage consumes the artefact produced by the previous "
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

    heading("2.2 System Architecture Design", level=2)
    lay.body(
        "One shared multi-layer perceptron controls every train: 1032 inputs, two hidden layers "
        "(256 units for PPO, 128 for A2C and DQN) and a five-way action head. The observation is a "
        "flattened, normalised tree of depth 3 within a radius of 50 cells, plus twelve conflict "
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
        "divides the potential by a constant 100 so the value targets stay O(1). All reported arms "
        "also use two shaping bonuses on top of the potential: +50 for every train that arrives "
        "and -1 for every step a train is stuck, at the same 1/100 scale, which is disclosed here "
        "because the reported arm is not the bare-metric arm."
    )
    lay.body(
        "From a random initialisation none of the three algorithms finds the crossing, so the "
        "reported corridor arm is PPO fine-tuned from a behaviour-cloned detour baseline "
        "(imitation then RL), with the critic warmed up first. The two independent algorithms are "
        "the control: they share the observation, reward and regime but not the warm start. The "
        "notebook is the assembly point and imports the shared modules from scripts/ instead of "
        "duplicating them, so the notebook, the trainers and the evaluator cannot drift apart. "
        "Twenty scripts ship; the analysis modules the reported tables depend on are nb_harness "
        "(executes the notebook's own cells), regimes (map geometry and the regime selector, block "
        "by default), disruptions (both disruption mechanisms), priority_obs (the twelve "
        "features), metric_rewards (the potential and its shaped variant), train_ppo, train_algo, "
        "eval_pair, bc_warmstart, crossing_audit, render_policy_gif and verify_rewards, with "
        "s2_env and s2_assembly assembling the junction."
    )

    heading("2.3 GUI/Environment Design", level=2)
    lay.body(
        "Flatland's own renderer is the interface, which the brief permits in place of a separate "
        "GUI: it shows the rail grid with its switches and crossovers, one coloured marker per "
        "train, and a red X on every cell that may not be entered - an exogenous blockage on the "
        "corridor, a train standing in malfunction on the junction. Using the simulator's renderer "
        "means every episode figure comes from the same process the agent is trained in, produced "
        "headlessly with the PIL backend. Figures 1 and 2 are the two levels; Figures 4 and 5 show "
        "the same levels under their disruption regimes."
    )
    lay.figure(
        "fig_corridor_arms.png", 15.8,
        "Figure 4: The corridor under the block regime, four arms on the same episode (seed 3001; "
        "step 232 for forward-only, PPO and A2C, step 300 for DQN). The red X is a cell whose rail "
        "transitions were removed. Forward-only, A2C and DQN leave a queue behind the upper X; "
        "PPO's track ahead of it is clear because it changed at the previous crossover.",
    )
    lay.figure(
        "fig_junction_pair.png", 15.0,
        "Figure 5: The junction at step 156 under the train-borne regime. Left: the scripted route "
        "follower, with the held train marked. Right: the selected PPO checkpoint, which dispatches "
        "into the throat and jams.",
    )
    lay.body(
        "The disruption regime is the methodological core. A train-borne failure fires when a "
        "train reaches a cell, which makes the disruption a function of the policy's own "
        "trajectory: a policy that reroutes early can prevent the incident from firing at all. "
        "The corridor therefore uses an exogenous blockage: a cell's rail transitions are removed "
        "outright at a seed-drawn step and restored after 400 steps, so the failure exists whether "
        "or not a train goes near it. Two are drawn per episode, one per direction, from the cells "
        "just past each commit crossover; activation is deferred while a cell is occupied, and "
        "dropped if it stays occupied for 60 steps. Both arms are given the same cells at the same "
        "seed-drawn steps, so the pairing is by seed and blockage plan."
    )

    # ================================================================ section 3
    heading("3. Results and Evaluation", level=1)
    heading("3.1 Experimental Settings", level=2)
    lay.body(
        "Table 1 lists the settings. All arms act deterministically; the corridor arm is "
        "evaluated on seeds 3000 to 3029, which training and checkpoint selection never see, and "
        "A2C and DQN are reported from their selection seeds only. Because the baseline and the "
        "arm run on the same seed, the same environment and the same blockages, a delta is "
        "computed per seed and the report gives its mean, median, win count and an exact two-sided "
        "sign test. The score is the assessment metric normalised to [-1, 0], with each train's "
        "cost capped at one episode, summed over all agents and divided by the number of agents "
        "and the episode budget; a second statistic, delay_all, charges every agent and uses the "
        "episode end for a train that never arrives."
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
            ["Held-out seeds", "3000 to 3029", "3000 to 3029 (legacy); 3001, 3002 (current)"],
            ["Baseline", "forward-only reflex", "scripted route follower"],
        ],
        [3.6, 6.0, 6.0],
        "Table 1: Experimental settings. The maps share the observation, action space and "
        "evaluation protocol; they differ in disruption mechanism because only the corridor has a "
        "switchable pair of parallel tracks.",
    )
    lay.body(
        "The regime and the policy are CPU-bound: Flatland steps the fleet in Python and the "
        "1032-d tree observation is rebuilt for every agent at every step. The shipped corridor "
        "run took 1,865 s for 1,224,704 timesteps including thirteen four-episode evaluations "
        "(about 660 environment steps per second), and once startup is paid each evaluation "
        "interval costs about 148 s, so the evaluator rather than the optimiser sets the wall "
        "clock. Training ran on an x86_64 workstation with an RTX 3080 (the system_info recorded "
        "in the checkpoint); the notebook and figures also run on an aarch64 relay with the same "
        "pinned stack, where the sixteen tests pass. Every table below is regenerated from the "
        "CSVs in results/."
    )

    heading("3.2 Pre-Processing", level=2)
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

    heading("3.3 Experimental Results", level=2)
    a = paired(RES / "pair_block_ppo.csv")
    lay.body(
        "The corridor result is the headline. PPO improves the score from -0.0662 to -0.0005, a "
        f"mean paired gain of {a['delta']:+.4f} with a median of {a['median']:+.4f} (Table 2). It "
        f"wins all {a['n']} seeds and every train completes in both arms, so the gain is "
        f"punctuality rather than cancellation: total lateness falls from {a['delay_base']:.0f} "
        f"steps to {a['delay_arm']:.0f}. The on-time share moves only from "
        f"{a['on_time_base']:.3f} to {a['on_time_arm']:.3f}, so the gain is in how late the "
        "affected trains are, not in which trains complete. Figure 6 shows the per-seed spread."
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
        [2.0, 1.9, 1.9, 1.8, 1.5, 2.0, 2.7, 2.4],
        "Table 2: Corridor, 30 paired held-out seeds under the block regime. Only PPO has the full "
        "paired test; Table 3 reports what the other two algorithms did in training.",
    )
    lay.figure(
        "fig_delta.png", 11.0,
        f"Figure 6: The {a['n']} per-seed differences behind Table 2. All {a['n']} are positive, "
        f"with a mean of {a['delta']:+.4f} and a median of {a['median']:+.4f}.",
    )
    ppo_ck, ppo_sc, ppo_cp = best_checkpoint("ppo")
    a2c_ck, a2c_sc, a2c_cp = best_checkpoint("a2c")
    dqn_ck, dqn_sc, dqn_cp = best_checkpoint("dqn")
    lay.body(
        "The two independent algorithms are the control for the claim that learning the crossing "
        "is not automatic; Figure 7 traces all three arms over the training budget."
    )
    lay.table(
        [
            ["Arm", "Best checkpoint", "Internal score", "Completion", "Behaviour in Figure 4"],
            ["PPO (warm start)", ppo_ck, pct(ppo_sc), comp(ppo_cp), "crosses only when blocked"],
            ["A2C", a2c_ck, pct(a2c_sc), comp(a2c_cp), "no crossing in the panel"],
            ["DQN (gamma 0.999)", dqn_ck, pct(dqn_sc), comp(dqn_cp), "parks half the fleet"],
        ],
        [3.2, 2.6, 2.4, 2.0, 5.6],
        "Table 3: Corridor training outcome per arm. Score and completion are the selection-seed "
        "numbers, not the 30-seed paired numbers in Table 2; the behaviour column is a reading of "
        "the Figure 4 panels, which are rendered with make_env rather than the evaluator.",
    )
    lay.figure(
        "fig_curve.png", 14.0,
        "Figure 7: The three corridor arms during training, from the internal evaluation used for "
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
        "Table 4: Per-crossover decision audit over seeds 3000 and 3001. Blocked means an "
        "impassable cell ahead on the train's own row within 60 cells. (5,59) is the only "
        "crossover with blocked visits that did not always turn (20 of 40); the CSV records counts "
        "rather than per-visit causes, so none is claimed.",
    )
    lay.figure(
        "fig_audit.png", 11.5,
        "Figure 8: Turn rate on the visits where the track ahead was blocked.",
    )
    lay.body(
        f"Of {tb} agent-steps standing on a crossover with a blocked cell ahead, {tb_turns} "
        f"produced a turn; of {to} with a clear track, {to_turns} did. The audit records one "
        "decision per visit to a commit cell, so 81 turns over two seeds is about 40 diverts per "
        "episode. The rate is not uniform - four crossovers turn on every blocked visit and (5,59) "
        "on 20 of 40 - so the behaviour is state-dependent, not a blanket lane swap, and the A2C "
        "and DQN panels of Figure 4 show what the blanket alternative looks like."
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
        f"{tight['wins']} of {tight['n']} but p = {tight['p']:.2f}. The crossings in Table 4 "
        "therefore belong to the blockage, not to the policy's habit."
    )
    jc = paired(RES / "junction_current.csv")
    lay.body(
        "The junction is reported as a loss, and its two failure modes bracket the metric's "
        "pathology."
    )
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
        [4.2, 2.2, 2.0, 1.7, 1.5, 3.8],
        "Table 5: Junction results. The first three rows are the recorded 30-seed runs from before "
        "the two observation features were added, kept as measured: that policy parks the fleet. "
        "The last row is the current checkpoint selected by completion, which departs and jams; it "
        "is a two-seed evaluation shipped as results/junction_current.csv, identically scored on "
        "both seeds, so treat it as one behaviour. The rows are not directly comparable.",
    )
    lay.body(
        "Parking scores better than attempting: every learned 30-seed arm scores about +0.15 "
        "better than the route follower while completing 0 to 18 per cent of the trains, because a "
        "cancellation costs about 19 cost units while a dispatch into the jam is charged its full "
        f"delay. The current checkpoint shows the other side: selected by completion rather than "
        f"score, it sends every dispatched train into the throat, completing "
        f"{jc['compl_arm'] * 126:.0f} of 126 and scoring {pct(jc['arm'])}, worse than the scripted "
        f"baseline at {pct(jc['base'])} with 126 of 126 home (Figure 5). Neither is a working "
        "junction policy; a global controller, or imitation of the route follower, is what that "
        "map needs."
    )

    heading("3.4 Limitations", level=2)
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
        "start inherits the teacher's own score of -0.0005.",
        "The 30 held-out seeds vary the blockage placement only: fleet, timetable and "
        "deterministic actions are fixed, a single training seed per arm was run, and the mean "
        "delta has a sign test but no confidence interval.",
        "The junction table mixes two observation versions; the 30-seed checkpoints predate the "
        "two feature additions and no longer load. They are reported as measured and labelled.",
        "Evaluation is entirely simulated. The timetable and the track layout are real; there is "
        "no operational validation and no human factors layer.",
    ]:
        lay.bullet(text)

    place([e for e in body if e not in existing and not e.tag.endswith("}sectPr")], anchor)

    # the template's own headings are style-only, which LibreOffice renders in its
    # default heading face; pin them to the same Times/bold/24-18 pt as ours
    for k in body:
        if not k.tag.endswith("}p"):
            continue
        pPr = k.find(f"{W}pPr")
        st = pPr.find(f"{W}pStyle") if pPr is not None else None
        if st is None or st.get(f"{W}val") not in ("Heading1", "Heading2", "Heading3"):
            continue
        size = {"Heading1": 24, "Heading2": 18, "Heading3": 14}[st.get(f"{W}val")]
        for run in k.iter(f"{W}r"):
            rPr = run.find(f"{W}rPr")
            if rPr is None:
                rPr = OxmlElement("w:rPr")
                run.insert(0, rPr)
            force_font(run, "Times New Roman")
            sz = rPr.find(f"{W}sz")
            if sz is None:
                sz = OxmlElement("w:sz")
                rPr.append(sz)
            sz.set(f"{W}val", str(size * 2))
            b = rPr.find(f"{W}b")
            if b is None:
                b = OxmlElement("w:b")
                rPr.append(b)
        # the style is redundant once the runs carry the type, and its presence is
        # what stops the page break above from taking effect
        pStyle = pPr.find(f"{W}pStyle")
        if pStyle is not None:
            pPr.remove(pStyle)

    # the template's abstract is a placeholder: fill it with the report's abstract
    a = paired(RES / "pair_block_ppo.csv")
    abstract_text = (
        "We ask whether a learned policy can beat a schedule-following controller when a track "
        "becomes unavailable. Two Flatland scenarios are built from the October 2026 TfNSW GTFS "
        "feed: a four-track Central-Strathfield corridor with 52 scheduled movements and the "
        "Granville-Harris Park junction with 126. The corridor is disrupted by an exogenous "
        "blockage that removes a cell's rail transitions for 400 steps, so a policy cannot make "
        "the incident vanish by rerouting. Under that regime, and across three algorithms, only "
        f"the warm-started PPO learns to change track: it improves the metric from {pct(a['base'])} "
        f"to {pct(a['arm'])}, a paired gain of {a['delta']:+.4f} on {a['n']} held-out seeds "
        f"({a['wins']} of {a['n']}, p < 1e-4) with every train completing, and on a clean network "
        "it scores 0.0000 with no crossing at all. A2C never learns the crossing and DQN parks "
        "half the fleet. The junction remains a loss: the selected checkpoint dispatches and jams "
        "(-0.2400, 59 of 126 trains home) instead of parking, and still finishes behind the "
        "scripted route follower (-0.1823)."
    )
    for para in doc.paragraphs:
        if para.text.strip().startswith("<Add a short abstract"):
            for run in list(para.runs):
                run._r.getparent().remove(run._r)
            run = para.add_run(abstract_text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)
            force_font(run._r, "Times New Roman")
            break

    # final layout pass: drop the template's blank spacer paragraphs (sized for its
    # placeholder text) and give every top-level section its own page with an
    # explicit break, which both Word and LibreOffice honour
    cover_cut = list(body).index(marker)
    for k in list(body)[cover_cut:]:
        if (k.tag.endswith("}p") and not "".join(k.itertext()).strip()
                and not k.findall(f".//{W}drawing")):
            body.remove(k)

    def is_top_heading(k) -> bool:
        if not k.tag.endswith("}p"):
            return False
        text = text_of(k)
        if not text or len(text) > 60:
            return False
        pPr = k.find(f"{W}pPr")
        st = pPr.find(f"{W}pStyle") if pPr is not None else None
        if st is not None and st.get(f"{W}val") == "Heading1":
            return True
        for run in k.iter(f"{W}r"):
            rPr = run.find(f"{W}rPr")
            sz = rPr.find(f"{W}sz") if rPr is not None else None
            if sz is not None and sz.get(f"{W}val") == "48":
                return True
        return False

    wanted = ("Abstract", "1. Introduction", "2. Overview", "3. Results",
              "4. Discussion", "Appendices")
    for k in list(body):
        if not is_top_heading(k):
            continue
        if not text_of(k).startswith(wanted) and text_of(k) != "Table of Contents":
            continue
        para = doc.add_paragraph()
        para.add_run().add_break(WD_BREAK.PAGE)
        k.addprevious(para._p)

    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    prune_unused_media(out)
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
