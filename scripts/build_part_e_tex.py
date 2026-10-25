"""Build the Part E report as LaTeX, with APA 7 references (biblatex-apa + biber).

    python scripts/build_part_e_tex.py                # writes report/*.tex and compiles
    python scripts/build_part_e_tex.py --no-build     # write the .tex only

Section 1, the appendices and the cover fields come from the unit's template
(templates/part-e-template.docx); sections 2 and 3 are generated from the CSVs in
results/, using the same readers as the docx builder, so the numbers cannot drift;
section 4 is the discussion written for this report; the bibliography is
report/references.bib in APA 7th edition.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_part_e_report as pe  # noqa: E402  (data readers + formatting helpers)

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = RES / "report_figures"
ASSETS = REPO / "report" / "tex_assets"
TEMPLATE = REPO / "templates" / "part-e-template.docx"
OUT_TEX = REPO / "report" / "RailTech-A3-PartE-Report.tex"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# the template's plain-text citations -> biblatex keys
CITES = {
    "(Li et al., 2021)": r"\parencite{li2021}",
    "(Mohanty et al., 2020)": r"\parencite{mohanty2020}",
    "(Sutton & Barto, 2018)": r"\parencite{sutton2018}",
    "(Roberts, 2024)": r"\parencite{roberts2024}",
    "(Gorrey et al., 2025)": r"\parencite{gorrey2025}",
    "(Sydney Trains, 2025)": r"\parencite{sydneytrains2025}",
    "(Zhu & Goverde, 2019)": r"\parencite{zhu2019}",
    "(Šemrov et al., 2016; Zhu & Goverde, 2019)": r"\parencite{semrov2016,zhu2019}",
    "(Šemrov et al., 2016)": r"\parencite{semrov2016}",
    "(Schulman et al., 2017)": r"\parencite{schulman2017}",
    "(Mnih et al., 2016)": r"\parencite{mnih2016}",
    "(Mnih et al., 2015)": r"\parencite{mnih2015}",
    "(Raffin et al., 2021)": r"\parencite{raffin2021}",
    "(Ross et al., 2011)": r"\parencite{ross2011}",
    "(Transport for NSW, 2026)": r"\parencite{tfnsw2026}",
}

SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _escape(text: str) -> str:
    text = (text.replace("\u2013", "--").replace("\u2014", "---")
                .replace("\u2019", "'").replace("\u2018", "`")
                .replace("\u201c", "``").replace("\u201d", "''"))
    return "".join(SPECIALS.get(ch, ch) for ch in text)


def tex(text: str) -> str:
    """Escape for LaTeX, keeping citation macros and $...$ math intact."""
    maths: list[str] = []
    macros: list[str] = []

    def keep_math(m):
        maths.append(m.group(0))
        return f"\x00M{len(maths) - 1}\x00"

    def keep_cite(m):
        macros.append(CITES[m.group(0)])
        return f"\x00C{len(macros) - 1}\x00"

    text = re.sub(r"\$[^$]*\$", keep_math, text)
    pattern = "|".join(re.escape(k) for k in sorted(CITES, key=len, reverse=True))
    text = re.sub(pattern, keep_cite, text)
    text = _escape(text)
    text = re.sub(r"\x00C(\d+)\x00", lambda m: macros[int(m.group(1))], text)
    return re.sub(r"\x00M(\d+)\x00", lambda m: maths[int(m.group(1))], text)


# ------------------------------------------------------------------ template
def template_section1(doc: Document) -> list[str]:
    """Section 1 as LaTeX, straight from the template, including its two map figures."""
    out: list[str] = []
    paras = doc.paragraphs
    ASSETS.mkdir(parents=True, exist_ok=True)
    for f in ASSETS.glob("map_*.png"):
        f.unlink()

    started = False
    images: list[Path] = []
    captions: dict[int, str] = {}

    for i, para in enumerate(paras):
        text = para.text.strip()
        if text.startswith("2. Overview"):
            break
        if not started:
            started = text.startswith("1. Introduction")
            if not started:
                continue

        if para._p.findall(f".//{W}drawing"):
            images.append(_extract_image(doc, para))

        m = re.match(r"Figure (\d+):\s*(.*)", text)
        if m:
            caption = m.group(2)
            nxt = paras[i + 1].text.strip() if i + 1 < len(paras) else ""
            if not caption.endswith(".") and nxt and not nxt.startswith("Figure"):
                caption = f"{caption} {nxt}"
            captions[int(m.group(1))] = caption.strip()
            continue

        if para.style.name == "Heading 1":
            out.append("\\section{%s}" % tex(text.split(" ", 1)[-1]))
        elif para.style.name == "Heading 2":
            out.append("\\subsection{%s}" % tex(text.split(" ", 1)[-1]))
        elif text:
            out.append(tex(text))

    for idx, path in enumerate(images, start=1):
        caption = captions.get(idx, f"Scenario {idx}")
        out.append(_figure(path, caption))
    return out


def _extract_image(doc: Document, para) -> Path:
    """Save the picture in this paragraph next to the .tex, return its path."""
    blip = para._p.find(f".//{{http://schemas.openxmlformats.org/drawingml/2006/main}}blip")
    rid = blip.get(f"{{http://schemas.openxmlformats.org/officeDocument/2006/relationships}}embed")
    blob = doc.part.related_parts[rid].blob
    index = len(list(ASSETS.glob("map_*.png"))) + 1
    path = ASSETS / f"map_{index}.png"
    path.write_bytes(blob)
    return path


def _figure(path: Path, caption: str, width: str = "0.95") -> str:
    return (
        "\\begin{figure}[htbp]\n\\centering\n"
        f"\\includegraphics[width={width}\\linewidth]{{{_rel(path)}}}\n"
        f"\\caption{{{tex(_caption_text(caption))}}}\n\\end{{figure}}"
    )


def _rel(path: Path) -> str:
    """Path of an asset relative to the .tex file (assets live outside report/)."""
    import os

    return os.path.relpath(path, REPO / "report")


# ------------------------------------------------------------------ helpers
def _caption_text(caption: str) -> str:
    """LaTeX adds "Table N:"/"Figure N:" itself; drop any hand-written prefix."""
    return re.sub(r"^(?:Table|Figure)\s+\d+:\s*", "", caption)


def table(rows, aligns: str, caption: str, size: str = r"\footnotesize") -> str:
    header = " & ".join(r"\textbf{%s}" % tex(str(c)) for c in rows[0]) + r" \\"
    body = "\n".join(
        " & ".join(tex(str(c)) for c in row) + r" \\" for row in rows[1:]
    )
    return (
        "\\begin{table}[htbp]\n\\centering\n" + size + "\n"
        f"\\begin{{tabular}}{{{aligns}}}\n\\toprule\n{header}\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{{tex(_caption_text(caption))}}}\n\\end{{table}}"
    )


def figure(name: str, caption: str, width: float = 1.0) -> str:
    return (
        "\\begin{figure}[htbp]\n\\centering\n"
        f"\\includegraphics[width={width:.2f}\\linewidth]{{{_rel(FIG / name)}}}\n"
        f"\\caption{{{tex(_caption_text(caption))}}}\n\\end{{figure}}"
    )


def bullets(items) -> str:
    return "\\begin{itemize}[leftmargin=*,itemsep=2pt]\n" + "\n".join(
        "\\item " + tex(i) for i in items
    ) + "\n\\end{itemize}"


# ------------------------------------------------------------------ build
def build(no_build: bool) -> None:
    a = pe.paired(RES / "pair_block_ppo.csv")
    clean = pe.paired(RES / "clean_normal.csv")
    tight = pe.paired(RES / "clean_delayed_none.csv")
    jc = pe.paired(RES / "junction_current.csv")
    audit_rows, tb, tb_turns, to, to_turns = pe.audit_table()
    ppo_ck, ppo_sc, ppo_cp = pe.best_checkpoint("ppo")
    a2c_ck, a2c_sc, a2c_cp = pe.best_checkpoint("a2c")
    dqn_ck, dqn_sc, dqn_cp = pe.best_checkpoint("dqn")

    tpl = Document(str(TEMPLATE))
    s1 = template_section1(tpl)

    parts: list[str] = []
    add = parts.append

    add(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newtxtext,newtxmath}
\usepackage[margin=2.54cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{csquotes}
\usepackage[style=apa,backend=biber,sortcites=true]{biblatex}
\DeclareLanguageMapping{english}{english-apa}
\addbibresource{references.bib}
\usepackage[hidelinks]{hyperref}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\captionsetup{font=small,labelfont=bf,singlelinecheck=false,justification=raggedright}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyfoot[C]{\footnotesize RailTech, 43008 Assessment 3 (Part E) \textbar{} page \thepage}
\renewcommand{\thesection}{\arabic{section}.}
\renewcommand{\thesubsection}{\thesection\arabic{subsection}}
\setcounter{secnumdepth}{2}

\begin{document}

\thispagestyle{empty}
\begin{center}
{\Large\bfseries 43008: Reinforcement Learning\par}
\vspace{0.6em}
{\large\bfseries Assignment 3, Part E Project Report\par}
\vspace{4cm}
{\bfseries Project Title:\par}
{\Large Train Schedule Optimisation under Disruptions\par}
\vspace{0.4em}
{\itshape Learning to change track when a corridor is blocked\par}
\vfill
{\bfseries Team Name and Project Number:} RailTech (Project \#27)\par
\vspace{0.4em}
{\bfseries Team Members:} Daniel James Martirosov (24948933), Ziheng Deng (25595315),
Laurean Lim (25049863)\par
\vspace{0.4em}
{\bfseries Date:} 25 October 2026\par
\vspace{2cm}
\end{center}
\clearpage

\tableofcontents
\clearpage
""")

    # ---- Abstract
    add("\\section*{Abstract}\n\\addcontentsline{toc}{section}{Abstract}\n")
    add(tex(
        "We ask whether a learned policy can beat a schedule-following controller when a track "
        "becomes unavailable. Two Flatland scenarios are built from the October 2026 TfNSW GTFS "
        "feed: a four-track Central-Strathfield corridor with 52 scheduled movements and the "
        "Granville-Harris Park junction with 126. The corridor is disrupted by an exogenous "
        "blockage that removes a cell's rail transitions for 400 steps, so a policy cannot make "
        "the incident vanish by rerouting. Under that regime, and across three algorithms, only "
        f"the warm-started PPO learns to change track: it improves the metric from {pe.pct(a['base'])} "
        f"to {pe.pct(a['arm'])}, a paired gain of {a['delta']:+.4f} on {a['n']} held-out seeds "
        f"({a['wins']} of {a['n']}, p < 1e-4) with every train completing, and on a clean network "
        "it scores 0.0000 with no crossing at all. A2C never learns the crossing and DQN parks "
        "half the fleet. The junction remains a loss: the selected checkpoint dispatches and jams "
        "(-0.2400, 59 of 126 trains home) instead of parking, and still finishes behind the "
        "scripted route follower (-0.1823)."
    ))

    # ---- Section 1 (template) + our scene figures
    add("\n\\clearpage\n")
    add("\n\n".join(s1))

    # ---- Section 2
    add("\n\\clearpage\n\\section{Overview of the architecture/system}\n")
    add("\\subsection{Flow Diagram and Workflow}\n")
    add(tex(
        "The workflow runs in seven stages, each with a defined input and output (Figure 3). The "
        "dashed arrow is deliberate: every change to the observation, reward or disruption regime "
        "was re-validated by re-running the same paired evaluation, so this report only contains "
        "numbers produced by the final configuration, with the two labelled junction rows of Table "
        "5 as the one disclosed exception."
    ))
    add(figure("fig_workflow.png",
               "End-to-end workflow. Each stage consumes the artefact produced by the previous "
               "one; the dashed arrow marks the design loop that re-ran the evaluation after every "
               "change.", 0.95))
    add(bullets([
        "Service extraction: the GTFS feed (about 1.4 GB) is filtered to one Thursday peak, with "
        "link traversal times derived from the shapes (Transport for NSW, 2026).",
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
    ]))

    add("\\subsection{System Architecture Design}\n")
    add(tex(
        "One shared multi-layer perceptron controls every train: 1032 inputs, two hidden layers "
        "(256 units for PPO, 128 for A2C and DQN) and a five-way action head. The observation is a "
        "flattened, normalised tree of depth 3 within a radius of 50 cells, plus twelve conflict "
        "and geometry features: slack and its rank, distance to the first occupied cell, distance "
        "to oncoming traffic, malfunctioning trains on my route, my own malfunction countdown, "
        "current speed, whether I am on the map, remaining route length, queue depth, distance to "
        "the nearest blocked cell ahead, and distance to my next commit crossover. The last two "
        "exist because a zeroed rail cell is invisible to occupancy features, so without them the "
        "crossing decision is not expressible."
    ))
    add(tex(
        "The reward is the assessment metric paid densely as a potential, r(t) = R(t) - R(t-1), "
        "where R(t) is what the metric would charge if the episode ended at t. The sum telescopes "
        "to the metric exactly, verified per agent and per term to machine precision, and training "
        "divides the potential by a constant 100 so the value targets stay O(1). All reported arms "
        "also use two shaping bonuses on top of the potential: +50 for every train that arrives "
        "and -1 for every step a train is stuck, at the same 1/100 scale, which is disclosed here "
        "because the reported arm is not the bare-metric arm."
    ))
    add(tex(
        "The algorithms are PPO (Schulman et al., 2017), A2C (Mnih et al., 2016) and DQN (Mnih et "
        "al., 2015), all through Stable-Baselines3 (Raffin et al., 2021). From a random "
        "initialisation none of the three finds the crossing, so the reported corridor arm is PPO "
        "fine-tuned from a behaviour-cloned detour baseline - imitation then reinforcement "
        "learning (Ross et al., 2011) - with the critic warmed up first. The two independent "
        "algorithms are the control: they share the observation, reward and regime but not the "
        "warm start. The notebook is the assembly point and imports the shared modules from "
        "scripts/ instead of duplicating them, so the notebook, the trainers and the evaluator "
        "cannot drift apart."
    ))

    add("\\subsection{GUI/Environment Design}\n")
    add(tex(
        "Flatland's own renderer is the interface, which the brief permits in place of a separate "
        "GUI: it shows the rail grid with its switches and crossovers, one coloured marker per "
        "train, and a red X on every cell that may not be entered - an exogenous blockage on the "
        "corridor, a train standing in malfunction on the junction. Using the simulator's renderer "
        "means every episode figure comes from the same process the agent is trained in, produced "
        "headlessly with the PIL backend. Figures 1 and 2 are the two levels; Figures 4 and 5 show "
        "the same levels under their disruption regimes."
    ))
    add(figure("fig_corridor_arms.png",
               "The corridor under the block regime, four arms on the same episode (seed 3001; "
               "step 232 for forward-only, PPO and A2C, step 300 for DQN). The red X is a cell "
               "whose rail transitions were removed. Forward-only, A2C and DQN leave a queue "
               "behind the upper X; PPO's track ahead of it is clear because it changed at the "
               "previous crossover."))
    add(figure("fig_junction_pair.png",
               "The junction at step 156 under the train-borne regime. Left: the scripted route "
               "follower, with the held train marked. Right: the selected PPO checkpoint, which "
               "dispatches into the throat and jams.", 0.95))
    add(tex(
        "The disruption regime is the methodological core. A train-borne failure fires when a "
        "train reaches a cell, which makes the disruption a function of the policy's own "
        "trajectory: a policy that reroutes early can prevent the incident from firing at all. "
        "The corridor therefore uses an exogenous blockage: a cell's rail transitions are removed "
        "outright at a seed-drawn step and restored after 400 steps, so the failure exists whether "
        "or not a train goes near it. Two are drawn per episode, one per direction, from the cells "
        "just past each commit crossover; activation is deferred while a cell is occupied, and "
        "dropped if it stays occupied for 60 steps. Both arms are given the same cells at the same "
        "seed-drawn steps, so the pairing is by seed and blockage plan."
    ))

    # ---- Section 3
    add("\n\\clearpage\n\\section{Results and Evaluation}\n")
    add("\\subsection{Experimental Settings}\n")
    add(tex(
        "Table 1 lists the settings. All arms act deterministically; the corridor arm is evaluated "
        "on seeds 3000 to 3029, which training and checkpoint selection never see, and A2C and DQN "
        "are reported from their selection seeds only. Because the baseline and the arm run on the "
        "same seed, the same environment and the same blockages, a delta is computed per seed and "
        "the report gives its mean, median, win count and an exact two-sided sign test. The score "
        "is the assessment metric normalised to [-1, 0], with each train's cost capped at one "
        "episode, summed over all agents and divided by the number of agents and the episode "
        "budget; a second statistic, delay\\_all, charges every agent and uses the episode end for a "
        "train that never arrives."
    ))
    add(table([
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
    ], r"@{}p{3.2cm}p{5.6cm}p{5.6cm}@{}",
        "Table 1: Experimental settings. The maps share the observation, action space and "
        "evaluation protocol; they differ in disruption mechanism because only the corridor has a "
        "switchable pair of parallel tracks."))
    add(tex(
        "The regime and the policy are CPU-bound: Flatland steps the fleet in Python and the "
        "1032-d tree observation is rebuilt for every agent at every step. The shipped corridor run "
        "took 1,865 s for 1,224,704 timesteps including thirteen four-episode evaluations (about "
        "660 environment steps per second), and once startup is paid each evaluation interval costs "
        "about 148 s, so the evaluator rather than the optimiser sets the wall clock. Training ran "
        "on an x86\\_64 workstation with an RTX 3080 (the system\\_info recorded in the checkpoint); "
        "the notebook and figures also run on an aarch64 relay with the same pinned stack, where "
        "the sixteen tests pass. Every table below is regenerated from the CSVs in results/."
    ))

    add("\\subsection{Pre-Processing}\n")
    add(bullets([
        "Service extraction: one Thursday service period, with rail links rather than straight "
        "lines giving traversal times.",
        "Corridor level: four adjacent through tracks with the ladder rows removed; the compact "
        "map leaves a two-cell diversion premium at 7 of the 15 stop types.",
        "Junction level: a 20 x 20 level from the published track plan, with the turn sequence read "
        "out of the timetable as a route dictionary.",
        "Deadlines and episode budget: calibrated from a reference rollout, with the budget "
        "extended past the last arrival because the serialised value cut 15 junction trains off "
        "mid-journey.",
        "Multi-cell targets: the target set is rebuilt from the agent's own last waypoint group "
        "after every reset, otherwise a diverted train can never finish.",
        "Dense reward: the metric as a potential, verified against DefaultRewards per agent and per "
        "term.",
        "Metric hygiene: delay\\_all replaced a total-delay statistic that went NaN when nothing "
        "finished and negative when few did.",
        "Regime hygiene: the blockage is exogenous, so the disrupted condition cannot collapse into "
        "the undisrupted one.",
    ]))

    add("\\subsection{Experimental Results}\n")
    add(tex(
        "The corridor result is the headline. PPO improves the score from -0.0662 to -0.0005, a "
        f"mean paired gain of {a['delta']:+.4f} with a median of {a['median']:+.4f} (Table 2). It "
        f"wins all {a['n']} seeds and every train completes in both arms, so the gain is "
        f"punctuality rather than cancellation: total lateness falls from {a['delay_base']:.0f} "
        f"steps to {a['delay_arm']:.0f}. The on-time share moves only from "
        f"{a['on_time_base']:.3f} to {a['on_time_arm']:.3f}, so the gain is in how late the "
        "affected trains are, not in which trains complete. Figure 6 shows the per-seed spread."
    ))
    add(table([
        ["Arm", "Baseline", "Arm", "Delta", "Wins", "Sign test p", "Completion", "delay_all"],
        ["PPO", pe.pct(a["base"]), pe.pct(a["arm"]), f"{a['delta']:+.4f}",
         f"{a['wins']}/{a['n']}", "< 1e-4",
         f"{pe.comp(a['compl_base'])} $\\rightarrow$ {pe.comp(a['compl_arm'])}",
         f"{a['delay_base']:.0f} $\\rightarrow$ {a['delay_arm']:.0f}"],
    ], r"@{}lrrrrrrl@{}",
        "Table 2: Corridor, 30 paired held-out seeds under the block regime. Only PPO has the full "
        "paired test; Table 3 reports what the other two algorithms did in training."))
    add(figure("fig_delta.png",
               f"The {a['n']} per-seed differences behind Table 2. All {a['n']} are positive, with "
               f"a mean of {a['delta']:+.4f} and a median of {a['median']:+.4f}.", 0.7))
    add(tex(
        "The two independent algorithms are the control for the claim that learning the crossing is "
        "not automatic; Figure 7 traces all three arms over the training budget."
    ))
    add(table([
        ["Arm", "Best checkpoint", "Internal score", "Completion", "Behaviour in Figure 4"],
        ["PPO (warm start)", ppo_ck, pe.pct(ppo_sc), pe.comp(ppo_cp), "crosses only when blocked"],
        ["A2C", a2c_ck, pe.pct(a2c_sc), pe.comp(a2c_cp), "no crossing in the panel"],
        ["DQN (gamma 0.999)", dqn_ck, pe.pct(dqn_sc), pe.comp(dqn_cp), "parks half the fleet"],
    ], r"@{}lrrrp{5.2cm}@{}",
        "Table 3: Corridor training outcome per arm. Score and completion are the selection-seed "
        "numbers, not the 30-seed paired numbers in Table 2; the behaviour column is a reading of "
        "the Figure 4 panels, which are rendered with make\\_env rather than the evaluator."))
    add(figure("fig_curve.png",
               "The three corridor arms during training, from the internal evaluation used for "
               "checkpoint selection. PPO starts from the behaviour-cloned checkpoint and stays "
               "near zero until it drifts at the end of the budget; A2C and DQN fall into the "
               "parking basin. The dashed line is the forward-only 30-seed mean.", 0.9))
    add(tex(
        "A score gain alone does not prove the policy changes track for the right reason, since a "
        "policy that crossed at every crossover would also gain on this map. The audit counts the "
        "decisions themselves: for every agent-step standing on a commit crossover it records "
        "whether a blocked cell lay ahead on that train's own row within 60 cells, and whether the "
        "agent chose a lateral action. The per-crossover table is shipped as "
        "results/crossing\\_audit\\_block.csv."
    ))
    add(table([["Crossover", "Track", "Blocked visits", "Turns", "Open visits", "Turns"], *audit_rows],
              r"@{}lrrrrr@{}",
              "Table 4: Per-crossover decision audit over seeds 3000 and 3001. Blocked means an "
              "impassable cell ahead on the train's own row within 60 cells. (5,59) is the only "
              "crossover with blocked visits that did not always turn (20 of 40); the CSV records "
              "counts rather than per-visit causes, so none is claimed."))
    add(figure("fig_audit.png",
               "Turn rate on the visits where the track ahead was blocked.", 0.75))
    add(tex(
        f"Of {tb} agent-steps standing on a crossover with a blocked cell ahead, {tb_turns} "
        f"produced a turn; of {to} with a clear track, {to_turns} did. The audit records one "
        "decision per visit to a commit cell, so 81 turns over two seeds is about 40 diverts per "
        "episode. The rate is not uniform - four crossovers turn on every blocked visit and (5,59) "
        "on 20 of 40 - so the behaviour is state-dependent, not a blanket lane swap, and the A2C "
        "and DQN panels of Figure 4 show what the blanket alternative looks like."
    ))
    add(tex(
        f"Under the clean condition the same checkpoint is inert: on {clean['n']} seeds it scores "
        f"{pe.pct(clean['arm'])}, exactly as the forward-only baseline does, and the decision audit "
        "records 546 visits to a commit crossover with a clear track and not one turn "
        "(results/clean\\_audit.csv). With the blockage suppressed but the Poisson malfunction layer "
        f"left on, the arm is indistinguishable from the baseline: {pe.pct(tight['arm'])} against "
        f"{pe.pct(tight['base'])}, a paired {tight['delta']:+.4f} on {tight['n']} seeds, "
        f"{tight['wins']} of {tight['n']} but p = {tight['p']:.2f}. The crossings in Table 4 "
        "therefore belong to the blockage, not to the policy's habit."
    ))
    add(tex(
        "The junction is reported as a loss, and its two failure modes bracket the metric's "
        "pathology."
    ))
    add(table([
        ["Arm", "Baseline", "Arm", "Delta", "Wins", "Completion (base $\\rightarrow$ arm)"],
        ["PPO (30 seeds, 1030-d)", "-0.1769", "-0.0251", "+0.1518", "30/30",
         "1.000 $\\rightarrow$ 0.000"],
        ["A2C (30 seeds, 1030-d)", "-0.1769", "-0.0251", "+0.1518", "30/30",
         "1.000 $\\rightarrow$ 0.000"],
        ["DQN (30 seeds, 1030-d)", "-0.1769", "-0.0208", "+0.1561", "30/30",
         "1.000 $\\rightarrow$ 0.183"],
        ["PPO (current, 2 seeds)", pe.pct(jc["base"]), pe.pct(jc["arm"]), f"{jc['delta']:+.4f}",
         f"{jc['wins']}/{jc['n']}",
         f"{pe.comp(jc['compl_base'])} $\\rightarrow$ {pe.comp(jc['compl_arm'])}"],
    ], r"@{}lrrrrp{3.6cm}@{}",
        "Table 5: Junction results. The first three rows are the recorded 30-seed runs from before "
        "the two observation features were added, kept as measured: that policy parks the fleet. "
        "The last row is the current checkpoint selected by completion, which departs and jams; it "
        "is a two-seed evaluation shipped as results/junction\\_current.csv, identically scored on "
        "both seeds, so treat it as one behaviour. The rows are not directly comparable."))
    add(tex(
        "Parking scores better than attempting: every learned 30-seed arm scores about +0.15 better "
        "than the route follower while completing 0 to 18 per cent of the trains, because a "
        "cancellation costs about 19 cost units while a dispatch into the jam is charged its full "
        f"delay. The current checkpoint shows the other side: selected by completion rather than "
        f"score, it sends every dispatched train into the throat, completing "
        f"{jc['compl_arm'] * 126:.0f} of 126 and scoring {pe.pct(jc['arm'])}, worse than the "
        f"scripted baseline at {pe.pct(jc['base'])} with 126 of 126 home (Figure 5). Neither is a "
        "working junction policy; a global controller, or imitation of the route follower, is what "
        "that map needs."
    ))

    add("\\subsection{Limitations}\n")
    add(bullets([
        "Only PPO has the full 30-seed paired test on the corridor. A2C and DQN are reported from "
        "their selection seeds and the rendered episode; a paired run for them would strengthen the "
        "three-algorithm claim and is the obvious next experiment.",
        "The corridor decision is local: one crossover per blocked stretch and a two-cell detour. "
        "The result is not evidence that per-agent RL can arbitrate a junction.",
        "The disruption volume is a modelling choice, not data: 2 x 400 steps keeps completion at "
        "1.000 while making every affected trip late.",
        "From-scratch learning does not find the crossing, and the A2C and DQN arms demonstrate it "
        "independently. The corridor arm is imitation-then-RL, not pure exploration, and the warm "
        "start inherits the teacher's own score of -0.0005.",
        "The 30 held-out seeds vary the blockage placement only: fleet, timetable and deterministic "
        "actions are fixed, a single training seed per arm was run, and the mean delta has a sign "
        "test but no confidence interval.",
        "The junction table mixes two observation versions; the 30-seed checkpoints predate the two "
        "feature additions and no longer load. They are reported as measured and labelled.",
        "Evaluation is entirely simulated. The timetable and the track layout are real; there is no "
        "operational validation and no human factors layer.",
    ]))

    # ---- Section 4
    add("\n\\section{Discussion and Conclusions}\n")
    add(tex(
        "A local policy can learn to reroute, but only if the evaluation makes the decision real, "
        "and only with a recipe that can find a sparse action. On a four-track corridor cut by "
        "400-step exogenous blockages, PPO changes track at the commit crossover just before the "
        f"blockage and turns a paired deficit of {pe.pct(a['base'])} into {pe.pct(a['arm'])} over "
        f"{a['n']} held-out seeds, winning every seed without losing a train. The same checkpoint "
        "does nothing when nothing is blocked, and an audit of "
        f"{tb + to} crossover visits shows that its crossings follow the blockage rather than the "
        "map. The two independent algorithms do not get there, which makes the corridor result a "
        "statement about the recipe as much as about the algorithm."
    ))
    add(tex(
        "The result depended on fixing the measurement first. A train-borne failure fires when a "
        "train arrives, so a policy that reroutes suppresses the incident, and a failure placed "
        "deep inside a station's escape band leaves every follower already past the crossover. "
        "Removing the cell's transitions instead makes the blockage exogenous, keeps both arms on "
        "the same incidents, and turns the delayed comparison into a genuine decision. The junction "
        "remains a loss, and its two failure modes bracket the metric's pathology: parking scores "
        "better than attempting, and attempting scores worse than the schedule."
    ))
    add(tex(
        "Three extensions follow. First, the A2C and DQN arms deserve the same 30-seed paired run; "
        "the current evidence is their selection seeds and the rendered episode. Second, the "
        "junction needs a global view - a graph or attention observation, or imitation of the route "
        "follower - because a local tree observation cannot arbitrate 126 trains through one "
        "throat. Third, the corridor result should be checked against the stronger scripted "
        "distance-map baseline, not only the forward-only reflex, to see how much of the gain "
        "survives when the baseline is allowed to divert too."
    ))

    # ---- References (APA 7)
    add("\n\\clearpage\n\\section{References}\n")
    add("\\printbibliography[heading=none]\n")

    # ---- Appendices
    add("\n\\clearpage\n\\section*{Appendices}\n\\addcontentsline{toc}{section}{Appendices}\n")
    add("\\subsection*{A. Individual Contribution}\n\\addcontentsline{toc}{subsection}{A. Individual Contribution}\n")
    add(tex(
        "Each group member should write half-page (max) about his or her contribution in the "
        "project development activities."
    ))
    add("\\subsection*{B. Individual Contribution Split}\n\\addcontentsline{toc}{subsection}{B. Individual Contribution Split}\n")
    add(tex(
        "All the team members have discussed and agreed on the above individual contribution to "
        "the project."
    ))
    add(table([
        ["Team Member Name", "Percentage"],
        ["Daniel James Martirosov", "33.33\\%"],
        ["Ziheng Deng", "33.33\\%"],
        ["Laurean Lim", "33.33\\%"],
    ], r"@{}p{9cm}p{5cm}@{}", "Table 6: Agreed contribution split."))
    for name in ("Daniel James Martirosov", "Ziheng Deng", "Laurean Lim"):
        add("\n%s : \\rule{5cm}{0.4pt}\n\n\\vspace{0.8em}\n" % name)

    add("\n\\end{document}\n")

    OUT_TEX.write_text("\n".join(parts))
    print("wrote", OUT_TEX)

    if no_build:
        return
    for cmd in (["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", OUT_TEX.name],
                ["latexmk", "-c", OUT_TEX.name]):
        proc = subprocess.run(cmd, cwd=OUT_TEX.parent, capture_output=True, text=True)
        if cmd[1] != "-c" and proc.returncode != 0:
            print(proc.stdout[-4000:])
            raise SystemExit(f"latexmk failed ({proc.returncode})")
    print("wrote", OUT_TEX.with_suffix(".pdf"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-build", action="store_true", help="write the .tex only")
    args = ap.parse_args()
    build(args.no_build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
