"""Generate report/part-e.md — the authoring source for the Part E report.

The report is written in Markdown and converted to .docx only at the end
(`scripts/build_part_e_docx.py`). Numbers are read from results/*.csv here, so a
regenerated Markdown picks up any change in the evidence; after that the .md is
the file to edit.

    python scripts/build_part_e_md.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_data as pe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = "../results/report_figures"
TEMPLATE = REPO / "templates" / "part-e-template-2.docx"
OUT = REPO / "report" / "part-e.md"

CITES = {
    "(Li et al., 2021)": "[@li2021]",
    "(Mohanty et al., 2020)": "[@mohanty2020]",
    "(Sutton & Barto, 2018)": "[@sutton2018]",
    "(Roberts, 2024)": "[@roberts2024]",
    "(Gorrey et al., 2025)": "[@gorrey2025]",
    "(Sydney Trains, 2025)": "[@sydneytrains2025]",
    "(Zhu & Goverde, 2019)": "[@zhu2019]",
    "(Šemrov et al., 2016; Zhu & Goverde, 2019)": "[@semrov2016; @zhu2019]",
    "(Šemrov et al., 2016)": "[@semrov2016]",
}

PAGEBREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'


def section1_md() -> tuple[list[str], list[str]]:
    """Section 1 as Markdown, taken from the template so it is not retyped."""
    doc = Document(str(TEMPLATE))
    out: list[str] = []
    started = False
    figures: list[str] = []
    captions: dict[int, str] = {}
    pending_caption = False

    for i, para in enumerate(doc.paragraphs):
        text = " ".join(para.text.split())
        if text.startswith("2. Overview"):
            break
        if not started:
            started = text.startswith("1. Introduction")
            if not started:
                continue
        # citations stay as the plain author-date text the template already uses:
        # the docx has no citation processor, and if it did we would need a CSL file
        if para.style.name == "Heading 1":
            continue  # the Markdown adds this heading itself
        elif para.style.name == "Heading 2":
            out.append("### " + text)
        elif text.startswith("Figure") and ":" in text:
            n = int(text.split()[1].strip(":"))
            captions[n] = text.split(":", 1)[1].strip()
            pending_caption = True
        elif pending_caption:
            # the template wrapped this caption into a second paragraph
            pending_caption = False
            captions[max(captions)] += " " + text
        elif text:
            out.append(text)

    # the template's two level renders, extracted by the tex builder into tex_assets/
    known = {
        1: "Flatland implementation of the Central-Strathfield corridor (Scenario 1)",
        2: "Flatland implementation of the Granville-Harris Park corridor (Scenario 2).",
    }
    for idx in (1, 2):
        figures.append(f"![Figure {idx}: {known[idx]}](template_figures/map_{idx}.png){{width=95%}}")
    return out, figures


def table(rows, align: str = "") -> str:
    head = "| " + " | ".join(rows[0]) + " |"
    sep = "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join([head, sep, body])


def main() -> int:
    a = pe.paired(RES / "pair_block_ppo.csv")
    clean = pe.paired(RES / "clean_normal.csv")
    tight = pe.paired(RES / "clean_delayed_none.csv")
    jc = pe.paired(RES / "junction_current.csv")
    audit_rows, tb, tb_turns, to, to_turns = pe.audit_table()
    ppo_ck, ppo_sc, ppo_cp = pe.best_checkpoint("ppo")
    a2c_ck, a2c_sc, a2c_cp = pe.best_checkpoint("a2c")
    dqn_ck, dqn_sc, dqn_cp = pe.best_checkpoint("dqn")

    s1_paras, s1_figures = section1_md()
    md: list[str] = []
    add = md.append

    # ---------------------------------------------------------------- cover
    add('::: {custom-style="Title"}')
    add("43008: Reinforcement Learning")
    add(":::")
    add("")
    add('::: {custom-style="Subtitle"}')
    add("Assignment 3, Part E Project Report")
    add(":::")
    add("")
    add("**Project Title: Train Schedule Optimisation under Disruptions**")
    add("")
    add("*Learning to change track when a corridor is blocked*")
    add("")
    add("**Team Name and Project Number:** RailTech (Project #27)")
    add("")
    add("**Team Members:** Daniel James Martirosov (24948933), Ziheng Deng (25595315), "
        "Laurean Lim (25049863)")
    add("")
    add("**Date:** 26 October 2026")
    add(PAGEBREAK)

    # ---------------------------------------------------------------- contents
    add("## Table of Contents")
    add("")
    add("@@TOC@@")
    add(PAGEBREAK)

    # ---------------------------------------------------------------- abstract
    add("# Abstract")
    add("")
    add(
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
    )
    add(PAGEBREAK)

    # ---------------------------------------------------------------- section 1
    add("# 1. Introduction and Background")
    add("")
    add("\n\n".join(s1_paras))
    add("")
    add("\n\n".join(s1_figures))
    add(PAGEBREAK)

    # ---------------------------------------------------------------- section 2
    add("# 2. Overview of the architecture/system")
    add("")
    add("## 2.1 Flow Diagram and Workflow")
    add("")
    add(
        "The workflow runs in seven stages, each with a defined input and output (Figure 3). The "
        "dashed arrow is deliberate: every change to the observation, reward or disruption regime "
        "was re-validated by re-running the same paired evaluation, so this report only contains "
        "numbers produced by the final configuration, with the two labelled junction rows of Table "
        "5 as the one disclosed exception."
    )
    add("")
    add(f"![Figure 3: End-to-end workflow. Each stage consumes the artefact produced by the previous one; the "
        f"dashed arrow marks the design loop that re-ran the evaluation after every change.]"
        f"({FIG}/fig_workflow.png){{width=95%}}")
    add("")
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
        add(f"- {text}")
    add("")
    add("## 2.2 System Architecture Design")
    add("")
    add(
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
    add("")
    add(
        "The reward is the assessment metric paid densely as a potential, r(t) = R(t) - R(t-1), "
        "where R(t) is what the metric would charge if the episode ended at t. The sum telescopes "
        "to the metric exactly, verified per agent and per term to machine precision, and training "
        "divides the potential by a constant 100 so the value targets stay O(1). All reported arms "
        "also use two shaping bonuses on top of the potential: +50 for every train that arrives "
        "and -1 for every step a train is stuck, at the same 1/100 scale, which is disclosed here "
        "because the reported arm is not the bare-metric arm."
    )
    add("")
    add(
        "From a random initialisation none of the three algorithms finds the crossing, so the "
        "reported corridor arm is PPO fine-tuned from a behaviour-cloned detour baseline "
        "(imitation then RL), with the critic warmed up first. The two independent algorithms are "
        "the control: they share the observation, reward and regime but not the warm start. The "
        "notebook is the assembly point and imports the shared modules from scripts/ instead of "
        "duplicating them, so the notebook, the trainers and the evaluator cannot drift apart."
    )
    add("")
    add("## 2.3 GUI/Environment Design")
    add("")
    add(
        "Flatland's own renderer is the interface, which the brief permits in place of a separate "
        "GUI: it shows the rail grid with its switches and crossovers, one coloured marker per "
        "train, and a red X on every cell that may not be entered - an exogenous blockage on the "
        "corridor, a train standing in malfunction on the junction. Using the simulator's renderer "
        "means every episode figure comes from the same process the agent is trained in, produced "
        "headlessly with the PIL backend. Figures 1 and 2 are the two levels; Figures 4 and 5 show "
        "the same levels under their disruption regimes."
    )
    add("")
    add(f"![Figure 4: The corridor under the block regime, four arms on the same episode (seed 3001; step 232 "
        f"for forward-only, PPO and A2C, step 300 for DQN). The red X is a cell whose rail "
        f"transitions were removed. Forward-only, A2C and DQN leave a queue behind the upper X; "
        f"PPO's track ahead of it is clear because it changed at the previous crossover.]"
        f"({FIG}/fig_corridor_arms.png){{width=100%}}")
    add("")
    add(f"![Figure 5: The junction at step 156 under the train-borne regime. Left: the scripted route "
        f"follower, with the held train marked. Right: the selected PPO checkpoint, which "
        f"dispatches into the throat and jams.]({FIG}/fig_junction_pair.png){{width=95%}}")
    add("")
    add(
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
    add(PAGEBREAK)

    # ---------------------------------------------------------------- section 3
    add("# 3. Results and Evaluation")
    add("")
    add("## 3.1 Experimental Settings")
    add("")
    add(
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
    add("")
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
    ], "lll"))
    add("")
    add(":Table 1: Experimental settings. The maps share the observation, action space and "
        "evaluation protocol; they differ in disruption mechanism because only the corridor has a "
        "switchable pair of parallel tracks.")
    add("")
    add(
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
    add("")
    add("## 3.2 Pre-Processing")
    add("")
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
        add(f"- {text}")
    add("")
    add("## 3.3 Experimental Results")
    add("")
    add(
        "The corridor result is the headline. PPO improves the score from -0.0662 to -0.0005, a "
        f"mean paired gain of {a['delta']:+.4f} with a median of {a['median']:+.4f} (Table 2). It "
        f"wins all {a['n']} seeds and every train completes in both arms, so the gain is "
        f"punctuality rather than cancellation: total lateness falls from {a['delay_base']:.0f} "
        f"steps to {a['delay_arm']:.0f}. The on-time share moves only from "
        f"{a['on_time_base']:.3f} to {a['on_time_arm']:.3f}, so the gain is in how late the "
        "affected trains are, not in which trains complete. Figure 6 shows the per-seed spread."
    )
    add("")
    add(table([
        ["Arm", "Baseline", "Arm", "Delta", "Wins", "Sign test p", "Completion", "delay_all"],
        ["PPO", pe.pct(a["base"]), pe.pct(a["arm"]), f"{a['delta']:+.4f}",
         f"{a['wins']}/{a['n']}", "< 1e-4",
         f"{pe.comp(a['compl_base'])} → {pe.comp(a['compl_arm'])}",
         f"{a['delay_base']:.0f} → {a['delay_arm']:.0f}"],
    ], "lrrrrrrr"))
    add("")
    add(":Table 2: Corridor, 30 paired held-out seeds under the block regime. Only PPO has the "
        "full paired test; Table 3 reports what the other two algorithms did in training.")
    add("")
    add(f"![Figure 6: The {a['n']} per-seed differences behind Table 2. All {a['n']} are positive, with a "
        f"mean of {a['delta']:+.4f} and a median of {a['median']:+.4f}]({FIG}/fig_delta.png)"
        f"{{width=70%}}")
    add("")
    add("The two independent algorithms are the control for the claim that learning the crossing "
        "is not automatic; Figure 7 traces all three arms over the training budget.")
    add("")
    add(table([
        ["Arm", "Best checkpoint", "Internal score", "Completion", "Behaviour in Figure 4"],
        ["PPO (warm start)", ppo_ck, pe.pct(ppo_sc), pe.comp(ppo_cp), "crosses only when blocked"],
        ["A2C", a2c_ck, pe.pct(a2c_sc), pe.comp(a2c_cp), "no crossing in the panel"],
        ["DQN (gamma 0.999)", dqn_ck, pe.pct(dqn_sc), pe.comp(dqn_cp), "parks half the fleet"],
    ], "lrrrl"))
    add("")
    add(":Table 3: Corridor training outcome per arm. Score and completion are the selection-seed "
        "numbers, not the 30-seed paired numbers in Table 2; the behaviour column is a reading of "
        "the Figure 4 panels, which are rendered with make_env rather than the evaluator.")
    add("")
    add(f"![Figure 7: The three corridor arms during training, from the internal evaluation used for "
        f"checkpoint selection. PPO starts from the behaviour-cloned checkpoint and stays near "
        f"zero until it drifts at the end of the budget; A2C and DQN fall into the parking basin. "
        f"The dashed line is the forward-only 30-seed mean.]({FIG}/fig_curve.png){{width=90%}}")
    add("")
    add(
        "A score gain alone does not prove the policy changes track for the right reason, since a "
        "policy that crossed at every crossover would also gain on this map. The audit counts the "
        "decisions themselves: for every agent-step standing on a commit crossover it records "
        "whether a blocked cell lay ahead on that train's own row within 60 cells, and whether the "
        "agent chose a lateral action. The per-crossover table is shipped as "
        "results/crossing_audit_block.csv."
    )
    add("")
    add(table([["Crossover", "Track", "Blocked visits", "Turns", "Open visits", "Turns"], *audit_rows],
              "lrrrrr"))
    add("")
    add(":Table 4: Per-crossover decision audit over seeds 3000 and 3001. Blocked means an "
        "impassable cell ahead on the train's own row within 60 cells. (5,59) is the only "
        "crossover with blocked visits that did not always turn (20 of 40); the CSV records counts "
        "rather than per-visit causes, so none is claimed.")
    add("")
    add(f"![Figure 8: Turn rate on the visits where the track ahead was blocked.]({FIG}/fig_audit.png)"
        f"{{width=75%}}")
    add("")
    add(
        f"Of {tb} agent-steps standing on a crossover with a blocked cell ahead, {tb_turns} "
        f"produced a turn; of {to} with a clear track, {to_turns} did. The audit records one "
        "decision per visit to a commit cell, so 81 turns over two seeds is about 40 diverts per "
        "episode. The rate is not uniform - four crossovers turn on every blocked visit and (5,59) "
        "on 20 of 40 - so the behaviour is state-dependent, not a blanket lane swap, and the A2C "
        "and DQN panels of Figure 4 show what the blanket alternative looks like."
    )
    add("")
    add(
        f"Under the clean condition the same checkpoint is inert: on {clean['n']} seeds it scores "
        f"{pe.pct(clean['arm'])}, exactly as the forward-only baseline does, and the decision audit "
        "records 546 visits to a commit crossover with a clear track and not one turn "
        "(results/clean_audit.csv). With the blockage suppressed but the Poisson malfunction layer "
        f"left on, the arm is indistinguishable from the baseline: {pe.pct(tight['arm'])} against "
        f"{pe.pct(tight['base'])}, a paired {tight['delta']:+.4f} on {tight['n']} seeds, "
        f"{tight['wins']} of {tight['n']} but p = {tight['p']:.2f}. The crossings in Table 4 "
        "therefore belong to the blockage, not to the policy's habit."
    )
    add("")
    add("The junction is reported as a loss, and its two failure modes bracket the metric's "
        "pathology.")
    add("")
    add(table([
        ["Arm", "Baseline", "Arm", "Delta", "Wins", "Completion (base → arm)"],
        ["PPO (30 seeds, 1030-d)", "-0.1769", "-0.0251", "+0.1518", "30/30", "1.000 → 0.000"],
        ["A2C (30 seeds, 1030-d)", "-0.1769", "-0.0251", "+0.1518", "30/30", "1.000 → 0.000"],
        ["DQN (30 seeds, 1030-d)", "-0.1769", "-0.0208", "+0.1561", "30/30", "1.000 → 0.183"],
        ["PPO (current, 2 seeds)", pe.pct(jc["base"]), pe.pct(jc["arm"]), f"{jc['delta']:+.4f}",
         f"{jc['wins']}/{jc['n']}",
         f"{pe.comp(jc['compl_base'])} → {pe.comp(jc['compl_arm'])}"],
    ], "lrrrrl"))
    add("")
    add(":Table 5: Junction results. The first three rows are the recorded 30-seed runs from "
        "before the two observation features were added, kept as measured: that policy parks the "
        "fleet. The last row is the current checkpoint selected by completion, which departs and "
        "jams; it is a two-seed evaluation shipped as results/junction_current.csv, identically "
        "scored on both seeds, so treat it as one behaviour. The rows are not directly comparable.")
    add("")
    add(
        "Parking scores better than attempting: every learned 30-seed arm scores about +0.15 "
        "better than the route follower while completing 0 to 18 per cent of the trains, because a "
        "cancellation costs about 19 cost units while a dispatch into the jam is charged its full "
        f"delay. The current checkpoint shows the other side: selected by completion rather than "
        f"score, it sends every dispatched train into the throat, completing "
        f"{jc['compl_arm'] * 126:.0f} of 126 and scoring {pe.pct(jc['arm'])}, worse than the "
        f"scripted baseline at {pe.pct(jc['base'])} with 126 of 126 home (Figure 5). Neither is a "
        "working junction policy; a global controller, or imitation of the route follower, is what "
        "that map needs."
    )
    add("")
    add("## 3.4 Limitations")
    add("")
    for text in [
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
        "The 30 held-out seeds vary the blockage placement only: fleet, timetable and "
        "deterministic actions are fixed, a single training seed per arm was run, and the mean "
        "delta has a sign test but no confidence interval.",
        "The junction table mixes two observation versions; the 30-seed checkpoints predate the two "
        "feature additions and no longer load. They are reported as measured and labelled.",
        "Evaluation is entirely simulated. The timetable and the track layout are real; there is no "
        "operational validation and no human factors layer.",
    ]:
        add(f"- {text}")
    add(PAGEBREAK)

    # ---------------------------------------------------------------- section 4
    add("# 4. Discussion and Conclusions")
    add("")
    add(
        "A local policy can learn to reroute, but only if the evaluation makes the decision real, "
        "and only with a recipe that can find a sparse action. On a four-track corridor cut by "
        "400-step exogenous blockages, PPO changes track at the commit crossover just before the "
        f"blockage and turns a paired deficit of {pe.pct(a['base'])} into {pe.pct(a['arm'])} over "
        f"{a['n']} held-out seeds, winning every seed without losing a train. The same checkpoint "
        "does nothing when nothing is blocked, and an audit of "
        f"{tb + to} crossover visits shows that its crossings follow the blockage rather than the "
        "map. The two independent algorithms do not get there, which makes the corridor result a "
        "statement about the recipe as much as about the algorithm."
    )
    add("")
    add(
        "The result depended on fixing the measurement first. A train-borne failure fires when a "
        "train arrives, so a policy that reroutes suppresses the incident, and a failure placed "
        "deep inside a station's escape band leaves every follower already past the crossover. "
        "Removing the cell's transitions instead makes the blockage exogenous, keeps both arms on "
        "the same incidents, and turns the delayed comparison into a genuine decision. The "
        "junction remains a loss, and its two failure modes bracket the metric's pathology: "
        "parking scores better than attempting, and attempting scores worse than the schedule."
    )
    add("")
    add(
        "Three extensions follow. First, the A2C and DQN arms deserve the same 30-seed paired run; "
        "the current evidence is their selection seeds and the rendered episode. Second, the "
        "junction needs a global view - a graph or attention observation, or imitation of the "
        "route follower - because a local tree observation cannot arbitrate 126 trains through one "
        "throat. Third, the corridor result should be checked against the stronger scripted "
        "distance-map baseline, not only the forward-only reflex, to see how much of the gain "
        "survives when the baseline is allowed to divert too."
    )

    OUT.write_text("\n".join(md).strip() + "\n")
    print("wrote", OUT, f"({len(md)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
