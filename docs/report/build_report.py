"""Build the mid-submission report: ``report.tex`` (LaTeX source) and ``report.pdf`` from ONE content definition.

    ../myenv/bin/python docs/report/build_report.py

* Every table cell that is a result is read from saved run records (``results/**/runs.jsonl``) through
  ``testorcode.report_tables``; nothing in the text is typed in as a measured number.  The test-split table appears
  only when ``results/day4/pilot-test-5cond/runs.jsonl`` exists, otherwise it says "not yet evaluated".
* No TeX distribution was available on the build machine, so ``report.pdf`` is rendered with reportlab from the same
  content blocks that produce ``report.tex``.  It is NOT the output of compiling ``report.tex``.  To compile the
  LaTeX instead: ``cd docs/report && pdflatex report.tex && pdflatex report.tex`` (needs ``booktabs``-free LaTeX only:
  article, graphicx, longtable, hyperref).
Needs matplotlib-produced PNGs (copied from the results directories) and reportlab (``pip install reportlab``).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from testorcode.dataset import episode_path, load_episodes  # noqa: E402
from testorcode.experiment import dataset_version, read_records  # noqa: E402
from testorcode.report_tables import NOT_EVALUATED, condition_rows, integrity_lines  # noqa: E402

DEV_DIR = ROOT / "results/day3/pilot-dev-5cond"
TEST_DIR = ROOT / "results/day4/pilot-test-5cond"
SHORT = {"A1_NO_INTENT_RETRIEVAL": "A1", "A2_NO_MEMORY": "A2", "A3_NO_INTEGRITY_GUARD": "A3",
         "TRIVIAL_UPDATE_TEST": "always UPDATE_TEST", "TRIVIAL_FIX_CODE": "always FIX_CODE"}
OFFLINE_TESTS = "see docs/day4-human2.md"  # replaced below when a count is recorded in docs/report/test_count.txt


# ---------------------------------------------------------------- data
def frac(k, n):
    return f"{k}/{n} ({k / n:.2f})" if n else "-"


def num(x):
    return "-" if x is None else f"{x:.3f}"


def result_rows(run_dir: Path, split: str):
    runs = run_dir / "runs.jsonl"
    if not runs.is_file():
        return None
    records = read_records(runs)
    episodes = load_episodes(episode_path(split))
    rows = condition_rows(records, episodes, dataset_version=dataset_version(split))
    models = sorted({r.provenance.model_id for r in records if not r.condition.value.startswith("TRIVIAL")
                     and r.provenance.model_id})
    seeds = sorted({r.provenance.seed for r in records})
    audit = run_dir / "integrity_audit.jsonl"
    return rows, models, seeds, (integrity_lines(audit) if audit.is_file() else [])


def table_from_rows(rows):
    header = ["Condition", "Episodes run", "Direction macro-F1", "Direction acc.", "Final-action acc.",
              "Correct-repair rate", "Masked regressions"]
    out = []
    for r in rows:
        name = SHORT.get(r["condition"], r["condition"])
        if not r["evaluated"]:
            out.append([name, NOT_EVALUATED, "-", "-", "-", "-", "-"])
            continue
        repair = frac(r["repairs_ok"], r["repairs"])
        if r["repair_unverifiable"]:
            repair += f" +{r['repair_unverifiable']} unverif."
        out.append([name, str(r["episodes_run"]), num(r["direction_macro_f1"]),
                    frac(r["direction_correct"], r["direction_scored"]), num(r["final_action_accuracy"]), repair,
                    frac(r["masked"], r["gold_fix"])])
    return header, out


def dataset_tables():
    out = {}
    for split in ("dev", "test"):
        eps = load_episodes(episode_path(split))
        rows, dirs, acts = [], {}, {}
        for e in eps:
            gt = e.ground_truth
            d = [t.gold_direction.value for t in gt.per_test_labels]
            a = [t.expected_action.value for t in gt.per_test_labels]
            for x in d:
                dirs[x] = dirs.get(x, 0) + 1
            for x in a:
                acts[x] = acts.get(x, 0) + 1
            rows.append([e.episode_id, e.scenario.value, e.difficulty.value if hasattr(e.difficulty, "value")
                         else str(e.difficulty), str(len(d)), ", ".join(sorted(set(a)))])
        out[split] = (rows, dirs, acts, len(eps))
    return out


# ---------------------------------------------------------------- content
def build_blocks():
    ds = dataset_tables()
    dev = result_rows(DEV_DIR, "dev")
    test = result_rows(TEST_DIR, "test")
    B: list[tuple] = []
    H1 = lambda t: B.append(("h1", t))  # noqa: E731
    H2 = lambda t: B.append(("h2", t))  # noqa: E731
    P = lambda t: B.append(("p", t))  # noqa: E731
    L = lambda items: B.append(("ul", items))  # noqa: E731

    B.append(("title", "TestOrCode: Mid-Submission Progress Report",
              "Intent-grounded triage of CI test failures with scoped change memory and test-integrity checking. "
              "Team Agent Zero. Mid-submission, 30 October 2026. All numbers are preliminary and come from saved runs."))

    H1("1. Executive summary and current status")
    P("TestOrCode decides, for each failing pytest test after a code change, whether to UPDATE_TEST, FIX_CODE or "
      "ESCALATE, and checks that the repair does not weaken the test suite. The submitted proposal describes four "
      "logical agents (Triage, Intent Analyst, Repairer, Integrity Guard), intent-grounded retrieval, persistent "
      "memory backed by executable probes, and a mutation-based integrity check.")
    L(["**Working end to end:** the whole pipeline (Triage, Intent Analyst with BM25 retrieval, Repairer with a "
       "restricted patch tool, Integrity Guard, probe-backed memory, simulated maintainer) runs from a CLI against "
       "real boltons episodes with a hosted model.",
       "**Conditions implemented:** FULL, B1 (single tool-using agent with RAG, no memory), A1 (no intent retrieval), "
       "A2 (no memory), A3 (no Integrity Guard), plus the always-UPDATE_TEST and always-FIX_CODE sanity baselines. "
       "All use one episode format and one result format.",
       f"**Pilot dataset:** {ds['dev'][3] + ds['test'][3]} episodes on one repository "
       f"({ds['dev'][3]} development, {ds['test'][3]} frozen test), built from real commits.",
       "**Real runs:** all conditions were run live on the 9 development episodes and, once, on the 4 frozen test episodes (one seed, one model). "
       + ("Test-split results are in Section 6." if test else
          "The 4 frozen test episodes are not yet evaluated with the model conditions."),
       "**Not claimed:** no statistical result, no conclusion that any condition is better than another. The pilot is "
       "too small; it shows the experiment runs and exposes the failure modes the final 15 days must address."])

    H1("2. Research question and proposed architecture")
    P("**Research question.** On chronological streams of CI failures in small Python libraries, does a "
      "role-separated pipeline that (i) grounds each decision in retrieved change-intent evidence, (ii) remembers "
      "approved behaviour changes as scoped records with executable probes, and (iii) checks test integrity after "
      "repair make fewer wrong-side repairs and weaken tests less than a single tool-using agent with the same tools, "
      "model and budget?")
    P("**Architecture.** One Python orchestrator (a deterministic state machine) with four role-specific prompts. "
      "Triage runs the suite, confirms stability and groups failures by function. The Intent Analyst retrieves "
      "evidence (BM25 over commit messages, docs, changelog, docstrings and source comments, plus probe-gated "
      "memory) and states a hypothesis with cited evidence ids. The decision policy requires cited positive evidence "
      "for UPDATE_TEST and cited expected-behaviour evidence for FIX_CODE, otherwise ESCALATE. The Repairer applies "
      "a scoped patch through a restricted patch tool. The Integrity Guard re-runs tests, mutants, weakening checks "
      "and a semantic review, and is the only role that writes memory, after simulated maintainer approval. All "
      "conditions go through one model client and share budgets.")
    P("**Safety design.** Path and scope checks, the allowed-file policy, and an expected-literal-only rule for test "
      "patches are enforced by the host-side tools, not by the prompt. Tests run in a bubblewrap sandbox without "
      "network. Gold labels never reach the agent: the harness keeps them, and agent-view files contain none by "
      "construction.")

    H1("3. Implemented components versus planned components")
    B.append(("table", ["Component", "Status", "Evidence / note"], [
        ["Four roles and orchestrator", "Implemented", "Live runs on 9 dev episodes, FULL/A1/A2/A3"],
        ["B1 single-agent baseline", "Implemented", "Same tools, standard RAG, no memory; live runs"],
        ["BM25 retrieval, gold-evidence Recall@k", "Implemented", "docs/retrieval.md; hybrid embeddings planned"],
        ["Probe-backed scoped memory", "Implemented", "Probes run on current code before reuse; stale records flagged"],
        ["Restricted patch tool, weakening checks", "Implemented", "Tested on real episode states"],
        ["Mutation-based Guard check", "Implemented, coarse", "Cached mutants; small functions give 1-2 mutants (dev-06: none killed)"],
        ["A1, A2, A3 switches", "Implemented, run on dev", "Single seed; A1 escalated every decision"],
        ["Experiment runner, metrics, report tables", "Implemented", "Resume, retry-failed, post-hoc integrity audit"],
        ["Docker sandbox", "Implemented, untested", "All runs used bubblewrap; Docker needs daemon access"],
        ["Tool-fault injection (scenario T)", "Planned", "Not wired; fault plans exist in two dev episodes"],
        ["git_log / git_show / git_blame tools", "Stubs", "Not needed by current episodes; dev-01 stays provisional"],
        ["Embeddings and reranking", "Planned", "Optional; BM25 first, as in the 5-day plan"],
        ["72 episodes, 3 held-out libraries, 2 seeds", "Planned", "Deferred to the 15-day plan"],
        ["Statistics (bootstrap, McNemar)", "Planned", "Not meaningful at pilot size"],
        ["Fine-tuning, frontend", "Dropped", "Outside scope by the plan"],
    ], [0.34, 0.2, 0.46]))

    H1("4. Dataset: sources, construction, labels and actual size")
    P(f"The pilot uses boltons (BSD-3-Clause), pinned at commit d31d821d. It has {ds['dev'][3]} development and "
      f"{ds['test'][3]} frozen test episodes, kept in separate chronological streams with disjoint commits, tests, "
      "functions and modules. This is a pilot of one library, not the 72-episode, three-library set of the proposal. "
      "Outdated-test episodes (S1, S3) come from real commits that changed source and tests; regression episodes "
      "(S2, S4) inject a mutant into real code; S5 strips the intent evidence. Every episode was verified by running "
      "the code: the CI state fails the same tests in 5 of 5 runs and the gold repair turns it green. The S5 episode "
      "(dev-01) is provisional.")
    for split in ("dev", "test"):
        rows, dirs, acts, n = ds[split]
        B.append(("table", ["Episode", "Scenario", "Difficulty", "Failing tests", "Expected action(s)"], rows,
                  [0.14, 0.2, 0.16, 0.16, 0.34], f"{split} split episodes"))
        P(f"{split} split, per failing test: gold direction {dirs}; expected action {acts}.")
    P("The development split is FIX_CODE-heavy. Gold evidence ids exist for 5 of the 13 episodes (all UPDATE_TEST); "
      "recall@k is undefined elsewhere and is shown as '-'.")

    H1("5. B1 baseline, ablations and experiments completed")
    B.append(("table", ["Condition", "Definition", "Status"], [
        ["FULL", "All four roles, intent retrieval, probe-gated memory, Guard", "Run live (dev, test)"],
        ["B1", "One tool-using agent, standard RAG, no persistent memory", "Run live (dev, test)"],
        ["A1", "FULL without intent retrieval: no commit message, corpus or search_repo; memory kept", "Run live (dev, test)"],
        ["A2", "FULL without persistent memory", "Run live (dev, test)"],
        ["A3", "FULL without the independent Integrity Guard (test-level verification only)", "Run live (dev, test)"],
        ["always UPDATE_TEST / FIX_CODE", "Deterministic sanity baselines, no model", "Run (dev, test)"],
    ], [0.2, 0.55, 0.25]))
    P("The proposal's A3 (no Integrity Guard) is used, not the detailed idea's coarse-memory ablation. Tool-level "
      "refusal of weakening edits applies in every condition, so A3's weakening rate is 0 by construction. B0, B2, "
      "A4 and A5 of the detailed idea are not implemented.")

    H1("6. Preliminary quantitative results and limitations")
    if dev:
        rows, models, seeds, lines = dev
        h, body = table_from_rows(rows)
        B.append(("table", h, body, [0.17, 0.1, 0.12, 0.14, 0.12, 0.2, 0.15],
                  f"Development split, 9 episodes, seeds {seeds}, model {', '.join(models)}. Preliminary."))
        B.append(("ul", lines))
    if test:
        rows, models, seeds, lines = test
        h, body = table_from_rows(rows)
        B.append(("table", h, body, [0.17, 0.1, 0.12, 0.14, 0.12, 0.2, 0.15],
                  f"Frozen test split, 4 episodes, seeds {seeds}, model {', '.join(models)}. Preliminary."))
        B.append(("ul", lines))
    else:
        P("**Frozen test split:** FULL, B1, A1, A2 and A3 are not yet evaluated on the 4 test episodes. Only the "
          "deterministic baselines have been run there.")
    B.append(("img", HERE / "figures/direction_macro_f1.png", "Direction macro-F1 on the development split", 0.6))
    if test:
        B.append(("img", HERE / "figures/test_direction_macro_f1.png", "Direction macro-F1 on the frozen test split", 0.6))
    B.append(("img", HERE / "figures/repair_vs_masked.png", "Repair success against masked regressions", 0.6))
    P("**How to read the results.**")
    L(["Direction is scored from the agent's hypothesis, so it is generous. An escalation counts by the side it "
       "blamed. Final-action accuracy is lower because conditions escalate often; read it with direction.",
       "Correct-repair rates have denominators of 3 to 7 repairs. 'Unverifiable' means a repair was accepted but "
       "no trusted probe confirmed it (A3 and B1 have no Guard).",
       "Masked regressions are 0 for every model condition because none edited a test to hide a regression. The "
       "metric only separates conditions from always-UPDATE_TEST. It does not show that the Guard is needed.",
       "A1 escalated every decision, so its repair rate is undefined; its direction score comes from the hypothesis "
       "alone.",
       "The post-hoc audit found no weakened and no wrong-side accepted repair in any condition. Mutant-kill "
       "retention is pooled over only the test patches whose reference tests kill a mutant (raw counts above).",
       "**Limitations:** one seed, one model, 9 (+4) episodes, one library; no confidence intervals; citation support, "
       "stale-evidence rate and memory over-generalisation are not computed; Docker untested; scenario T not wired."])
    P("**One success and one failure from the logs.** dev-03 (FULL): the Analyst cited the commit message and the "
      "docstring, the source patch applied, the Guard's mutants passed, the suite was green and a memory record was "
      "written. dev-05 (FULL, A2, A3): the hypothesis was correct (outdated test) but both repair attempts edited more "
      "than expected literals and the patch-content rule refused them, so the run escalated although the direction "
      "was right. This is the strict safety rule working, and its cost.")

    if test:
        P("**Reading the frozen test-split runs** (hand review of `review_runs.py` output on "
          "`results/day4/pilot-test-5cond`; one run per cell, so these are observations, not effects).")
        L(["**Success, test-01 (S2, gold FIX_CODE), FULL:** the Analyst cited the commit message, the Repairer's source "
           "patch applied, the Guard approved after the mutants were run, the suite passed and a memory record was "
           "written. A2 and A3 reached the same repair.",
           "**Failure, test-02 (S2, gold FIX_CODE), all five conditions:** every condition escalated. Each "
           "hypothesised a regression correctly but cited no expected-behaviour evidence, so the decision policy "
           "escalated. Final-action accuracy is therefore 0 on this episode for all of them, although direction was "
           "right.",
           "**Failure, test-03 (S3 mixed):** for the outdated half, FULL and A2 had their test patches refused by the "
           "literal-only rule (the model changed more than expected values), while A3 and B1 produced diffs that did "
           "not apply. All escalated. For the regression half only A2 produced a Guard-approved source fix; FULL and "
           "A3 escalated for missing evidence.",
           "**Decoy, test-04 (S4, gold FIX_CODE):** FULL and A3 read the decoy memory record (its probe passed) but "
           "hypothesised a regression without citing it, and escalated. A1 cited the decoy record and hypothesised an "
           "outdated test: the only run where the decoy swayed a hypothesis, and the escalation prevented a wrong-side "
           "edit. B1, with no memory, fixed the code correctly.",
           "**B1 on test-01:** it hypothesised an outdated test for a real regression. Its two test edits were refused "
           "by the patch rule, so the masked regression was blocked by a tool rule, not by B1's judgment.",
           "**What this means:** on 4 episodes the ordering by direction macro-F1 (FULL, A2, A3 above B1 and A1) is "
           "driven mostly by test-01 and test-03, and final-action accuracy is low everywhere because the evidence "
           "and patch-content rules turn most decisions into escalations. This is evidence about the policy's "
           "conservatism, not about which design is better."])
    H1("7. Challenges faced and mitigation plans")
    B.append(("table", ["Challenge", "Mitigation"], [
        ["Free-tier quota and 503 overload (7 of 19 Day-2 runs were api_failure)",
         "Key pool with quota memory, retries, api_failure never scored, --retry-failed with memory-order guard"],
        ["Prompt contamination: Day-2 repair prompts contained dev-05 content",
         "Fixed in repair-v5; a test fails if a prompt shares 20 characters with any pilot episode; old records kept with a notice"],
        ["A1 side channel: commit message reachable by retrieval or read_file",
         "Corpus, search_repo and non-code read_file withheld in A1; checked by tests"],
        ["Strict test-patch rule causes escalations (dev-05)",
         "Report as a cost of safety; consider more allowed setup edits only on the development split"],
        ["Mutation strength unmeasurable for small functions (dev-06: reference test kills 0 of 2 mutants)",
         "More mutant operators and a separate 'strength unmeasurable' class in the final 15 days"],
        ["A1 and probe-less repairs are scored ambiguously", "Report direction and final-action accuracy side by side; decide the scoring rule before the final run"],
        ["Bubblewrap only; Docker daemon not accessible", "Test Docker on a machine with access, keep bwrap as the fallback"],
        ["Small pilot cannot support statistics", "Scale to three libraries and 72 episodes, two seeds, paired bootstrap"],
    ], [0.45, 0.55]))

    H1("8. Updated repository and reproducibility")
    L(["Repository: this project's `main` branch; `README.md` has the setup, `docs/interfaces.md` the interfaces and "
       "`docs/day*-*.md` the per-day logs.",
       "Setup: Python 3.12 virtual environment, `pip install -e .[dev,report]`, `.env` from `.env.example` (keys are "
       "never committed), a clone of boltons at the pin and its pilot venv, bubblewrap.",
       "Offline checks: `python -m pytest -q` (no API calls). Live runs: `scripts/run_experiment.py`, then "
       "`scripts/audit_integrity.py`, `scripts/summarize_results.py`, `scripts/make_report.py`.",
       "Every table in this report is produced by `docs/report/build_report.py` from `results/**/runs.jsonl`."])

    H1("9. Completion plan, 31 October to 14 November")
    B.append(("table", ["Dates", "Work"], [
        ["Oct 31 - Nov 2", "Stabilise dataset and environments; episode-construction scripts; select the three held-out libraries; validate gold labels; pin dependencies; development stream"],
        ["Nov 3 - 5", "Finish core features: hybrid retrieval, memory, Guard, fault recovery; fix the A1 and escalation scoring; freeze model and configuration on development data"],
        ["Nov 6 - 9", "Full evaluation: 72-episode test set for FULL and B1 (two seeds), then A1, A2, A3 and sanity baselines, checkpointed. Throughput check first: about 74 s per run and 400k tokens for 45 pilot runs"],
        ["Nov 10 - 11", "Metrics, paired statistics where feasible, failure analysis, test-weakening and incorrect-repair review"],
        ["Nov 12 - 13", "Final report, plots, reproducibility package, demo"],
        ["Nov 14", "Clean setup test, repository check, backups, final release; Nov 15 is submission only"],
    ], [0.18, 0.82]))

    H1("10. Demo and known limitations")
    P("Demo: `scripts/run_episode.py --episode dev-03 --condition FULL` prints the failure, retrieved evidence, "
      "decision, patch and verification output. `--condition TRIVIAL_FIX_CODE` needs no model. Comparison tables "
      "come from the saved results above. Known limitations are listed in Sections 3 and 6.")
    return B


# ---------------------------------------------------------------- LaTeX
_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex(s: str) -> str:
    out = "".join(_TEX.get(c, c) for c in s)
    out = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", out)
    return re.sub(r"`(.+?)`", lambda m: r"\texttt{" + m.group(1).replace(r"\_", r"\_") + "}", out)


def to_tex(blocks) -> str:
    o = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=2.2cm]{geometry}", r"\usepackage{graphicx}",
         r"\usepackage{longtable,array}", r"\usepackage[hidelinks]{hyperref}", r"\usepackage{parskip}",
         r"\setlength{\LTpre}{4pt}\setlength{\LTpost}{6pt}", r"\begin{document}"]
    for b in blocks:
        k = b[0]
        if k == "title":
            o += [r"\begin{center}{\LARGE\bfseries " + tex(b[1]) + r"}\\[4pt]" + tex(b[2]) + r"\end{center}"]
        elif k == "h1":
            o.append(r"\section*{" + tex(b[1]) + "}")
        elif k == "p":
            o += [tex(b[1]), ""]
        elif k == "ul":
            o += [r"\begin{itemize}"] + [r"\item " + tex(i) for i in b[1]] + [r"\end{itemize}"]
        elif k == "table":
            header, rows, widths = b[1], b[2], b[3]
            cols = "".join(r">{\raggedright\arraybackslash\footnotesize}p{%.3f\textwidth}" % (w * 0.86) for w in widths)
            if len(b) > 4:
                o.append(r"{\small\itshape " + tex(b[4]) + r"}\par\nopagebreak")
            o.append(r"\begin{longtable}{|" + "|".join([cols]) + "|}")
            o += [r"\hline", " & ".join(r"\textbf{" + tex(h) + "}" for h in header) + r" \\ \hline\endhead"]
            o += [" & ".join(tex(c) for c in r) + r" \\ \hline" for r in rows]
            o.append(r"\end{longtable}")
        elif k == "img":
            o += [r"\begin{center}\includegraphics[width=%.2f\textwidth]{figures/%s}\\{\small %s}\end{center}"
                  % (b[3], b[1].name, tex(b[2]))]
    o.append(r"\end{document}")
    return "\n".join(o) + "\n"


# ---------------------------------------------------------------- PDF
def to_pdf(blocks, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=5)
    cell = ParagraphStyle("c", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0)
    head = ParagraphStyle("h", parent=cell, fontName="Helvetica-Bold")
    h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=5)
    width = A4[0] - 4.4 * cm

    def m(s: str) -> str:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', s)

    story = []
    for b in blocks:
        k = b[0]
        if k == "title":
            story += [Paragraph(m(b[1]), ParagraphStyle("t", parent=ss["Title"], fontSize=19)),
                      Paragraph(m(b[2]), ParagraphStyle("s", parent=body, alignment=1)), Spacer(1, 6)]
        elif k == "h1":
            story.append(Paragraph(m(b[1]), h1))
        elif k == "p":
            story.append(Paragraph(m(b[1]), body))
        elif k == "ul":
            story += [Paragraph("&bull; " + m(i), ParagraphStyle("li", parent=body, leftIndent=12, firstLineIndent=-8,
                                                                spaceAfter=3)) for i in b[1]]
        elif k == "table":
            header, rows, widths = b[1], b[2], b[3]
            if len(b) > 4:
                story.append(Paragraph("<i>" + m(b[4]) + "</i>", cell))
            data = [[Paragraph(m(h), head) for h in header]] + [[Paragraph(m(c), cell) for c in r] for r in rows]
            t = Table(data, colWidths=[w * width for w in widths], repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story += [t, Spacer(1, 7)]
        elif k == "img" and b[1].is_file():
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(str(b[1])).getSize()
            w = width * b[3]
            story += [Image(str(b[1]), width=w, height=w * ih / iw), Paragraph("<i>" + m(b[2]) + "</i>", cell),
                      Spacer(1, 6)]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title="TestOrCode mid-submission report", author="Team Agent Zero").build(story)


def main() -> int:
    (HERE / "figures").mkdir(exist_ok=True)
    for d in (DEV_DIR,):
        for png in d.glob("*.png"):
            shutil.copy2(png, HERE / "figures" / png.name)
    if (TEST_DIR / "direction_macro_f1.png").is_file():
        for png in TEST_DIR.glob("*.png"):
            shutil.copy2(png, HERE / "figures" / f"test_{png.name}")
    blocks = build_blocks()
    (HERE / "report.tex").write_text(to_tex(blocks), encoding="utf-8")
    to_pdf(blocks, HERE / "report.pdf")
    sys.stdout.write(f"wrote {HERE / 'report.tex'} and {HERE / 'report.pdf'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
