"""Build the FINAL research report skeleton (``final/report.tex``, ``report.pdf``, ``report.md``) from saved results.

    python docs/report/build_final_report.py --dev results/final/dev --test results/final/test \\
        --dataset final --freeze configs/final.json

The mid-submission report (``build_report.py``, dated 30 October) is left as it was submitted.  This builder writes the
final report's structure (problem, research questions, related work, method, dataset, setup, results, qualitative
examples, failure analysis, limitations, future work, reproducibility):

* every number comes from the given results directories (``runs.jsonl``, and when present ``integrity_audit.jsonl``,
  ``stats.json`` from ``scripts/compare_conditions.py``, ``contamination*.json`` and ``*.score.md`` from
  ``scripts/citation_support.py``); a missing input is printed as "not yet evaluated" / "not computed", never as 0;
* prose the team must write is marked ``[TO WRITE: ...]`` so it cannot be mistaken for content;
* the implementation-status table describes the code as it is now, and its "runs found" column is counted from the
  results directories.

Defaults (``--dev``/``--test``) point at the pilot runs, so the skeleton can be previewed before the final runs exist.
Rendering (LaTeX source and the reportlab PDF) is shared with ``build_report.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from build_report import to_pdf, to_tex

from testorcode.conditions import DESCRIPTION, PROPOSAL_NAME
from testorcode.dataset import DEFAULT_DATASET, episode_path, load_episodes
from testorcode.evaluation import summarize
from testorcode.experiment import dataset_version, read_records
from testorcode.freeze import load_rules
from testorcode.report_tables import NOT_EVALUATED, ORDER, condition_rows, safety_rows

TODO = "[TO WRITE: {}]"


def frac(k, n) -> str:
    return "-" if not n or k is None else f"{k}/{n} ({k / n:.2f})"


def num(x) -> str:
    return "-" if x is None else f"{x:.3f}"


# --------------------------------------------------------------------------- data


class Split:
    """One results directory of one split, with everything the report reads from it."""

    def __init__(self, name: str, directory: Path | None, dataset: str, rules) -> None:
        self.name, self.dir, self.dataset, self.rules = name, directory, dataset, rules
        runs = directory / "runs.jsonl" if directory else None
        self.records = read_records(runs) if runs and runs.is_file() else []
        self.episodes = load_episodes(episode_path(name, dataset=dataset))
        self.version = dataset_version(name, dataset)
        audit = directory / "integrity_audit.jsonl" if directory else None
        self.audit = audit if audit and audit.is_file() else None
        self.config = self._json("config.json")
        self.stats = self._json("stats.json")

    def _json(self, name: str) -> dict | None:
        path = self.dir / name if self.dir else None
        return json.loads(path.read_text()) if path and path.is_file() else None

    def files(self, pattern: str) -> list[Path]:
        return sorted(self.dir.glob(pattern)) if self.dir and self.dir.is_dir() else []

    def runs_of(self, condition) -> int:
        return sum(r.condition is condition for r in self.records)


def condition_table(rows) -> tuple[list[str], list[list[str]]]:
    header = ["Condition", "Episodes run", "Exec. failures", "Direction macro-F1", "Direction acc.", "Final-action acc.",
              "Correct-repair rate", "Masked regressions"]
    body = []
    for r in rows:
        if not r["evaluated"]:
            body.append([r["condition"], NOT_EVALUATED, str(r["execution_failures"]), "-", "-", "-", "-", "-"])
            continue
        repair = frac(r["repairs_ok"], r["repairs"]) + (f" +{r['repair_unverifiable']} unverif." if r["repair_unverifiable"] else "")
        body.append([r["condition"], str(r["episodes_run"]), str(r["execution_failures"]), num(r["direction_macro_f1"]),
                     frac(r["direction_correct"], r["direction_scored"]), num(r["final_action_accuracy"]), repair,
                     frac(r["masked"], r["gold_fix"])])
    return header, body


def safety_table(rows) -> tuple[list[str], list[list[str]]]:
    header = ["Condition", "Mutant-kill retention (audit)", "Test weakening (audit)", "Unsafe taken effect",
              "Unsafe attempted", "Over-generalisation (S4)", "Patch-scope acc.", "Invalid tool calls", "Error recovery",
              "Stale-evidence errors"]
    body = []
    for r in rows:
        audited = r["audited"]
        body.append([
            r["condition"],
            frac(r["retained"], r["retention_denominator"]) if audited else "not audited",
            frac(r["weakened"], r["weakening_measured"]) if audited else "not audited",
            frac(r.get("unsafe_effect", 0), r["runs"]), frac(r.get("unsafe_attempt", 0), r["runs"]),
            frac(r.get("over_generalised", 0), r.get("decoy_runs")), frac(r.get("scope_ok", 0), r.get("patches")),
            frac(r.get("invalid_calls", 0), r.get("agent_calls")), frac(r.get("recovered", 0), r.get("faults")),
            frac(r.get("stale_err", 0), r.get("stale_den")),
        ])
    return header, body


def dataset_rows(splits: list[Split]) -> list[list[str]]:
    out = []
    for s in splits:
        by: dict[tuple[str, str], list] = {}
        for e in s.episodes:
            by.setdefault((e.agent_input.repo.repo_id, e.scenario.value), []).append(e)
        for (repo, scenario), eps in sorted(by.items()):
            tests = sum(len(e.ground_truth.per_test_labels) for e in eps)
            out.append([s.name, repo, scenario, str(len(eps)), str(tests)])
    return out


# --------------------------------------------------------------------------- content


def build_blocks(dev: Split, test: Split, frozen, thresholds, rules) -> list[tuple]:
    B: list[tuple] = []

    def h1(t):
        B.append(("h1", t))

    def p(t):
        B.append(("p", t))

    def ul(items):
        B.append(("ul", items))

    def table(header, rows, widths, caption=None):
        B.append(("table", header, rows, widths, *((caption,) if caption else ())))

    B.append(("title", ("TestOrCode: Intent-Grounded Triage of CI Test Failures with Scoped Change Memory and "
                        "Test-Integrity Checking"),
              "Team Agent Zero. Final report. " + TODO.format("date and author list")))

    h1("Abstract")
    p(TODO.format("150-200 words: problem, method, dataset size, the headline comparison with its CI and the "
                  "hypothesis verdict from Section 7, one limitation"))

    h1("1. Problem and research questions")
    p("When CI fails after a change, which side is wrong: the test (the behaviour changed on purpose) or the code (it "
      "regressed)? A coding agent rewarded by a green build can edit the test either way. TestOrCode decides per "
      "failing test between UPDATE_TEST, FIX_CODE and ESCALATE, grounds the decision in change-intent evidence, "
      "remembers approved changes with executable probes, and checks that a repair does not weaken the suite.")
    p("**Primary RQ.** On chronological streams of CI failures in small Python libraries, does a role-separated "
      "pipeline that (i) grounds each repair decision in retrieved change-intent evidence, (ii) remembers "
      "maintainer-approved behaviour changes as scoped records with executable probes, and (iii) checks test integrity "
      "after repair make fewer wrong-side repairs and weaken tests less than a single tool-using coding agent with the "
      "same tools, model and budget?")
    p("**Secondary RQ.** Does scoping memory to a function, behaviour signature and executable probe reduce "
      "over-generalisation on decoy regressions compared with coarse module-level memory, while still saving tool "
      "calls on later covered changes?")
    t = thresholds
    p(f"**Pre-registered hypothesis** (frozen config: {frozen.name if frozen else 'none given: proposal defaults'}). "
      f"Against B1, FULL gains at least +{t.min_direction_f1_gain} direction macro-F1, keeps the masked-regression rate "
      f"at most {t.max_masked_regression_full} (B1 at least {t.min_masked_regression_b1}), mutant-kill retention at "
      f"least {t.min_retention_full} (B1 at most {t.max_retention_b1}), and saves at least "
      f"{t.min_covered_tool_call_saving:.0%} of tool calls on memory-covered episodes; scoped memory over-generalises on "
      f"at most {t.max_overgeneralisation_scoped:.0%} of decoys against at least {t.min_overgeneralisation_coarse:.0%} "
      f"for coarse memory. It is rejected if the paired-bootstrap 95% CI of the FULL - B1 direction macro-F1 "
      f"difference includes 0, or if B2 closes the gap to under {t.b2_gap}.")

    h1("2. Related work")
    p(TODO.format("verify every citation before using it (the proposal marks several as 'verify before citing')"))
    ul(["Issue-resolution agents and benchmarks that use tests as the oracle (SWE-bench, SWE-agent, Agentless).",
        "Automated test repair (TaRGet / TaRBench, ReAssert): they assume the test should change.",
        "Test and code co-evolution studies.",
        "Mutation testing as a test-strength measure (mutation testing surveys, mutmut).",
        "Agent memory verified by execution or reflection (Voyager, Reflexion, ExpeL).",
        "Gap: the test-versus-code decision with integrity measurement and scoped memory across commits."])

    h1("3. Method")
    p("One deterministic Python orchestrator runs four roles: **Triage** reruns and groups the failing tests by "
      "changed function; the **Intent Analyst** retrieves change-intent evidence (commit messages, changelog, docs, "
      "docstrings, source comments, git history up to the CI commit's parent) and probe-gated memory, and states a "
      "hypothesis with cited evidence ids; the **Repairer** writes a scoped patch through restricted tools; the "
      "**Integrity Guard** reruns the tests, checks scope and weakening, measures test strength on mutants, asks for a "
      "semantic review and is the only role that writes memory, after a simulated maintainer approves. A decision "
      "policy requires cited positive evidence for UPDATE_TEST and cited expected-behaviour evidence for FIX_CODE; "
      "otherwise the group escalates. Safety rules are enforced by the tools, not the prompts.")
    p(TODO.format("architecture diagram; memory record format; retrieval details (the settings actually used are "
                  "listed in Section 5)"))
    header = ["Id", "What it is", "Proposal name", "Runs found (dev / test)"]
    rows = [[c.value, DESCRIPTION[c], PROPOSAL_NAME.get(c, "same"), f"{dev.runs_of(c)} / {test.runs_of(c)}"]
            for c in ORDER]
    table(header, rows, [0.16, 0.5, 0.14, 0.2], "Conditions. A3 and A4 are numbered as in the 5-day plan; the "
          "proposal swaps them. B0, B2, A4 and A5 differ from their neighbours only as stated (see conditions.py).")

    h1("4. Implementation status")
    status = [
        ["Orchestrator, four roles, decision policy", "Implemented"],
        ["B0 plain LLM / B1 single agent / B2 strong prompt", "Implemented (one call / tool loop / B1 + integrity rules)"],
        ["A1-A5 ablations", "Implemented (A4: module-scope memory without probes; A5: merged roles, Guard checks kept)"],
        ["Retrieval: BM25, hybrid BM25 + dense + RRF, optional rerank", "Implemented; mode is a frozen setting"],
        ["git_log / git_show / git_blame with history cutoff", "Implemented; tested on the real clone"],
        ["Probe-gated scoped memory, maintainer-approved writes", "Implemented"],
        ["Mutation check: extended operators, 'strength unmeasurable' policy, cached mutants",
         "Implemented; policy is a frozen setting"],
        ["Scenario T: per-episode fault plans and a runner-level fault rate", "Implemented"],
        ["Docker sandbox routing and image recipe", "Implemented; " + TODO.format("live Docker result or 'bubblewrap only'")],
        ["Several libraries in one experiment", "Implemented"],
        ["Metrics: safety, memory, tools, stale evidence, failure causes; paired statistics; hypothesis check",
         "Implemented"],
        ["Contamination probe, provider check, citation-support labelling", "Implemented; results in Section 7 if run"],
        ["Fine-tuning (optional appendix of the proposal), frontend", "Dropped"],
    ]
    table(["Component", "Status"], status, [0.6, 0.4])

    h1("5. Dataset and experimental setup")
    table(["Split", "Library", "Scenario", "Episodes", "Failing tests"], dataset_rows([dev, test]),
          [0.12, 0.22, 0.22, 0.2, 0.24], f"Dataset '{dev.dataset}': episodes per library and scenario (from the files).")
    p(TODO.format("how the episodes were built and verified (real commits, injected mutants, 5-run stability filter, "
                  "leak audit), library selection, gold-evidence curation, known label caveats"))
    for s in (dev, test):
        cfg = s.config or {}
        settings = cfg.get("settings") if isinstance(cfg.get("settings"), dict) else {}
        frozen_cfg = cfg.get("frozen") or {}
        seeds = sorted({r.provenance.seed for r in s.records}, key=lambda x: -1 if x is None else x)
        models = sorted({r.provenance.model_id for r in s.records if r.provenance.model_id
                         and not r.condition.value.startswith("TRIVIAL")})
        if not s.records:
            p(f"**{s.name} split:** {NOT_EVALUATED}.")
            continue
        ul([f"**{s.name} split** ({s.dir}): {len(s.records)} runs, seeds {seeds}, model(s) {', '.join(models) or '-'}.",
            f"Frozen config: {frozen_cfg.get('name', 'none recorded')}"
            + (f" (sha256 {frozen_cfg['sha256'][:12]})" if frozen_cfg.get("sha256") else "") + ".",
            "Retrieval " + str(settings.get("retrieval_mode", "not recorded")) + ", Guard unmeasurable-strength policy "
            + str(settings.get("guard_unmeasurable_strength", "not recorded")) + ", test-patch rule "
            + str(settings.get("test_patch_rule", "not recorded")) + ", tool-fault rate "
            + str(cfg.get("tool_fault_rate", "not recorded")) + ".",
            rules.describe() + "."])

    h1("6. Results")
    for s in (dev, test):
        if not s.records:
            p(f"**{s.name} split:** {NOT_EVALUATED}.")
            continue
        h, body = condition_table(condition_rows(s.records, s.episodes, dataset_version=s.version, rules=rules))
        table(h, body, [0.13, 0.1, 0.09, 0.11, 0.13, 0.11, 0.18, 0.15],
              f"Main results, {s.name} split. Raw counts next to rates; '-' = not computable.")
        h, body = safety_table(safety_rows(s.records, s.episodes, dataset_version=s.version, audit_jsonl=s.audit))
        table(h, body, [0.1] * 10, f"Integrity, safety, memory and tool-call metrics, {s.name} split.")

    h1("7. Statistics and the hypothesis")
    for s in (dev, test):
        if s.stats is None:
            p(f"**{s.name} split:** not computed (run `scripts/compare_conditions.py {s.dir or '<dir>'}/runs.jsonl "
              "--freeze <frozen config>`).")
            continue
        rows = []
        for pair in s.stats.get("pairs", []):
            f1 = pair["direction_macro_f1"]
            ci = f"[{num(f1['ci95'][0])}, {num(f1['ci95'][1])}]" if f1.get("ci95") else "-"
            md = pair["mcnemar_direction"]
            rows.append([f"{f1['a']} - {f1['b']}", str(f1["n_episodes"]), num(f1["difference"]), ci,
                         f"{md['a_only_correct']} / {md['b_only_correct']}, p={md['p_value']:.3f}"])
        table(["Pair", "Episodes", "Direction macro-F1 diff", "95% CI", "McNemar (A only / B only)"], rows,
              [0.2, 0.12, 0.2, 0.22, 0.26], f"Paired statistics, {s.name} split.")
        hyp = s.stats.get("hypothesis")
        if hyp:
            p(f"**Hypothesis verdict ({s.name} split): {hyp['verdict']}.** " + "; ".join(
                f"{r['test']}: {'fires' if r['fires'] else 'does not fire' if r['fires'] is False else 'not computable'}"
                for r in hyp["rejection_tests"]) + ".")
            table(["Criterion", "Observed", "Threshold", "Met"],
                  [[c["criterion"], num(c["observed"]), c["rule"], str(c["met"]) if c["met"] is not None else "-"]
                   for c in hyp["criteria"]], [0.5, 0.15, 0.2, 0.15])
    for s in (dev, test):
        for path in s.files("contamination*.json"):
            data = json.loads(path.read_text())
            summary = data.get("summary", {})
            p(f"**Contamination probe ({path.name}, model {data.get('model_id')}):** pooled added-line recall "
              f"{num(summary.get('pooled_added_line_recall'))} over {summary.get('episodes_scored')} episodes; "
              f"{summary.get('episodes_memorised')} episode(s) over the memorised threshold.")
        for path in s.files("*.score.md"):
            p(f"**Citation support ({path.name}):** " + " ".join(
                line for line in path.read_text().splitlines() if line and not line.startswith("|")))
    p(TODO.format("interpret the comparisons with their n; a negative result against B2 is a valid finding"))

    h1("8. Qualitative examples")
    p(TODO.format("one short case each, from review_runs.py output, with run ids"))
    ul(["A masked regression, and how it slipped through (or that none did).",
        "A decoy where memory was wrongly reused, and one where the probe blocked it.",
        "Stale docstrings that led to a wrong hypothesis.",
        "A case where the Analyst and the Guard disagreed.",
        "A repaired test that passes but kills fewer mutants than the developer's.",
        "A tool failure (injected fault) and the recovery.",
        "An escalation a human would have skipped."])

    h1("9. Failure analysis")
    for s in (dev, test):
        if not s.records:
            continue
        summaries = summarize(s.records, s.episodes, dataset_version=s.version, rules=rules)
        causes = sorted({k for x in summaries for k, v in x.metrics.items() if k.startswith("failure_cause[") and v})
        if causes:
            table(["Condition", *[c.removeprefix("failure_cause[").removesuffix("]") for c in causes]],
                  [[x.condition.value, *[str(int(x.metrics.get(c) or 0)) for c in causes]] for x in summaries],
                  [0.2] + [0.8 / len(causes)] * len(causes), f"Failure causes (first wrong step), {s.name} split.")
    p(TODO.format("what the dominant causes mean; API failures are logged apart and never counted against an agent"))

    h1("10. Limitations and threats to validity")
    ul(["Mutants are single-operator changes, not real regressions; S5 assumes 'no evidence means escalate'.",
        "Mutation score is a proxy for test strength; small functions give few mutants (the unmeasurable policy).",
        "Contamination: popular libraries may be in the model's training data (see the probe).",
        "One hosted model for every condition; the second-model check covers only a subset.",
        TODO.format("dataset size and libraries actually used, seeds, Docker status, label caveats (e.g. dev-01)")])

    h1("11. Future work")
    p(TODO.format("e.g. more libraries, real-regression set, the optional fine-tuning appendix, better rerankers"))

    h1("12. Reproducibility")
    ul([("Setup: Python 3.12, `pip install -r requirements-lock.txt && pip install --no-deps -e .`, `.env` from "
         "`.env.example`, the pinned clones and their venvs (`repos.json`), bubblewrap or Docker."),
        "Offline checks: `python -m pytest -q`.",
        ("Runs: `scripts/run_experiment.py --dataset <d> --repos repos.json --freeze configs/final.json ...`, then "
         "`scripts/audit_integrity.py`, `scripts/make_report.py`, `scripts/compare_conditions.py --freeze ...`."),
        "This report: `docs/report/build_final_report.py --dev <dir> --test <dir> --dataset <d> --freeze <file>`."])
    return B


def to_markdown(blocks) -> str:
    out = []
    for b in blocks:
        kind = b[0]
        if kind == "title":
            out += [f"# {b[1]}", "", b[2], ""]
        elif kind == "h1":
            out += [f"## {b[1]}", ""]
        elif kind == "p":
            out += [b[1], ""]
        elif kind == "ul":
            out += [*(f"- {i}" for i in b[1]), ""]
        elif kind == "table":
            if len(b) > 4:
                out += [f"*{b[4]}*", ""]
            out += ["| " + " | ".join(b[1]) + " |", "|" + "---|" * len(b[1])]
            out += ["| " + " | ".join(r) + " |" for r in b[2]] + [""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev", type=Path, default=ROOT / "results/day3/pilot-dev-5cond")
    parser.add_argument("--test", type=Path, default=ROOT / "results/day4/pilot-test-5cond")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--freeze", default=None, help="frozen config: scoring rules and hypothesis thresholds")
    parser.add_argument("--out", type=Path, default=HERE / "final")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    rules, thresholds, frozen = load_rules(args.freeze)
    dev = Split("dev", args.dev, args.dataset, rules)
    test = Split("test", args.test, args.dataset, rules)
    blocks = build_blocks(dev, test, frozen, thresholds, rules)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(to_markdown(blocks), encoding="utf-8")
    (args.out / "report.tex").write_text(to_tex(blocks), encoding="utf-8")
    if not args.no_pdf:
        to_pdf(blocks, args.out / "report.pdf")
    sys.stdout.write(f"wrote {args.out}/report.md, report.tex{'' if args.no_pdf else ', report.pdf'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
