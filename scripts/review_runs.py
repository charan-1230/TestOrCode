"""Review agent decisions, execution traces and failures from saved run records (no model, no repository).

    python scripts/review_runs.py results/day2/pilot-full-b1/runs.jsonl
    python scripts/review_runs.py results/day2/*.json results/day2/flash-lite/*.json --condition FULL,B1
    python scripts/review_runs.py runs.jsonl --episode dev-06 --tools     # also list every tool call

Accepts ``runs.jsonl`` files (``run_experiment``) and single ``<run_id>.json`` records (``run_episode.py``);
``*.summary.json`` and ``config.json`` files are skipped.  For each run it prints the per-group trace (function,
decision, Guard verdicts, disposition, patches), the predictions (scored against gold when the record carries the
harness evaluation), tool calls by tool/role/status, errors and the escalation's missing evidence.  It ends with an
aggregate of why groups escalated and why runs failed, so recurring agent-workflow failures stand out.

The "why" labels are a reading of the recorded reasons (string patterns), meant for triage of the logs, not a metric.
Nothing here reads gold labels other than what ``experiment.scored`` already wrote into a record's ``evaluation``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from testorcode.schemas.result_schemas import AgentRole, ExperimentRunRecord, RunStatus

_EXECUTION_FAILURES = frozenset({
    RunStatus.TOOL_FAILURE, RunStatus.API_FAILURE, RunStatus.BUDGET_EXHAUSTED, RunStatus.ORCHESTRATOR_ERROR,
    RunStatus.INCOMPLETE,
})


def load(paths: list[Path]) -> tuple[list[ExperimentRunRecord], list[str]]:
    records: list[ExperimentRunRecord] = []
    problems: list[str] = []
    for path in paths:
        if path.name == "config.json" or path.name.endswith(".summary.json"):
            continue
        try:
            if path.suffix == ".jsonl":
                lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
                records += [ExperimentRunRecord.model_validate_json(ln) for ln in lines]
            else:
                records.append(ExperimentRunRecord.model_validate_json(path.read_text()))
        except (OSError, ValidationError, ValueError) as exc:
            problems.append(f"{path}: {type(exc).__name__}")
    return records, problems


def escalation_cause(record: ExperimentRunRecord, group_id: str) -> str:
    """Why one group escalated, read from who decided and the recorded reasons."""
    group = next((g for g in record.failure_groups if g.group_id == group_id), None)
    prediction = next((p for p in record.predictions if p.group_id == group_id), None)
    rationale = (prediction.rationale or "") if prediction else ""
    by = prediction.decided_by if prediction else None
    if by is AgentRole.INTEGRITY_GUARD or (group and group.guard_decision and group.guard_decision.value == "escalate"):
        reasons = " ".join(group.guard_reasons) if group else rationale
        if "semantic review flagged" in reasons:  # not "...the semantic review found it consistent, but ..."
            return "guard: semantic review flagged the repair"
        if "mutant" in reasons:
            return "guard: test strength not shown by mutants"
        if "probe" in reasons:
            return "guard: no trusted probe"
        return "guard: other"
    if "never retrieved" in rationale:
        return "unsupported citation (id never retrieved)"
    if "both for and against" in rationale:
        return "decision policy: contradictory evidence"
    if "rejected after" in rationale:
        return "repair rejected after revision"
    if by is AgentRole.ORCHESTRATOR:
        return "decision policy: missing evidence"
    if by is AgentRole.REPAIRER:
        return "repairer escalated (no safe in-policy repair)"
    if by is AgentRole.INTENT_ANALYST:
        return "analyst recommended ESCALATE"
    if by is AgentRole.SINGLE_AGENT:
        if "out of steps" in rationale or "no decision" in rationale:
            return "B1: no decision for these tests"
        if "applicable edit" in rationale:
            return "B1: edit could not be placed"
        return "B1: escalated or failed the evidence policy"
    return "other"


def review(record: ExperimentRunRecord, *, tools: bool) -> list[str]:
    p, o = record.provenance, record.outcome
    out = [
        (f"=== {p.episode_id}  {p.condition.value}  seed={p.seed}  {p.model_id or '-'}  "
         f"prompts={p.prompt_template_version or '-'}  run={p.run_id}"),
        (f"status={o.status.value}  final_suite={record.final_test_status.value}  "
         f"llm_calls={record.usage.llm_calls}  tokens={record.usage.total_tokens}  revisions={o.revisions_used}"),
    ]
    for g in record.failure_groups:
        verdicts = "/".join(r for r in [g.guard_decision.value if g.guard_decision else None] if r) or "-"
        out.append(f"  group {g.group_id}: {g.function_qualname or '-'} (mapped by {g.mapped_by or 'agent'}) "
                   f"tests={len(g.test_ids)} decision={g.decision.value if g.decision else '-'} guard={verdicts} "
                   f"disposition={g.disposition or '-'} patches={g.patch_ids or '-'}")
        for reason in g.guard_reasons:
            out.append(f"    guard reason: {reason[:220]}")
    scores = {s.test_id: s for s in (record.evaluation.decisions if record.evaluation else [])}
    for pred in record.predictions:
        score = scores.get(pred.test_id)
        gold = (f" gold={score.expected_action.value} {'OK' if score.correct else 'WRONG'}"
                + (f" ({score.wrong_side})" if score.wrong_side else "")) if score else ""
        out.append(f"  decision {pred.test_id}: {pred.predicted_action.value} by {pred.decided_by.value} "
                   f"hypothesis={pred.hypothesis.value if pred.hypothesis else '-'} "
                   f"cited={pred.cited_evidence_ids or '-'}{gold}")
    for patch in record.patches:
        out.append(f"  patch {patch.patch_id} {patch.scope.value} {patch.target_files} -> "
                   f"{patch.application_status.value}" + (f": {patch.apply_error[:200]}" if patch.apply_error else ""))
    calls = Counter((e.tool.value, e.called_by.value, e.status.value) for e in record.tool_events)
    out.append("  tool calls: " + (", ".join(f"{t}/{r}/{s}={n}" for (t, r, s), n in sorted(calls.items())) or "-"))
    if tools:
        for e in record.tool_events:
            out.append(f"    {e.tool.value} by {e.called_by.value} [{e.arguments.get('budget', '-')}] {e.status.value}"
                       + (f": {e.error_message[:160]}" if e.error_message else ""))
    if record.memory_reads or record.memory_writes:
        reads = [f"{m.record_id}:{m.probe_status.value}{'*' if m.used_as_evidence else ''}" for m in record.memory_reads]
        writes = [f"{w.record_id}:{'written' if w.written else w.blocked_reason}" for w in record.memory_writes]
        out.append(f"  memory reads (*=cited): {reads or '-'}  writes: {writes or '-'}")
    if o.escalation:
        for item in o.escalation.evidence_missing:
            out.append(f"  missing: {item[:220]}")
    for err in record.errors:
        out.append(f"  error: {err.category.value} at {err.operation} (fatal={err.fatal}): {err.message[:200]}")
    return out


def aggregate(records: list[ExperimentRunRecord]) -> list[str]:
    status = Counter((r.condition.value, r.outcome.status.value) for r in records)
    failures = Counter(
        (r.condition.value, f"{e.category.value} at {e.operation.split(':')[0]}")
        for r in records if r.outcome.status in _EXECUTION_FAILURES for e in r.errors if e.fatal
    )
    escalations = Counter(
        (r.condition.value, escalation_cause(r, g))
        for r in records
        for g in dict.fromkeys(p.group_id for p in r.predictions if p.predicted_action.value == "ESCALATE" and p.group_id)
    )
    out = ["", "=== aggregate", "run status by condition:"]
    out += [f"  {c:<22} {s:<18} {n}" for (c, s), n in sorted(status.items())]
    out.append("execution failures (fatal error, stage):")
    out += [f"  {c:<22} {k:<40} {n}" for (c, k), n in sorted(failures.items())] or ["  -"]
    out.append("escalated groups by cause:")
    out += [f"  {c:<22} {k:<48} {n}" for (c, k), n in sorted(escalations.items())] or ["  -"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="runs.jsonl files and/or <run_id>.json records")
    parser.add_argument("--condition", default="", help="comma separated, e.g. FULL,B1")
    parser.add_argument("--episode", default="", help="comma separated episode ids")
    parser.add_argument("--tools", action="store_true", help="list every tool call")
    args = parser.parse_args()
    records, problems = load(args.paths)
    wanted_c = {c.strip() for c in args.condition.split(",") if c.strip()}
    wanted_e = {e.strip() for e in args.episode.split(",") if e.strip()}
    records = [r for r in records if (not wanted_c or r.condition.value in wanted_c)
               and (not wanted_e or r.episode_id in wanted_e)]
    for problem in problems:
        sys.stderr.write(f"skipped (not a run record of this schema): {problem}\n")
    if not records:
        sys.stderr.write("no run records selected\n")
        return 1
    for record in sorted(records, key=lambda r: (r.episode_id, r.condition.value, r.provenance.started_at)):
        sys.stdout.write("\n".join(review(record, tools=args.tools)) + "\n\n")
    sys.stdout.write("\n".join(aggregate(records)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
