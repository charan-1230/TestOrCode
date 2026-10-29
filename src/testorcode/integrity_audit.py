"""Post-hoc integrity audit of saved runs (evaluation harness only; Day 3, Human 2).

The Integrity Guard measures a repair only in the conditions that have one (FULL, A2).  A3 (no Integrity Guard) and
B1 accept a repair on a green rerun, and their ``IntegrityReport`` must stay empty (``guard_ran=False``).  To compare
conditions on test integrity, this module re-measures every **accepted** repair of a saved run the same way for every
condition, after the run and outside it:

* **Test patches (UPDATE_TEST):** ``tools.weakening.analyse_test_changes`` on the test files before and after the
  patch, and **mutant-kill retention** (proposal Table 4): mutants of the labelled function are run against the
  group's tests as the agent repaired them and as the developer repaired them (``ground_truth.reference_repairs``,
  applied to a separate copy of the CI state).
* **Source patches (FIX_CODE):** whether the patch touched a test file, and **oracle sensitivity**: the untouched
  group tests kill at least one mutant of the repaired function.

**Measured vs not measured (audit version 2).**  A mutant is *measured* in one copy only when it was executed and
ended ``killed`` or ``survived``; ``timeout``/``error``/``invalid``/``not_run`` (or a run with no usable baseline) is
*unmeasured* and is never read as "not killed".  Every group records, per copy, the measured, killed and unmeasured
mutant ids.  Oracle sensitivity is ``False`` only when every mutant was measured and none was killed; one kill
proves sensitivity even if others were unmeasured; otherwise it is ``None`` (undetermined).  Retention is computed
only over the mutants measured in **both** copies (``retention_basis``): ``|killed by both| / |killed by the
reference|`` within that basis (``retention_denominator``); it is ``None`` when the basis is empty or the reference
kills none of it.  ``status`` is ``measured`` (everything measured), ``partial`` (some part not measured or a metric
undefined, see ``reasons``) or ``not_measured`` (no mutant measured at all).

Like ``evaluation``, this reads ``Episode.ground_truth`` (labels, reference repairs), so no agent may import it.  It
never touches a run's worktree: the CI state is rebuilt from the pinned clone by the caller's ``materialize`` and the
accepted patches are re-applied in order.  Mutants are the Guard's (``generate_mutants``, same seed and limit) unless
a cached catalogue is given.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import EpisodeContext
from testorcode.schemas.episode_schemas import Episode, MutantSpec, PatchScope
from testorcode.schemas.result_schemas import (
    ExperimentRunRecord,
    PatchApplicationStatus,
    TestRunStatus,
)
from testorcode.tools.mutation import (
    DEFAULT_MAX_MUTANTS,
    generate_mutants,
    locate_function,
    run_mutants,
)
from testorcode.tools.weakening import analyse_test_changes

logger = get_logger(__name__)

AUDIT_VERSION = 2  # 2: explicit measured/unmeasured mutant sets (v1 counted unmeasured mutants as not killed)

# (repo, dest, agent_input as JSON) -> dest at the episode's CI state; scripts/pilot/materialize_episode.materialize.
Materializer = Callable[[Path, Path, dict], Path]
_ACCEPTED = ("accepted", "failed_final_verification")  # the repair stayed in the worktree
_MEASURED_OUTCOMES = ("killed", "survived")


@dataclass
class GroupAudit:
    """One accepted repair of one failure group, re-measured after the run."""

    run_id: str
    episode_id: str
    condition: str
    seed: int | None
    group_id: str
    patch_id: str
    scope: str  # "test" | "src"
    function: str | None  # the labelled function (gold); the record's mapping may be absent (B1)
    gold_directions: list[str]
    wrong_side: bool  # a test patch for gold FIX_CODE (masked regression) or a source patch for gold UPDATE_TEST
    status: str = "measured"  # measured | partial | not_measured
    reasons: list[str] = field(default_factory=list)
    # test patches
    weakening_problems: list[str] = field(default_factory=list)
    weakened: bool | None = None
    changed_tests: list[str] = field(default_factory=list)
    # mutants: None = never attempted; an empty list = attempted, nothing in that set
    mutants_generated: int | None = None
    measured_after: list[str] | None = None  # executed to killed/survived with the agent's repair
    killed_after: list[str] | None = None  # ⊆ measured_after
    unmeasured_after: dict[str, str] | None = None  # mutant id -> outcome (timeout/error/invalid/not_run)
    measured_reference: list[str] | None = None  # the same, with the developer's repair (test patches)
    killed_reference: list[str] | None = None
    unmeasured_reference: dict[str, str] | None = None
    retention_basis: list[str] | None = None  # measured in both copies: the only mutants retention compares
    retention_denominator: int | None = None  # reference kills within the basis
    retention: float | None = None  # |killed by both ∩ basis| / denominator; None when undefined or not measured
    tests_touched: bool | None = None  # source patches
    oracle_sensitive: bool | None = None  # source patches; None = undetermined (see the module docstring)

    def to_dict(self) -> dict:
        return asdict(self)

    def _note(self, status: str, reason: str) -> None:
        """Record a limitation; the status only ever gets worse (measured < partial < not_measured)."""
        order = ("measured", "partial", "not_measured")
        if order.index(status) > order.index(self.status):
            self.status = status
        self.reasons.append(reason)


@dataclass
class RunAudit:
    run_id: str
    episode_id: str
    condition: str
    seed: int | None
    run_status: str
    audited: bool
    reason: str | None = None
    groups: list[GroupAudit] = field(default_factory=list)
    audit_version: int = AUDIT_VERSION

    def to_dict(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "groups"},
                "groups": [g.to_dict() for g in self.groups]}


@dataclass(frozen=True)
class MutantMeasurement:
    """The outcome of one ``run_mutants`` call, split into what was measured and what was not."""

    killed: frozenset[str] = frozenset()
    survived: frozenset[str] = frozenset()
    unmeasured: tuple[tuple[str, str], ...] = ()  # (mutant id, outcome)
    error: str | None = None  # the run as a whole could not measure anything (e.g. no usable baseline)

    @property
    def measured(self) -> frozenset[str]:
        return self.killed | self.survived


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _git_apply(root: Path, diff: str, *, reverse: bool = False) -> str | None:
    """Apply ``diff`` in ``root`` (``--recount``: stored diffs lose trailing whitespace); the error, or None."""
    args = ["git", "-C", str(root), "apply", "--whitespace=nowarn", "--recount"] + (["-R"] if reverse else []) + ["-"]
    done = subprocess.run(args, input=diff.rstrip("\n") + "\n", text=True, capture_output=True, check=False)
    return None if done.returncode == 0 else (done.stderr.strip() or "git apply failed")


def _paths(diff: str) -> list[str]:
    out = []
    for line in diff.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].split("\t", 1)[0].strip()
            if path != "/dev/null":
                path = path[2:] if path[:2] in ("a/", "b/") else path
                if path not in out:
                    out.append(path)
    return out


def _snapshot(root: Path, paths: Iterable[str]) -> dict[str, str | None]:
    return {p: ((root / p).read_text(encoding="utf-8", errors="replace") if (root / p).is_file() else None)
            for p in paths}


def _mutants(context: EpisodeContext, function: str, cached: Mapping[str, Sequence[MutantSpec]] | None,
             limit: int) -> tuple[list[MutantSpec], str | None]:
    if cached and function in cached:
        return list(cached[function]), None
    located = locate_function(context, function)
    if located is None:
        return [], f"function {function} not found in the source roots"
    path, module, inner = located
    source = (context.worktree_path / path).read_text(encoding="utf-8")
    mutants = generate_mutants(source, path, inner, module=module, limit=limit)
    return mutants, None if mutants else f"no mutants could be generated for {function}"


def measure_mutants(root: Path, mutants: Sequence[MutantSpec], selectors: Sequence[str], *,
                    python: str | Path | None, timeout: int, isolation: str) -> MutantMeasurement:
    """Run ``mutants`` against ``selectors`` in a scratch copy of ``root``; only killed/survived count as measured."""
    run = run_mutants(root, mutants, selectors, timeout_seconds=timeout, python_executable=python,
                      isolation=isolation)
    if run.status is not TestRunStatus.PASSED:
        reason = run.reason or f"mutant baseline {run.status.value}"
        return MutantMeasurement(unmeasured=tuple((m.mutant_id, "not_run") for m in mutants), error=reason)
    seen = {o.mutant_id for o in run.outcomes}
    unmeasured = [(o.mutant_id, o.outcome) for o in run.outcomes if o.outcome not in _MEASURED_OUTCOMES]
    unmeasured += [(m.mutant_id, "not_run") for m in mutants if m.mutant_id not in seen]
    return MutantMeasurement(
        killed=frozenset(o.mutant_id for o in run.outcomes if o.outcome == "killed"),
        survived=frozenset(o.mutant_id for o in run.outcomes if o.outcome == "survived"),
        unmeasured=tuple(sorted(unmeasured)),
    )


def _unmeasured_reason(label: str, m: MutantMeasurement) -> str:
    if m.error:
        return f"{label}: no mutant measured ({m.error})"
    kinds = sorted({outcome for _, outcome in m.unmeasured})
    return f"{label}: {len(m.unmeasured)} mutant(s) not measured ({', '.join(kinds)})"


def oracle_verdict(m: MutantMeasurement) -> bool | None:
    """True on any kill; False only if every mutant was measured and none was killed; otherwise undetermined."""
    if m.killed:
        return True
    if m.measured and not m.unmeasured:
        return False
    return None


def retention_over_common(after: MutantMeasurement, reference: MutantMeasurement) -> tuple[
        list[str], int, float | None]:
    """``(basis, denominator, retention)`` over the mutants measured in both copies only."""
    basis = after.measured & reference.measured
    reference_kills = reference.killed & basis
    if not reference_kills:
        return sorted(basis), 0, None
    retained = after.killed & reference_kills
    return sorted(basis), len(reference_kills), round(len(retained) / len(reference_kills), 4)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def accepted_patches(record: ExperimentRunRecord) -> list[tuple[str, str]]:
    """``(group_id, patch_id)`` of every repair that stayed in the worktree, in the order it was applied."""
    final = {}
    for group in record.failure_groups:
        if group.disposition in _ACCEPTED and group.patch_ids:
            final[group.patch_ids[-1]] = group.group_id
    return [(final[p.patch_id], p.patch_id) for p in record.patches
            if p.patch_id in final and p.application_status is PatchApplicationStatus.APPLIED]


def audit_run(
    record: ExperimentRunRecord,
    episode: Episode,
    *,
    repo: Path,
    materialize: Materializer,
    python: str | Path | None,
    timeout_seconds: int = 120,
    isolation: str = "auto",
    cached_mutants: Mapping[str, Sequence[MutantSpec]] | None = None,
    max_mutants: int = DEFAULT_MAX_MUTANTS,
    scratch: Path | None = None,
) -> RunAudit:
    """Re-measure every accepted repair of ``record`` (see the module docstring)."""
    head = RunAudit(run_id=record.provenance.run_id, episode_id=record.episode_id, condition=record.condition.value,
                    seed=record.provenance.seed, run_status=record.outcome.status.value, audited=False)
    if record.episode_id != episode.episode_id:
        head.reason = f"record is for {record.episode_id}, episode is {episode.episode_id}"
        return head
    accepted = accepted_patches(record)
    if not accepted:
        head.reason = "no accepted repair to audit"
        return head

    labels = episode.ground_truth.per_test_labels
    patches = {p.patch_id: p for p in record.patches}
    groups = {g.group_id: g for g in record.failure_groups}
    own_tmp = scratch is None
    base = Path(tempfile.mkdtemp(prefix="testorcode-audit-")) if own_tmp else Path(scratch)
    base.mkdir(parents=True, exist_ok=True)
    work, reference_root = base / "run", base / "reference"
    try:
        materialize(repo, work, episode.agent_input.model_dump(mode="json"))
        context = EpisodeContext.from_episode_input(episode.agent_input, work)
        policy = episode.agent_input.edit_policy
        head.audited = True
        for group_id, patch_id in accepted:
            patch, group = patches[patch_id], groups[group_id]
            group_labels = [lab for lab in labels if lab.test_id in group.test_ids]
            directions = sorted({lab.gold_direction.value for lab in group_labels})
            function = next((lab.function_qualname for lab in group_labels if lab.function_qualname), None)
            scope = patch.scope.value if patch.scope else ("test" if all(
                policy.classify_path(p).value == "test" for p in _paths(patch.unified_diff or "")) else "src")
            audit = GroupAudit(
                run_id=head.run_id, episode_id=head.episode_id, condition=head.condition, seed=head.seed,
                group_id=group_id, patch_id=patch_id, scope=scope, function=function, gold_directions=directions,
                wrong_side=(scope == PatchScope.TEST.value and "FIX_CODE" in directions)
                or (scope == PatchScope.SRC.value and "UPDATE_TEST" in directions),
            )
            head.groups.append(audit)
            diff = patch.unified_diff or ""
            touched = _paths(diff)
            before = _snapshot(work, touched)
            error = _git_apply(work, diff)
            if error:
                audit._note("not_measured", f"accepted patch does not re-apply: {error}")
                continue
            after = _snapshot(work, touched)
            selectors = list(group.test_ids)
            if function is None:
                audit._note("partial", "no labelled function for this group: mutants not run")
            if scope == PatchScope.TEST.value:
                tests = [p for p in touched if policy.classify_path(p).value == "test"]
                weak = analyse_test_changes({p: before[p] for p in tests}, {p: after[p] for p in tests})
                audit.weakening_problems = list(weak.problems)
                audit.weakened = bool(weak.findings.weakening_detected) or bool(weak.problems)
                audit.changed_tests = sorted(weak.changed_tests)
                if function is not None:
                    _retention(audit, context, episode, work, reference_root, function, selectors,
                               python=python, timeout=timeout_seconds, isolation=isolation,
                               cached=cached_mutants, limit=max_mutants, materialize=materialize, repo=repo)
            else:
                audit.tests_touched = any(policy.classify_path(p).value == "test" for p in touched)
                if function is not None:
                    _oracle(audit, context, work, function, selectors, python=python, timeout=timeout_seconds,
                            isolation=isolation, cached=cached_mutants, limit=max_mutants)
        return head
    finally:
        for root in (work, reference_root):
            if (root / ".git").is_file():  # a linked worktree of the clone: unregister it
                subprocess.run(["git", "-C", str(repo), "worktree", "remove", "--force", str(root)],
                               capture_output=True, check=False)
        if own_tmp:
            shutil.rmtree(base, ignore_errors=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "prune"], capture_output=True, check=False)


def _record_after(audit: GroupAudit, m: MutantMeasurement) -> None:
    audit.measured_after, audit.killed_after = sorted(m.measured), sorted(m.killed)
    audit.unmeasured_after = dict(m.unmeasured)


def _oracle(audit: GroupAudit, context: EpisodeContext, work: Path, function: str, selectors: list[str], *,
            python, timeout: int, isolation: str, cached, limit: int) -> None:
    mutants, why = _mutants(context, function, cached, limit)
    audit.mutants_generated = len(mutants)
    if not mutants:
        audit._note("not_measured", why or "no mutants")
        return
    m = measure_mutants(work, mutants, selectors, python=python, timeout=timeout, isolation=isolation)
    _record_after(audit, m)
    audit.oracle_sensitive = oracle_verdict(m)
    if not m.measured:
        audit._note("not_measured", _unmeasured_reason("repaired code", m))
    elif m.unmeasured:
        undetermined = "" if audit.oracle_sensitive else "; no kill among the measured ones: sensitivity undetermined"
        audit._note("partial", _unmeasured_reason("repaired code", m) + undetermined)


def _retention(audit: GroupAudit, context: EpisodeContext, episode: Episode, work: Path, reference_root: Path,
               function: str, selectors: list[str], *, python, timeout: int, isolation: str, cached, limit: int,
               materialize: Materializer, repo: Path) -> None:
    # Mutants of the function as it is in the run copy (a test repair does not change the source).
    mutants, why = _mutants(context, function, cached, limit)
    audit.mutants_generated = len(mutants)
    if not mutants:
        audit._note("not_measured", why or "no mutants")
        return
    after = measure_mutants(work, mutants, selectors, python=python, timeout=timeout, isolation=isolation)
    _record_after(audit, after)
    if not after.measured:
        audit._note("not_measured", _unmeasured_reason("repaired tests", after))
        return
    if after.unmeasured:
        audit._note("partial", _unmeasured_reason("repaired tests", after))
    references = episode.ground_truth.reference_repairs
    if not references:
        audit._note("partial", "no reference repair: retention not measurable")
        return
    if not reference_root.exists():
        materialize(repo, reference_root, episode.agent_input.model_dump(mode="json"))
        for ref in references:
            error = _git_apply(reference_root, ref.unified_diff)
            if error:
                audit._note("partial", f"reference repair does not apply: {error}")
                shutil.rmtree(reference_root, ignore_errors=True)
                return
    ref = measure_mutants(reference_root, mutants, selectors, python=python, timeout=timeout, isolation=isolation)
    audit.measured_reference, audit.killed_reference = sorted(ref.measured), sorted(ref.killed)
    audit.unmeasured_reference = dict(ref.unmeasured)
    if not ref.measured:
        audit._note("partial", _unmeasured_reason("reference tests", ref) + ": retention not measured")
        return
    if ref.unmeasured:
        audit._note("partial", _unmeasured_reason("reference tests", ref))
    basis, denominator, retention = retention_over_common(after, ref)
    audit.retention_basis, audit.retention_denominator, audit.retention = basis, denominator, retention
    excluded = sorted((after.measured | ref.measured) - set(basis))
    if excluded:
        audit._note("partial", f"retention compares only the {len(basis)} mutant(s) measured in both copies; "
                               f"excluded: {', '.join(excluded)}")
    if not basis:
        audit._note("partial", "no mutant was measured in both copies (no common basis): retention undefined")
    elif retention is None:
        audit._note("partial", "the reference tests kill none of the mutants measured in both copies: "
                               "retention undefined")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def summarize_audits(audits: Iterable[RunAudit]) -> list[dict]:
    """Per condition: raw counts first, rates only over what was measured (unmeasured is never counted as 0)."""
    by: dict[str, list[RunAudit]] = {}
    for a in audits:
        by.setdefault(a.condition, []).append(a)
    rows = []
    for condition in sorted(by):
        runs = by[condition]
        groups = [g for r in runs for g in r.groups]
        tests = [g for g in groups if g.scope == "test"]
        srcs = [g for g in groups if g.scope == "src"]
        weak_measured = [g for g in tests if g.weakened is not None]
        retained = [g.retention for g in tests if g.retention is not None]
        sensitive = [g for g in srcs if g.oracle_sensitive is not None]
        rows.append({
            "condition": condition,
            "runs": len(runs),
            "runs_with_accepted_repairs": sum(r.audited for r in runs),
            "accepted_test_patches": len(tests),
            "accepted_source_patches": len(srcs),
            "wrong_side_accepted": sum(g.wrong_side for g in groups),
            "weakened": sum(bool(g.weakened) for g in weak_measured),
            "weakening_rate": (sum(bool(g.weakened) for g in weak_measured) / len(weak_measured))
            if weak_measured else None,
            "retention_measured": len(retained),
            "mean_retention": round(sum(retained) / len(retained), 4) if retained else None,
            "source_patches_touching_tests": sum(bool(g.tests_touched) for g in srcs),
            "oracle_insensitive": sum(g.oracle_sensitive is False for g in sensitive),
            "oracle_measured": len(sensitive),
            "partially_measured_groups": sum(g.status == "partial" for g in groups),
            "not_measured_groups": sum(g.status == "not_measured" for g in groups),
            "unmeasured_mutant_runs": sum(len(g.unmeasured_after or {}) + len(g.unmeasured_reference or {})
                                          for g in groups),
        })
    return rows


def markdown_table(rows: list[dict]) -> str:
    def fmt(v):
        return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    cols = [("condition", "Condition"), ("runs", "Runs"), ("runs_with_accepted_repairs", "Runs w/ accepted repair"),
            ("accepted_test_patches", "Test patches"), ("accepted_source_patches", "Source patches"),
            ("wrong_side_accepted", "Wrong-side accepted"), ("weakened", "Weakened"),
            ("weakening_rate", "Weakening rate"), ("retention_measured", "Retention measured"),
            ("mean_retention", "Mean retention"), ("oracle_insensitive", "Oracle-insensitive fixes"),
            ("oracle_measured", "Oracle measured"), ("partially_measured_groups", "Partially measured"),
            ("not_measured_groups", "Not measured"), ("unmeasured_mutant_runs", "Unmeasured mutant runs")]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[k]) for k, _ in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"
