"""Post-hoc integrity audit (``testorcode.integrity_audit``) on a fixture repository.

Real: git, ``tools.weakening``, generated mutants and ``run_mutants`` (pytest under ``isolation="trusted-host"``,
which the interfaces reserve for trusted fixtures such as this one).  Synthetic: the repository, the episode and the
run records (built directly, as the evaluation tests do).
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import subprocess
from pathlib import Path

import pytest

from testorcode.integrity_audit import accepted_patches, audit_run, markdown_table, summarize_audits
from testorcode.schemas.episode_schemas import DecisionLabel as D
from testorcode.schemas.episode_schemas import (
    MutantSpec,
    PatchScope,
    ReferenceRepair,
    RepairDirection,
)
from testorcode.schemas.result_schemas import (
    AgentRole,
    ExperimentCondition,
    ExperimentRunRecord,
    FailureGroupRecord,
    PatchApplicationStatus,
    PatchRecord,
    PredictedDecision,
    RunOutcome,
    RunProvenance,
    RunStatus,
)
from testorcode.schemas.tests.conftest import NORM, T_NORM, label, make_s1
from testorcode.tools.mutation import MutantOutcome as mutation_outcome

TEST_FILE = "tests/test_synthlib.py"
CORE = (
    "def normalise(s):\n"
    "    s = s.strip()\n"
    "    if len(s) > 3:\n"
    "        return s[:3].capitalize()\n"
    "    return s.capitalize()\n"
)
OLD_TESTS = "from synthlib.core import normalise\n\n\ndef test_normalise():\n    assert normalise('ab') == 'ab'\n"
LITERAL = OLD_TESTS.replace("== 'ab'", "== 'Ab'")  # the agent's literal update
# The developer's repair: the same update plus a stronger check of the truncation branch.
REFERENCE = LITERAL + "    assert normalise(' abcdef ') == 'Abc'\n"
WEAKENED = OLD_TESTS.replace("assert normalise('ab') == 'ab'", "assert normalise('ab')")  # == -> truthiness
FIXED_CORE = CORE  # the source side is unchanged by a test repair


def diff(path: str, old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{path}", f"b/{path}"))


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture
def base(tmp_path: Path) -> Path:
    root = tmp_path / "base"
    for rel, text in {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n',
                      "src/synthlib/__init__.py": "", "src/synthlib/core.py": CORE, TEST_FILE: OLD_TESTS}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci state")
    return root


def materialize(repo: Path, dest: Path, agent_input: dict) -> Path:
    subprocess.run(["git", "clone", "-q", str(repo), str(dest)], check=True, capture_output=True)
    return dest


def episode():
    s1 = make_s1()
    truth = s1.ground_truth.model_copy(update={
        "reference_repairs": [ReferenceRepair(unified_diff=diff(TEST_FILE, OLD_TESTS, REFERENCE),
                                              changed_files=[TEST_FILE])],
        "per_test_labels": [label(T_NORM, RepairDirection.UPDATE_TEST, function=NORM, evidence=["ev-msg"])],
    })
    return s1.model_copy(update={"ground_truth": truth})


def run_record(condition, patch_diff: str, *, scope=PatchScope.TEST, disposition="accepted",
               status=PatchApplicationStatus.APPLIED) -> ExperimentRunRecord:
    ep = episode()
    return ExperimentRunRecord(
        provenance=RunProvenance(run_id=f"run-{condition.value}", episode_id=ep.episode_id, condition=condition,
                                 seed=0, model_id="test-model-1"),
        predictions=[PredictedDecision(test_id=T_NORM, predicted_action=D.UPDATE_TEST,
                                       decided_by=AgentRole.INTENT_ANALYST)],
        failure_groups=[FailureGroupRecord(group_id="g1", test_ids=[T_NORM], decision=D.UPDATE_TEST,
                                           patch_ids=["g1-r0"], disposition=disposition)],
        patches=[PatchRecord(patch_id="g1-r0", scope=scope, test_ids=[T_NORM], target_files=[TEST_FILE],
                             unified_diff=patch_diff.rstrip("\n"), application_status=status)],
        outcome=RunOutcome(status=RunStatus.COMPLETED),
    )


def audit(record, base: Path, tmp_path: Path):
    return audit_run(record, episode(), repo=base, materialize=materialize, python=None, isolation="trusted-host",
                     timeout_seconds=60, scratch=tmp_path / f"scratch-{record.condition.value}")


A3, B1 = ExperimentCondition.A3_NO_INTEGRITY_GUARD, ExperimentCondition.B1


def test_literal_update_is_not_weakening_and_retention_is_measured_against_the_reference(base, tmp_path) -> None:
    result = audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path)
    assert result.audited and result.condition == "A3"
    [g] = result.groups
    assert g.scope == "test" and g.function == NORM and g.gold_directions == ["UPDATE_TEST"] and not g.wrong_side
    assert g.weakened is False and g.changed_tests == [f"{TEST_FILE}::test_normalise"]
    # Fully measured: every generated mutant ran to killed/survived in both copies.
    assert g.status == "measured" and g.reasons == []
    assert g.mutants_generated and len(g.measured_after) == len(g.measured_reference) == g.mutants_generated
    assert g.unmeasured_after == {} and g.unmeasured_reference == {}
    assert g.retention_basis == g.measured_after
    assert set(g.killed_after) <= set(g.killed_reference)  # the reference adds an assertion, never fewer kills
    assert len(g.killed_reference) > len(g.killed_after)  # it also checks the truncation branch
    assert g.retention_denominator == len(g.killed_reference)
    assert g.retention == round(len(g.killed_after) / len(g.killed_reference), 4) and 0 < g.retention < 1
    assert (base / TEST_FILE).read_text() == OLD_TESTS  # the source repository is never written


def test_weakened_assertion_is_detected_without_a_guard(base, tmp_path) -> None:
    [g] = audit(run_record(B1, diff(TEST_FILE, OLD_TESTS, WEAKENED)), base, tmp_path).groups
    assert g.weakened is True and g.weakening_problems
    assert g.retention is not None and g.retention < 1  # truthiness kills fewer mutants than the reference


def test_unaccepted_repairs_are_not_audited(base, tmp_path) -> None:
    reverted = run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL), disposition="reverted",
                          status=PatchApplicationStatus.REVERTED)
    assert accepted_patches(reverted) == []
    result = audit(reverted, base, tmp_path)
    assert not result.audited and result.reason == "no accepted repair to audit" and result.groups == []


def test_source_fix_reports_oracle_sensitivity_and_a_wrong_side_edit(base, tmp_path) -> None:
    fix = diff("src/synthlib/core.py", CORE, CORE.replace("return s.capitalize()\n", "return s\n"))
    [g] = audit(run_record(A3, fix, scope=PatchScope.SRC), base, tmp_path).groups
    assert g.scope == "src" and g.tests_touched is False
    assert g.wrong_side is True  # gold is UPDATE_TEST: the source was edited (reverted intent)
    assert g.oracle_sensitive is not None and g.mutants_generated and g.unmeasured_after == {}
    assert len(g.measured_after) == g.mutants_generated and g.status == "measured"


def test_patch_that_no_longer_applies_is_not_measured(base, tmp_path) -> None:
    stale = diff(TEST_FILE, OLD_TESTS.replace("'ab'", "'zz'"), LITERAL)
    [g] = audit(run_record(A3, stale), base, tmp_path).groups
    assert g.status == "not_measured" and "does not re-apply" in g.reasons[0] and g.weakened is None


def test_summary_counts_measured_repairs_only(base, tmp_path) -> None:
    audits = [audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path),
              audit(run_record(B1, diff(TEST_FILE, OLD_TESTS, WEAKENED)), base, tmp_path)]
    rows = {r["condition"]: r for r in summarize_audits(audits)}
    assert rows["A3"]["weakened"] == 0 and rows["A3"]["weakening_rate"] == 0
    assert rows["B1"]["weakened"] == 1 and rows["B1"]["weakening_rate"] == 1
    assert rows["A3"]["oracle_measured"] == 0 and rows["A3"]["mean_retention"] is not None
    assert "| A3 |" in markdown_table(list(rows.values()))


# --------------------------------------------------------------------------- #
# D1 regressions: an unmeasured mutant is never a "not killed" mutant
# --------------------------------------------------------------------------- #

FIX = diff("src/synthlib/core.py", CORE, CORE.replace("return s.capitalize()\n", "return s\n"))
BOGUS = "--- a/src/synthlib/core.py\n+++ b/src/synthlib/core.py\n@@ -1 +1 @@\n-no such line\n+x\n"


def bogus_mutants(n: int = 3) -> dict:
    return {NORM: [MutantSpec(mutant_id=f"bogus-{i}", function_qualname=NORM, file_path="src/synthlib/core.py",
                              unified_diff=BOGUS) for i in range(n)]}


def rewrite_outcomes(monkeypatch, change):
    """Run the real mutants, then pass each outcome through ``change(root_name, outcome) -> outcome``."""
    from testorcode import integrity_audit
    from testorcode.tools import mutation

    real = mutation.run_mutants

    def wrapped(root, *args, **kwargs):
        run = real(root, *args, **kwargs)
        return dataclasses.replace(run, outcomes=tuple(change(Path(root).name, o) for o in run.outcomes))

    monkeypatch.setattr(integrity_audit, "run_mutants", wrapped)


def as_timeout(o):
    return mutation_outcome(o.mutant_id, "timeout", detail="stopped (test)")


def roundtrip(result):
    """Serialization path: the JSONL line must carry the same verdicts."""
    return json.loads(json.dumps(result.to_dict(), sort_keys=True))["groups"]


def row(audits) -> dict:
    [r] = summarize_audits(audits)
    cells = [c.strip() for c in markdown_table([r]).splitlines()[2].strip("|").split("|")]
    return r, cells


def test_all_mutants_unmeasured_is_not_measured_not_an_insensitive_oracle(base, tmp_path) -> None:
    """The exact audit reproduction (D1): every mutant invalid used to give oracle_sensitive=False and count as an
    oracle-insensitive fix with 'oracle measured = 1' and 'not measured = 0'."""
    result = audit_run(run_record(A3, FIX, scope=PatchScope.SRC), episode(), repo=base, materialize=materialize,
                       python=None, isolation="trusted-host", timeout_seconds=60, cached_mutants=bogus_mutants(),
                       scratch=tmp_path / "s")
    [g] = result.groups
    assert g.status == "not_measured" and g.oracle_sensitive is None
    assert g.mutants_generated == 3 and g.measured_after == [] and g.killed_after == []
    assert g.unmeasured_after == {"bogus-0": "invalid", "bogus-1": "invalid", "bogus-2": "invalid"}
    assert g.reasons == ["repaired code: 3 mutant(s) not measured (invalid)"]
    [j] = roundtrip(result)
    assert j["oracle_sensitive"] is None and j["status"] == "not_measured" and j["measured_after"] == []
    r, cells = row([result])
    assert r["oracle_insensitive"] == 0 and r["oracle_measured"] == 0
    assert r["not_measured_groups"] == 1 and r["unmeasured_mutant_runs"] == 3
    assert cells[10:15] == ["0", "0", "0", "1", "3"]  # insensitive, oracle measured, partial, not measured, unmeasured


def test_all_mutants_timing_out_is_not_measured_for_both_scopes(base, tmp_path, monkeypatch) -> None:
    rewrite_outcomes(monkeypatch, lambda root, o: as_timeout(o))
    src = audit(run_record(A3, FIX, scope=PatchScope.SRC), base, tmp_path)
    test = audit(run_record(B1, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path)
    [s], [t] = src.groups, test.groups
    assert s.status == "not_measured" and s.oracle_sensitive is None and s.measured_after == []
    assert set(s.unmeasured_after.values()) == {"timeout"}
    assert t.status == "not_measured" and t.retention is None and t.retention_denominator is None
    assert t.measured_after == [] and t.measured_reference is None  # the reference is not even attempted
    assert t.weakened is False  # weakening does not depend on mutants and is still measured
    rows = {r["condition"]: r for r in summarize_audits([src, test])}
    assert rows["A3"]["oracle_measured"] == 0 and rows["A3"]["oracle_insensitive"] == 0
    assert rows["B1"]["retention_measured"] == 0 and rows["B1"]["mean_retention"] is None
    assert rows["B1"]["weakening_rate"] == 0


def test_partial_measurement_without_a_kill_leaves_the_oracle_undetermined(base, tmp_path, monkeypatch) -> None:
    # The real run kills some mutants and lets others survive; every kill becomes a timeout (unmeasured).
    rewrite_outcomes(monkeypatch, lambda root, o: as_timeout(o) if o.outcome == "killed" else o)
    [g] = audit(run_record(A3, FIX, scope=PatchScope.SRC), base, tmp_path).groups
    assert g.measured_after and g.unmeasured_after, "fixture must leave survivors and unmeasured mutants"
    assert g.killed_after == [] and g.oracle_sensitive is None and g.status == "partial"
    assert any("sensitivity undetermined" in r for r in g.reasons)


def test_one_kill_proves_sensitivity_even_when_others_are_unmeasured(base, tmp_path, monkeypatch) -> None:
    rewrite_outcomes(monkeypatch, lambda root, o: as_timeout(o) if o.outcome == "survived" else o)
    [g] = audit(run_record(A3, FIX, scope=PatchScope.SRC), base, tmp_path).groups
    assert g.killed_after and g.oracle_sensitive is True
    assert g.status == ("partial" if g.unmeasured_after else "measured")


def test_retention_compares_only_mutants_measured_in_both_copies(base, tmp_path) -> None:
    """A mutant the reference kills but that timed out with the agent's tests must leave the basis, not count as
    'lost' (which would understate retention)."""
    full = audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path / "full")
    [f] = full.groups
    lost_only_in_reference = sorted(set(f.killed_reference) - set(f.killed_after))
    assert lost_only_in_reference, "fixture: the reference must kill a mutant the agent's tests do not"
    dropped = lost_only_in_reference[0]

    from _pytest.monkeypatch import MonkeyPatch

    with MonkeyPatch.context() as mp:
        rewrite_outcomes(mp, lambda root, o: as_timeout(o) if root == "run" and o.mutant_id == dropped else o)
        [g] = audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path / "partial").groups
    assert g.unmeasured_after == {dropped: "timeout"} and g.unmeasured_reference == {}
    assert dropped not in g.retention_basis and len(g.retention_basis) == g.mutants_generated - 1
    expected_den = len(set(g.killed_reference) - {dropped})
    assert g.retention_denominator == expected_den
    assert g.retention == round(len(set(g.killed_after) & set(g.killed_reference)) / expected_den, 4)
    assert g.retention > f.retention  # the old formula counted the unmeasured mutant as lost
    assert g.status == "partial" and any("measured in both copies" in r for r in g.reasons)
    [j] = roundtrip(audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path / "again"))
    assert j["retention"] == f.retention  # without the injected timeout the fully measured value is unchanged


def test_unmeasured_reference_means_retention_not_measured(base, tmp_path, monkeypatch) -> None:
    rewrite_outcomes(monkeypatch, lambda root, o: as_timeout(o) if root == "reference" else o)
    result = audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path)
    [g] = result.groups
    assert g.measured_after and g.measured_reference == [] and g.killed_reference == []
    assert g.retention is None and g.retention_denominator is None and g.status == "partial"
    r, _ = row([result])
    assert r["retention_measured"] == 0 and r["mean_retention"] is None and r["partially_measured_groups"] == 1


def test_disjoint_measured_sets_have_no_common_basis(base, tmp_path, monkeypatch) -> None:
    """The agent's copy measures only odd-numbered mutants, the reference only even-numbered ones."""
    def change(root, o):
        odd = int(o.mutant_id.rsplit("-", 1)[-1]) % 2 == 1
        keep = odd if root == "run" else not odd
        return o if keep else as_timeout(o)

    rewrite_outcomes(monkeypatch, change)
    result = audit(run_record(A3, diff(TEST_FILE, OLD_TESTS, LITERAL)), base, tmp_path)
    [g] = result.groups
    assert g.measured_after and g.measured_reference, "fixture: each copy must measure something"
    assert not set(g.measured_after) & set(g.measured_reference)
    assert g.retention_basis == [] and g.retention_denominator == 0 and g.retention is None
    assert g.status == "partial"
    assert any("no mutant was measured in both copies (no common basis)" in r for r in g.reasons)
    assert not any("reference tests kill none" in r for r in g.reasons)
    r, _ = row([result])
    assert r["retention_measured"] == 0 and r["mean_retention"] is None
