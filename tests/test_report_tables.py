"""``report_tables`` / ``scripts/make_report.py``: the results table is built from records only; missing is not zero."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from testorcode.dataset import episode_path, load_episodes
from testorcode.experiment import dataset_version
from testorcode.report_tables import (
    NOT_EVALUATED,
    condition_rows,
    integrity_lines,
    markdown_table,
    write_plots,
)
from testorcode.schemas.episode_schemas import DecisionLabel as D
from testorcode.schemas.result_schemas import (
    AgentRole,
    ExperimentCondition,
    ExperimentRunRecord,
    PredictedDecision,
    RunOutcome,
    RunProvenance,
    RunStatus,
)

ROOT = Path(__file__).resolve().parents[1]
DEV = load_episodes(episode_path("dev"))
BY_ID = {e.episode_id: e for e in DEV}
VERSION = dataset_version("dev")
B1, A2 = ExperimentCondition.B1, ExperimentCondition.A2_NO_MEMORY


def run(eid: str, action: D | None, *, condition=B1, status=RunStatus.COMPLETED, version=VERSION) -> ExperimentRunRecord:
    tests = [lbl.test_id for lbl in BY_ID[eid].ground_truth.per_test_labels]
    preds = [] if action is None else [
        PredictedDecision(test_id=t, predicted_action=action, decided_by=AgentRole.SINGLE_AGENT) for t in tests
    ]
    return ExperimentRunRecord(
        provenance=RunProvenance(run_id=f"{eid}-{condition.value}", episode_id=eid, condition=condition, seed=0,
                                 model_id="test-model-1", split="dev", dataset_version=version),
        predictions=preds, outcome=RunOutcome(status=status),
    )


def rows_for(records, conditions=(B1, A2)):
    return {r["condition"]: r for r in condition_rows(records, DEV, dataset_version=VERSION, conditions=conditions)}


def test_counts_are_raw_and_a_missing_condition_is_not_zero() -> None:
    rows = rows_for([run("dev-03", D.UPDATE_TEST), run("dev-05", D.UPDATE_TEST)])  # dev-03 gold FIX_CODE x2: masked
    b1 = rows["B1"]
    assert b1["evaluated"] and b1["episodes_run"] == 2
    assert (b1["masked"], b1["gold_fix"]) == (2, 2)
    assert (b1["direction_correct"], b1["direction_scored"]) == (1, 3)
    a2 = rows["A2"]
    assert not a2["evaluated"] and a2["direction_macro_f1"] is None and a2["masked"] is None
    text = markdown_table(list(rows.values()), title="t")
    assert "2/2 (1.00)" in text and f"| A2 | {NOT_EVALUATED} |" in text and "Preliminary" in text


def test_execution_failures_are_counted_apart_and_a_failed_only_condition_is_not_evaluated() -> None:
    rows = rows_for([run("dev-05", D.UPDATE_TEST), run("dev-02", None, status=RunStatus.API_FAILURE),
                     run("dev-05", None, condition=A2, status=RunStatus.API_FAILURE)])
    assert rows["B1"]["episodes_run"] == 1 and rows["B1"]["execution_failures"] == 1
    assert not rows["A2"]["evaluated"] and rows["A2"]["execution_failures"] == 1
    assert "| A2 | not yet evaluated | 1 |" in markdown_table(list(rows.values()), title="t")


def test_records_of_another_dataset_revision_are_not_counted() -> None:
    rows = rows_for([run("dev-05", D.UPDATE_TEST, version="pilot@old")])
    assert not rows["B1"]["evaluated"] and rows["B1"]["episodes_run"] == 0


def test_integrity_lines_do_not_count_unmeasured_as_zero(tmp_path) -> None:
    group = {"scope": "test", "weakened": None, "wrong_side": False, "status": "not_measured"}
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({"condition": "A3", "groups": [group, {**group, "weakened": False, "status": "measured"}]}) + "\n")
    [line] = integrity_lines(path)
    assert "weakened 0/1 measured" in line and "1 not fully measured" in line


def test_plots_skip_unevaluated_conditions_and_write_pngs(tmp_path) -> None:
    rows = list(rows_for([run("dev-05", D.UPDATE_TEST)]).values())
    paths = write_plots(rows, tmp_path)
    assert {p.name for p in paths} == {"direction_macro_f1.png", "repair_vs_masked.png", "scenario_accuracy.png"}
    assert all(p.stat().st_size > 1000 for p in paths)
    assert write_plots([r for r in rows if not r["evaluated"]], tmp_path / "none") == []


def test_cli_writes_report_next_to_runs(tmp_path, capsys) -> None:
    spec = importlib.util.spec_from_file_location("make_report_cli", ROOT / "scripts" / "make_report.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    runs = tmp_path / "runs.jsonl"
    runs.write_text(run("dev-05", D.UPDATE_TEST).to_jsonl() + "\n")
    assert cli.main([str(runs), "--no-plots"]) == 0
    text = (tmp_path / "report.md").read_text()
    assert "| B1 | 1 |" in text and "not a statistical comparison" in text and "No integrity audit" in text
    (tmp_path / "empty.jsonl").write_text("")
    assert cli.main([str(tmp_path / "empty.jsonl")]) == 2
