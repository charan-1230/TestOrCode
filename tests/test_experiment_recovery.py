# ruff: noqa: F811
"""Recovery of a FULL stream after a provider failure or an interrupted run (simulated, no network).

What is checked: a failed run writes no memory; a resume never duplicates a memory record or a run line; memory is
never read from a *later* episode of the stream.  The fixture stream, scripted model and runner are those of
``test_experiment_full``; failures are injected as scripted exceptions or by interrupting the runner.
Current behaviour that is *documented*, not endorsed: ``run_experiment`` treats any record, an ``api_failure`` one
included, as done, so a resume never retries it unless ``retry_failed`` is set (tests at the end), which replaces the failed line and is refused when it would break
the memory stream's order.
"""

from __future__ import annotations

import json

import pytest
from test_experiment_full import (
    A2,
    FULL,
    SETTINGS,
    default_run_one,
    memory_dir,
    sandbox,
    scripted,
    stream,  # noqa: F401  (a fixture, used by name as a test argument)
    update_test_llm,
)

from testorcode import experiment
from testorcode.errors import LLMProviderOutageError
from testorcode.experiment import ExperimentConfig, read_records, run_experiment
from testorcode.memory import memory_path
from testorcode.schemas.result_schemas import AgentRole, RunStatus

pytestmark = pytest.mark.skipif(not sandbox.bwrap_available(), reason="bubblewrap unavailable")


def run(tmp_path, stream, monkeypatch, conditions, seeds, clients, *, retry_failed=False):
    """``test_experiment_full.run``, but every call (a first run, a resume) gets fresh worktrees, as a real rerun does."""
    pairs, materialize = stream
    attempt = len(list(tmp_path.glob("wt-*")))
    config = ExperimentConfig(conditions=tuple(conditions), seeds=tuple(seeds), out=tmp_path / "out",
                              worktrees=tmp_path / f"wt-{attempt}", repo=tmp_path / "unused", retry_failed=retry_failed)
    ordered = sorted(pairs, key=lambda p: p[1].stream_position)
    monkeypatch.setattr(experiment, "load_split", lambda cfg: ordered)
    (tmp_path / f"wt-{attempt}").mkdir()
    run_one = default_run_one(config, settings=SETTINGS, llm_factory=scripted(clients), materializer=materialize)
    result = run_experiment(config, run_one=run_one)
    return config, result, read_records(config.out / "runs.jsonl")


def outage() -> object:
    from testorcode.llm_client import FakeLLMClient

    return FakeLLMClient({AgentRole.INTENT_ANALYST: [LLMProviderOutageError("simulated 503")]})


def stored_ids(config) -> list[str]:
    path = memory_path(memory_dir(config), "FULL", 0, "dev")
    if not path.exists():
        return []
    return [json.loads(line)["record"]["record_id"] for line in path.read_text().splitlines() if line.strip()]


def first_ok():
    return update_test_llm("== 'a'", "== 'A'", "ev-msg")


def second_ok(cited="ev-msg"):
    return update_test_llm("== 'b'", "== 'B'", cited)


def test_an_api_failure_writes_no_memory_and_the_stream_goes_on(tmp_path, stream, monkeypatch) -> None:
    config, result, records = run(tmp_path, stream, monkeypatch, [FULL], [0],
                                  {(FULL, 0): [outage(), second_ok()]})
    failed, later = records
    assert failed.outcome.status is RunStatus.API_FAILURE and failed.predictions == []
    assert not any(w.written for w in failed.memory_writes)
    assert later.outcome.status is RunStatus.COMPLETED and later.memory_reads == []  # nothing from the failed episode
    assert stored_ids(config) == ["ep-0002-g1"]
    assert result.ran == 2


def test_a_resume_treats_an_api_failure_as_done_and_changes_nothing(tmp_path, stream, monkeypatch) -> None:
    config, _, before = run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [outage(), second_ok()]})
    memory_before = memory_path(memory_dir(config), "FULL", 0, "dev").read_text()

    config2, result, after = run(tmp_path, stream, monkeypatch, [FULL], [0], {})  # a model call would raise KeyError
    assert config2.out == config.out
    assert result.ran == 0 and result.skipped_done == 2
    assert [r.provenance.run_id for r in after] == [r.provenance.run_id for r in before]  # no duplicate lines
    assert memory_path(memory_dir(config), "FULL", 0, "dev").read_text() == memory_before


def test_an_interrupted_run_resumes_without_duplicate_memory_or_lines(tmp_path, stream, monkeypatch) -> None:
    real = experiment.scored
    calls = {"n": 0}

    def crash_on_second(record, gold, split):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash after the Guard wrote memory, before the line was appended")
        return real(record, gold, split)

    monkeypatch.setattr(experiment, "scored", crash_on_second)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(tmp_path, stream, monkeypatch, [FULL], [0],
            {(FULL, 0): [first_ok(), second_ok("ep-0001-g1")]})
    out = tmp_path / "out"
    assert [r.episode_id for r in read_records(out / "runs.jsonl")] == ["ep-0001"]  # ep-0002's line never written
    path = memory_path(memory_dir(experiment.ExperimentConfig(conditions=(FULL,), out=out)), "FULL", 0, "dev")
    assert {json.loads(x)["record"]["record_id"] for x in path.read_text().splitlines()} == {
        "ep-0001-g1", "ep-0002-g1"}  # the crashed episode had already written its record

    monkeypatch.setattr(experiment, "scored", real)
    config, result, records = run(tmp_path, stream, monkeypatch, [FULL], [0],
                                  {(FULL, 0): [second_ok("ep-0001-g1")]})  # identical replay of ep-0002
    assert result.ran == 1 and result.skipped_done == 1
    assert [r.episode_id for r in records] == ["ep-0001", "ep-0002"]
    ids = stored_ids(config)
    assert sorted(ids) == ["ep-0001-g1", "ep-0002-g1"] and len(ids) == len(set(ids))  # no duplicate write


def test_a_changed_replay_never_overwrites_a_stored_record(tmp_path, stream, monkeypatch) -> None:
    real = experiment.scored
    calls = {"n": 0}

    def crash_on_second(record, gold, split):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real(record, gold, split)

    monkeypatch.setattr(experiment, "scored", crash_on_second)
    with pytest.raises(RuntimeError):
        run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [first_ok(), second_ok("ep-0001-g1")]})
    path = memory_path(memory_dir(experiment.ExperimentConfig(conditions=(FULL,), out=tmp_path / "out")),
                       "FULL", 0, "dev")
    original = path.read_text()
    monkeypatch.setattr(experiment, "scored", real)

    # The replay decides differently (cites another id), so its record differs from the stored ep-0002-g1.
    _, _, records = run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [second_ok("ev-msg")]})
    lines = path.read_text().splitlines()
    ids = [json.loads(x)["record"]["record_id"] for x in lines]
    assert len(ids) == len(set(ids))  # still one line per record id
    assert original.splitlines()[0] in lines  # the ep-0001 record is untouched
    assert records[-1].episode_id == "ep-0002"  # the replay produced a line, whatever its status


def lines(config) -> list[str]:
    return [json.loads(x)["provenance"]["run_id"] for x in (config.out / "runs.jsonl").read_text().splitlines()]


def test_the_default_never_retries_a_failed_run(tmp_path, stream, monkeypatch) -> None:
    run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [outage(), second_ok()]})
    _, result, records = run(tmp_path, stream, monkeypatch, [FULL], [0], {})  # retry_failed off: no model call
    assert result.ran == 0 and result.retried == 0 and result.skipped_done == 2
    assert [r.outcome.status for r in records] == [RunStatus.API_FAILURE, RunStatus.COMPLETED]


def test_a_memory_free_condition_retries_a_failed_run_and_replaces_its_line(tmp_path, stream, monkeypatch) -> None:
    first, _, before = run(tmp_path, stream, monkeypatch, [A2], [0],
                           {(A2, 0): [outage(), second_ok()]})
    failed_id = before[0].provenance.run_id
    config, result, records = run(tmp_path, stream, monkeypatch, [A2], [0], {(A2, 0): [first_ok()]}, retry_failed=True)
    assert result.ran == 1 and result.retried == 1 and result.skipped_done == 1
    assert len(records) == 2 and len(set(lines(config))) == 2  # one line per run, no duplicate
    assert failed_id not in lines(config)  # the failed line is gone
    assert {r.episode_id: r.outcome.status for r in records} == {
        "ep-0001": RunStatus.COMPLETED, "ep-0002": RunStatus.COMPLETED}
    assert first.out == config.out


def test_a_failed_last_episode_of_a_memory_stream_retries_in_order(tmp_path, stream, monkeypatch) -> None:
    config, _, before = run(tmp_path, stream, monkeypatch, [FULL], [0],
                            {(FULL, 0): [first_ok(), outage()]})
    assert before[1].outcome.status is RunStatus.API_FAILURE and stored_ids(config) == ["ep-0001-g1"]
    _, result, records = run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [second_ok("ep-0001-g1")]},
                             retry_failed=True)
    assert result.ran == 1 and result.retried == 1 and len(records) == 2
    retried = records[1]
    assert retried.outcome.status is RunStatus.COMPLETED
    assert [m.record_id for m in retried.memory_reads] == ["ep-0001-g1"]  # the earlier episode's record, in order
    ids = stored_ids(config)
    assert sorted(ids) == ["ep-0001-g1", "ep-0002-g1"] and len(ids) == len(set(ids))  # no duplicate memory write


def test_an_unsafe_retry_is_refused_before_anything_runs(tmp_path, stream, monkeypatch) -> None:
    config, _, _ = run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [outage(), second_ok()]})
    runs_before = (config.out / "runs.jsonl").read_text()
    memory_before = memory_path(memory_dir(config), "FULL", 0, "dev").read_text()
    with pytest.raises(ValueError, match=r"fresh --out") as caught:
        run(tmp_path, stream, monkeypatch, [FULL], [0], {}, retry_failed=True)  # a model call would raise KeyError
    assert "ep-0001/FULL/seed 0" in str(caught.value) and "ep-0002" in str(caught.value)
    assert (config.out / "runs.jsonl").read_text() == runs_before  # nothing changed
    assert memory_path(memory_dir(config), "FULL", 0, "dev").read_text() == memory_before


def test_a_failed_attempt_that_already_wrote_memory_is_not_retried(tmp_path, stream, monkeypatch) -> None:
    from testorcode.experiment import unsafe_retries
    from testorcode.schemas.result_schemas import MemoryWriteAttempt

    pairs, _ = stream
    ordered = sorted(pairs, key=lambda p: p[1].stream_position)
    config, _, records = run(tmp_path, stream, monkeypatch, [FULL], [0], {(FULL, 0): [first_ok(), outage()]})
    wrote = MemoryWriteAttempt(record_id="ep-0002-g1", approved=True, written=True)
    tainted = [records[0], records[1].model_copy(update={"memory_writes": [wrote]})]
    problems = unsafe_retries(ordered, config.model_copy(update={"retry_failed": True}), tainted)
    assert problems and "already wrote memory" in problems[0]
    assert unsafe_retries(ordered, config, records) == []  # the untouched failed last episode is safe


def test_a_retry_that_fails_again_keeps_a_single_line(tmp_path, stream, monkeypatch) -> None:
    run(tmp_path, stream, monkeypatch, [A2], [0], {(A2, 0): [outage(), second_ok()]})
    config, result, records = run(tmp_path, stream, monkeypatch, [A2], [0], {(A2, 0): [outage()]}, retry_failed=True)
    assert result.retried == 1 and len(records) == 2 and len(set(lines(config))) == 2
    assert records[-1].episode_id == "ep-0001" and records[-1].outcome.status is RunStatus.API_FAILURE


def test_a_retry_that_cannot_start_leaves_the_failed_line_in_place(tmp_path, stream, monkeypatch) -> None:
    from testorcode.errors import ConfigError

    config, _, before = run(tmp_path, stream, monkeypatch, [A2], [0], {(A2, 0): [outage(), second_ok()]})

    def unavailable(settings):
        raise ConfigError("no model configured")

    _, materialize = stream
    cfg = ExperimentConfig(conditions=(A2,), seeds=(0,), out=config.out, worktrees=tmp_path / "wt-x",
                           repo=tmp_path / "u", retry_failed=True)
    (tmp_path / "wt-x").mkdir()
    result = run_experiment(cfg, run_one=default_run_one(cfg, settings=SETTINGS, llm_factory=unavailable,
                                                         materializer=materialize))
    assert result.ran == 0 and len(result.skipped_unavailable) == 1
    assert [r.provenance.run_id for r in read_records(config.out / "runs.jsonl")] == [
        r.provenance.run_id for r in before]


def test_the_cli_flag_is_opt_in_and_a_refusal_exits_2(tmp_path, monkeypatch, capsys) -> None:
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_experiment_cli", Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = {}

    def fake(config):
        seen["retry"] = config.retry_failed
        raise ValueError("refusing to retry: x. re-run into a fresh --out.")

    monkeypatch.setattr(cli, "run_experiment", fake)
    base = ["run_experiment.py", "--conditions", "FULL", "--out", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", base)
    assert cli.main() == 2 and seen["retry"] is False
    monkeypatch.setattr(sys, "argv", [*base, "--retry-failed"])
    assert cli.main() == 2 and seen["retry"] is True and "fresh --out" in capsys.readouterr().err
