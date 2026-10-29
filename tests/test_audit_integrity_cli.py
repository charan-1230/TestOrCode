"""``scripts/audit_integrity.py``: input validation, duplicates, unknown episodes and safe output (D2).

Real: the CLI, the audit, git and mutant runs (``--isolation trusted-host`` on the fixture repository).  Replaced:
the split loader and the materialiser (the fixture episode and a clone of the fixture repository instead of the
pinned boltons clone).  Everything is written under ``tmp_path``; saved results are never read or written.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from testorcode.schemas.result_schemas import ExperimentCondition, PatchApplicationStatus

ROOT = Path(__file__).resolve().parents[1]
_audit_tests = importlib.util.spec_from_file_location("audit_fixtures", ROOT / "tests" / "test_integrity_audit.py")
fx = importlib.util.module_from_spec(_audit_tests)
_audit_tests.loader.exec_module(fx)

A3, B1 = ExperimentCondition.A3_NO_INTEGRITY_GUARD, ExperimentCondition.B1
JSONL, MD = "integrity_audit.jsonl", "integrity_audit.md"


def load_cli():
    spec = importlib.util.spec_from_file_location("audit_integrity_cli", ROOT / "scripts" / "audit_integrity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    base = _fixture_repo(tmp_path)
    module = load_cli()
    episode = fx.episode()
    monkeypatch.setattr(module, "_load_episodes", lambda splits: {episode.episode_id: episode})
    monkeypatch.setattr(module, "_materializer", lambda: fx.materialize)

    def run(runs: Path, *extra: str) -> int:
        return module.main(["--runs", str(runs), "--repo", str(base), "--python", sys.executable,
                            "--isolation", "trusted-host", "--timeout", "60", *extra])

    run.module = module
    return run


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "base"
    for rel, text in {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n',
                      "src/synthlib/__init__.py": "", "src/synthlib/core.py": fx.CORE,
                      fx.TEST_FILE: fx.OLD_TESTS}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "ci"]):
        fx.git(root, *args)
    return root


def accepted(condition=A3, run_id: str | None = None):
    record = fx.run_record(condition, fx.diff(fx.TEST_FILE, fx.OLD_TESTS, fx.LITERAL))
    if run_id:
        record = record.model_copy(update={"provenance": record.provenance.model_copy(update={"run_id": run_id})})
    return record


def reverted(condition=B1, run_id: str | None = None):
    """A run with nothing accepted: audited without running any mutant (fast)."""
    record = fx.run_record(condition, fx.diff(fx.TEST_FILE, fx.OLD_TESTS, fx.LITERAL), disposition="reverted",
                           status=PatchApplicationStatus.REVERTED)
    if run_id:
        record = record.model_copy(update={"provenance": record.provenance.model_copy(update={"run_id": run_id})})
    return record


def write_runs(directory: Path, *lines: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "runs.jsonl"
    path.write_text("".join(line + "\n" for line in lines))
    return path


def outputs(directory: Path) -> dict[str, str | None]:
    return {n: ((directory / n).read_text() if (directory / n).exists() else None) for n in (JSONL, MD)}


def table_row(markdown: str, condition: str) -> list[str]:
    [line] = [x for x in markdown.splitlines() if x.startswith(f"| {condition} |")]
    return [c.strip() for c in line.strip().strip("|").split("|")]


# --------------------------------------------------------------------------- #
# Valid input
# --------------------------------------------------------------------------- #


def test_valid_input_writes_both_reports_and_they_agree(cli, tmp_path) -> None:
    runs = write_runs(tmp_path / "exp", accepted().to_jsonl().strip(), reverted().to_jsonl().strip())
    assert cli(runs) == 0
    lines = [json.loads(x) for x in (tmp_path / "exp" / JSONL).read_text().splitlines()]
    assert [x["condition"] for x in lines] == ["A3", "B1"] and all(x["audit_version"] == 2 for x in lines)
    [g] = lines[0]["groups"]
    assert g["status"] == "measured" and 0 < g["retention"] < 1
    md = (tmp_path / "exp" / MD).read_text()
    a3 = table_row(md, "A3")
    assert a3[1:4] == ["1", "1", "1"] and a3[8] == "1" and float(a3[9]) == pytest.approx(g["retention"], abs=5e-4)
    assert table_row(md, "B1")[1:3] == ["1", "0"]  # one run, nothing accepted
    assert not list((tmp_path / "exp").glob(".integrity_audit*"))  # no temporaries left behind


# --------------------------------------------------------------------------- #
# Invalid input: refused before anything is audited or written
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("content, message", [
    ("", "no run records"),
    ("\n\n", "no run records"),
])
def test_empty_input_is_refused(cli, tmp_path, capsys, content, message) -> None:
    runs = tmp_path / "exp" / "runs.jsonl"
    runs.parent.mkdir()
    runs.write_text(content)
    assert cli(runs) == 2
    assert message in capsys.readouterr().err and outputs(runs.parent) == {JSONL: None, MD: None}


def test_missing_input_is_refused(cli, tmp_path, capsys) -> None:
    (tmp_path / "exp").mkdir()
    assert cli(tmp_path / "exp" / "runs.jsonl") == 2
    assert "input not found" in capsys.readouterr().err


def test_malformed_record_is_refused_with_its_line_number(cli, tmp_path, capsys) -> None:
    runs = write_runs(tmp_path / "exp", reverted().to_jsonl().strip(), '{"not": "a record"}', "{broken json")
    assert cli(runs) == 2
    err = capsys.readouterr().err
    assert "line 2:" in err and "line 3:" in err and "line 1:" not in err
    assert outputs(tmp_path / "exp") == {JSONL: None, MD: None}


def test_unknown_episode_refuses_the_whole_audit(cli, tmp_path, capsys) -> None:
    unknown = reverted(run_id="run-unknown").model_copy(update={"provenance": reverted().provenance.model_copy(
        update={"episode_id": "ep-9999", "run_id": "run-unknown"})})
    runs = write_runs(tmp_path / "exp", accepted().to_jsonl().strip(), unknown.to_jsonl().strip())  # mixed
    assert cli(runs) == 2
    err = capsys.readouterr().err
    assert "ep-9999 (run run-unknown)" in err and "nothing audited" in err
    assert outputs(tmp_path / "exp") == {JSONL: None, MD: None}  # no partial report


# --------------------------------------------------------------------------- #
# Duplicates and repeated runs
# --------------------------------------------------------------------------- #


def test_exact_duplicate_is_audited_once_and_reported(cli, tmp_path, capsys) -> None:
    line = reverted().to_jsonl().strip()
    runs = write_runs(tmp_path / "exp", line, line, line)
    assert cli(runs) == 0
    assert "line 2 repeats run run-B1" in capsys.readouterr().err
    assert len((tmp_path / "exp" / JSONL).read_text().splitlines()) == 1
    md = (tmp_path / "exp" / MD).read_text()
    assert "2 exact duplicate line(s) audited once: line 2 (run-B1), line 3 (run-B1)" in md
    assert table_row(md, "B1")[1] == "1"  # Runs: not inflated


def test_distinct_runs_sharing_an_episode_are_all_audited(cli, tmp_path, capsys) -> None:
    runs = write_runs(tmp_path / "exp", reverted(run_id="run-1").to_jsonl().strip(),
                      reverted(run_id="run-2").to_jsonl().strip())
    assert cli(runs) == 0
    episode = fx.episode().episode_id
    assert f"repeated runs (each audited): {episode} B1 seed=0 (2 runs)" in capsys.readouterr().err
    assert [json.loads(x)["run_id"] for x in (tmp_path / "exp" / JSONL).read_text().splitlines()] == [
        "run-1", "run-2"]
    md = (tmp_path / "exp" / MD).read_text()
    assert "Repeated runs" in md and table_row(md, "B1")[1] == "2"


def test_conflicting_records_with_one_run_id_are_refused(cli, tmp_path, capsys) -> None:
    runs = write_runs(tmp_path / "exp", reverted(run_id="same").to_jsonl().strip(),
                      accepted(condition=B1, run_id="same").to_jsonl().strip())
    assert cli(runs) == 2
    assert "run_id same: line 1 and line 2 differ" in capsys.readouterr().err
    assert outputs(tmp_path / "exp") == {JSONL: None, MD: None}


# --------------------------------------------------------------------------- #
# Output protection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("existing", [(JSONL, MD), (MD,), (JSONL,)])
def test_existing_reports_are_never_overwritten_without_permission(cli, tmp_path, capsys, existing) -> None:
    runs = write_runs(tmp_path / "exp", reverted().to_jsonl().strip())
    for name in existing:
        (tmp_path / "exp" / name).write_text(f"KEEP {name}\n")
    before = outputs(tmp_path / "exp")
    assert cli(runs) == 2
    assert "pass --overwrite" in capsys.readouterr().err
    assert outputs(tmp_path / "exp") == before


def test_overwrite_replaces_both_reports(cli, tmp_path) -> None:
    runs = write_runs(tmp_path / "exp", reverted().to_jsonl().strip())
    for name in (JSONL, MD):
        (tmp_path / "exp" / name).write_text("OLD\n")
    assert cli(runs, "--overwrite") == 0
    after = outputs(tmp_path / "exp")
    assert "OLD" not in after[JSONL] and "OLD" not in after[MD] and json.loads(after[JSONL].splitlines()[0])


@pytest.mark.parametrize("make", ["missing", "file"])
def test_invalid_output_directory_is_refused(cli, tmp_path, capsys, make) -> None:
    runs = write_runs(tmp_path / "exp", reverted().to_jsonl().strip())
    target = tmp_path / "out"
    if make == "file":
        target.write_text("not a directory")
    assert cli(runs, "--out-dir", str(target)) == 2
    assert "output directory" in capsys.readouterr().err
    assert outputs(tmp_path / "exp") == {JSONL: None, MD: None}


def test_output_dir_option_writes_elsewhere(cli, tmp_path) -> None:
    runs = write_runs(tmp_path / "exp", reverted().to_jsonl().strip())
    (tmp_path / "out").mkdir()
    assert cli(runs, "--out-dir", str(tmp_path / "out")) == 0
    assert outputs(tmp_path / "out")[JSONL] and outputs(tmp_path / "exp") == {JSONL: None, MD: None}


def test_a_failed_second_write_rolls_back_the_first(cli, tmp_path, monkeypatch) -> None:
    runs = write_runs(tmp_path / "exp", reverted().to_jsonl().strip())
    for name in (JSONL, MD):
        (tmp_path / "exp" / name).write_text(f"PREVIOUS {name}\n")
    before = outputs(tmp_path / "exp")
    real_replace = cli.module.os.replace

    def failing_replace(src, dst):
        if Path(dst).name == MD and Path(src).name.endswith(".tmp"):
            raise OSError("disk full (test)")
        return real_replace(src, dst)

    monkeypatch.setattr(cli.module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        cli(runs, "--overwrite")
    monkeypatch.setattr(cli.module.os, "replace", real_replace)
    assert outputs(tmp_path / "exp") == before  # both previous reports intact
    assert sorted(p.name for p in (tmp_path / "exp").iterdir()) == sorted([JSONL, MD, "runs.jsonl"])


def test_an_unexpected_error_during_the_audit_writes_nothing(cli, tmp_path, monkeypatch) -> None:
    """Exit status 1 (an uncaught error, traceback on stderr): no report, not even a partial one."""
    def broken(repo, dest, agent_input):
        raise RuntimeError("clone cannot be materialised (test)")

    monkeypatch.setattr(cli.module, "_materializer", lambda: broken)
    runs = write_runs(tmp_path / "exp", accepted().to_jsonl().strip())
    with pytest.raises(RuntimeError, match="cannot be materialised"):
        cli(runs)
    assert outputs(tmp_path / "exp") == {JSONL: None, MD: None}
    assert sorted(p.name for p in (tmp_path / "exp").iterdir()) == ["runs.jsonl"]
