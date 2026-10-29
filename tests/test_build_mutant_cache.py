"""``scripts/build_mutant_cache.py``: an unmeasured catalogue entry is never written as plausible zero counts (D3).

Real: the CLI, mutant generation and ``run_mutants`` (``--isolation trusted-host`` on a fixture repository).
Replaced: the dev split loader and the ``git archive`` export of the pinned clone (the fixture files instead).
Everything is written under ``tmp_path``; the committed catalogue is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from testorcode.schemas.episode_schemas import MutantSpec, RepairDirection
from testorcode.schemas.result_schemas import TestRunStatus
from testorcode.schemas.tests.conftest import NORM, T_NORM, label, make_s1
from testorcode.tools.mutation import MutationRun

ROOT = Path(__file__).resolve().parents[1]
CORE = "def normalise(s):\n    return s.lower()\n"
TESTS = "from synthlib.core import normalise\n\n\ndef test_normalise():\n    assert normalise('AB') == 'ab'\n"


def load_cli():
    spec = importlib.util.spec_from_file_location("build_mutant_cache_cli", ROOT / "scripts" / "build_mutant_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    module = load_cli()
    episode = make_s1()
    files = {"pyproject.toml": '[tool.pytest.ini_options]\npythonpath = ["src"]\n',
             "src/synthlib/__init__.py": "", "src/synthlib/core.py": CORE, "tests/test_synthlib.py": TESTS}

    def export(repo, sha, dest):
        for rel, text in files.items():
            (Path(dest) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(dest) / rel).write_text(text)

    monkeypatch.setattr(module, "export", export)
    monkeypatch.setattr(module, "load_episodes", lambda path: [episode])
    out = tmp_path / "catalogue.json"
    rejected = tmp_path / "catalogue.json.incomplete"

    def run(*extra: str) -> tuple[int, dict]:
        code = module.main(["--python", sys.executable, "--isolation", "trusted-host", "--timeout", "60",
                            "--episodes", episode.episode_id, "--out", str(out), *extra])
        return code, json.loads((out if code == 0 else rejected).read_text())

    run.out, run.rejected = out, rejected
    run.module, run.files, run.episode = module, files, episode
    return run


def test_complete_measurement_exits_zero_with_integer_counts(cli, capsys) -> None:
    code, data = cli()
    [entry] = data["entries"]
    assert code == 0 and entry["measurement"] == "complete" and entry["run_status"] == "passed"
    assert all(isinstance(v, int) for v in entry["counts"].values()) and entry["counts"]["killed"] >= 1
    assert entry["counts"]["killed"] + entry["counts"]["survived"] == len(entry["outcomes"])
    assert "NOT MEASURED" not in capsys.readouterr().out
    assert Path(data["python"]).is_absolute()


def test_a_genuine_baseline_error_is_not_measured_with_null_counts(cli, capsys) -> None:
    cli.files["tests/test_synthlib.py"] = "def test_broken(:\n"  # collection error: no usable baseline
    code, data = cli()
    [entry] = data["entries"]
    assert code == 1
    assert entry["measurement"] == "not_measured" and entry["run_status"] == "error" and entry["counts"] is None
    assert entry["outcomes"] == []
    captured = capsys.readouterr()
    assert "NOT MEASURED: error" in captured.out and "measurement incomplete or failed" in captured.err


def test_a_forced_baseline_error_is_not_measured(cli, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli.module, "run_mutants",
                        lambda *a, **k: MutationRun(TestRunStatus.ERROR, reason="forced baseline error"))
    code, data = cli()
    [entry] = data["entries"]
    assert code == 1 and entry["counts"] is None and entry["measurement"] == "not_measured"
    assert entry["run_reason"] == "forced baseline error"
    assert "NOT MEASURED: error forced baseline error" in capsys.readouterr().out


def test_partly_unmeasured_mutants_are_incomplete_and_fail_the_run(cli, monkeypatch, capsys) -> None:
    real = cli.module.generate_mutants
    bogus = MutantSpec(mutant_id="bogus", function_qualname=NORM, file_path="src/synthlib/core.py",
                       unified_diff="--- a/src/synthlib/core.py\n+++ b/src/synthlib/core.py\n@@ -1 +1 @@\n-nope\n+x\n")
    monkeypatch.setattr(cli.module, "generate_mutants", lambda *a, **k: [*real(*a, **k), bogus])
    code, data = cli()
    [entry] = data["entries"]
    assert code == 1 and entry["measurement"] == "incomplete"
    assert entry["counts"]["invalid"] == 1 and entry["counts"]["killed"] >= 1  # real counts kept, per outcome
    assert "INCOMPLETE: 1 mutant(s) not measured" in capsys.readouterr().out


def test_a_missing_function_is_recorded_not_dropped(cli, capsys) -> None:
    episode = cli.episode
    truth = episode.ground_truth.model_copy(update={"per_test_labels": [
        label(T_NORM, RepairDirection.UPDATE_TEST, function="nopackage.module.f", evidence=["ev-msg"])]})
    cli.module.load_episodes = lambda path: [episode.model_copy(update={"ground_truth": truth})]
    code, data = cli()
    [entry] = data["entries"]
    assert code == 1 and entry["measurement"] == "not_measured" and entry["counts"] is None
    assert "not found" in entry["run_reason"] and "NOT MEASURED" in capsys.readouterr().err


def test_no_generated_mutant_is_not_a_complete_zero(cli, monkeypatch) -> None:
    monkeypatch.setattr(cli.module, "generate_mutants", lambda *a, **k: [])
    code, data = cli()
    [entry] = data["entries"]
    assert code == 1 and entry["measurement"] == "not_measured" and entry["counts"] is None
    assert entry["run_reason"] == "no mutants generated"


def test_a_failed_build_leaves_an_existing_catalogue_untouched(cli, monkeypatch, tmp_path) -> None:
    cli.out.write_text('{"previous": "valid catalogue"}\n')
    monkeypatch.setattr(cli.module, "run_mutants",
                        lambda *a, **k: MutationRun(TestRunStatus.ERROR, reason="forced baseline error"))
    code, data = cli()
    assert code == 1 and data["entries"][0]["measurement"] == "not_measured"
    assert cli.out.read_text() == '{"previous": "valid catalogue"}\n'
    assert cli.rejected.is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["catalogue.json", "catalogue.json.incomplete"]  # no temps


def test_a_failed_build_does_not_create_the_catalogue(cli, monkeypatch) -> None:
    monkeypatch.setattr(cli.module, "generate_mutants", lambda *a, **k: [])
    code, _ = cli()
    assert code == 1 and not cli.out.exists()


def test_a_complete_build_replaces_the_catalogue_and_clears_the_rejected_one(cli, tmp_path) -> None:
    cli.out.write_text('{"previous": 1}\n')
    cli.rejected.write_text("stale")
    code, data = cli()
    assert code == 0 and json.loads(cli.out.read_text()) == data and "previous" not in data
    assert not cli.rejected.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["catalogue.json"]


def test_a_write_error_leaves_the_catalogue_and_no_temporary_file(cli, monkeypatch, tmp_path) -> None:
    cli.out.write_text("old\n")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cli.module.os, "replace", boom)
    with pytest.raises(OSError):
        cli()
    assert cli.out.read_text() == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["catalogue.json"]


def test_the_serialised_schema_is_the_documented_one(cli) -> None:
    code, data = cli()
    assert code == 0
    assert set(data) == {"generator", "kills_measured_against", "python", "isolation", "entries"}
    assert set(data["generator"]) == {"seed", "limit", "tool"}
    [entry] = data["entries"]
    assert set(entry) == {"episode_id", "function_qualname", "commit", "test_selector", "measurement", "run_status",
                          "run_reason", "baseline_failing", "counts", "seconds", "mutation_fixture", "outcomes"}
    assert set(entry["counts"]) == {"killed", "survived", "timeout", "error", "invalid"}
    assert set(entry["outcomes"][0]) == {"mutant_id", "outcome", "killed_by", "detail"}
    assert entry["measurement"] in {"complete", "incomplete", "not_measured"}
