"""The final-report builder: numbers from saved runs only, missing inputs never shown as 0, prose marked TO WRITE."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("build_final_report", ROOT / "docs/report/build_final_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_from_pilot_runs_marks_missing_conditions_and_inputs(tmp_path: Path) -> None:
    out = tmp_path / "final"
    assert _builder().main(["--out", str(out), "--no-pdf", "--test", str(tmp_path / "no-such-dir")]) == 0
    text = (out / "report.md").read_text()
    assert "| FULL | 9 | 0 | 1.000 |" in text  # the pilot dev table, from runs.jsonl
    assert "| B2 | not yet evaluated |" in text and "**test split:** not yet evaluated." in text
    assert "not computed (run `scripts/compare_conditions.py" in text  # no stats.json next to the pilot runs
    assert "[TO WRITE:" in text and (out / "report.tex").read_text().startswith("\\documentclass")
