"""The pilot environment is pinned exactly, transitive dependencies included (F-10).

``scripts/pilot/requirements-pilot.txt`` must reproduce the environment the dataset was verified in: its canonical ``pip freeze`` text
hashes to the ``dependency_lock_hash`` recorded in ``datasets/pilot_manifest.json`` (read-only here), and the builder refuses to run in an
environment that differs from it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "scripts" / "pilot" / "requirements-pilot.txt"
VERIFIED = {  # versions of the verified pilot environment (the one recorded in datasets/pilot_manifest.json)
    "iniconfig": "2.3.1",
    "packaging": "26.3",
    "pluggy": "1.7.0",
    "pygments": "2.21.0",
    "pytest": "8.4.2",
}


def pin_lines() -> list[str]:
    return [ln.strip() for ln in PINS.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


@pytest.fixture(scope="module")
def build():
    scripts = str(ROOT / "scripts" / "pilot")
    sys.path.insert(0, scripts)
    try:
        import build_dataset

        yield build_dataset
    finally:
        sys.path.remove(scripts)


def test_every_pin_is_exact_and_the_four_transitive_packages_are_pinned() -> None:
    lines = pin_lines()
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==\d+(\.\d+)*", ln) for ln in lines), lines  # no ranges, no extras
    pinned = {ln.split("==")[0].lower(): ln.split("==")[1] for ln in lines}
    assert pinned == VERIFIED


def test_pytest_pin_matches_the_builder_manifest() -> None:
    manifest = json.loads((ROOT / "scripts" / "pilot" / "manifest.json").read_text())
    assert dict(line.split("==") for line in pin_lines())["pytest"] == manifest["pin"]["pytest"]


def test_pins_reproduce_the_recorded_dependency_hash() -> None:
    """The canonical `pip freeze` text of the pinned set hashes to the value recorded when the dataset was verified."""
    freeze = "".join(f"{ln}\n" for ln in sorted(pin_lines(), key=str.lower))
    recorded = json.loads((ROOT / "datasets" / "pilot_manifest.json").read_text())["dependency_lock_hash"]
    assert hashlib.sha256(freeze.encode()).hexdigest() == recorded


FREEZE = "".join(f"{ln}\n" for ln in sorted(pin_lines(), key=str.lower))


def test_builder_accepts_the_pinned_environment_exactly(build) -> None:
    build.check_pilot_environment(FREEZE, PINS.read_text())


@pytest.mark.parametrize(
    ("freeze", "fragment"),
    [
        (FREEZE.replace("pluggy==1.7.0", "pluggy==1.8.0"), "pluggy: installed 1.8.0, pinned 1.7.0"),
        (FREEZE.replace("packaging==26.3", "packaging==26.4"), "packaging: installed 26.4, pinned 26.3"),
        (FREEZE.replace("iniconfig==2.3.1\n", ""), "iniconfig: pinned 2.3.1 but not installed"),
        (FREEZE + "pytest-xdist==3.6.1\n", "pytest-xdist==3.6.1: installed but not pinned"),
        (FREEZE + "-e git+https://example.invalid/x.git#egg=x\n", "unexpected freeze line"),
        (FREEZE.replace("Pygments==2.21.0", "pygments==2.20.0"), "pygments: installed 2.20.0, pinned 2.21.0"),
    ],
)
def test_builder_refuses_a_drifted_environment(build, freeze: str, fragment: str) -> None:
    with pytest.raises(build.BuildError, match="requirements-pilot.txt") as exc:
        build.check_pilot_environment(freeze, PINS.read_text())
    assert fragment in str(exc.value)


def test_normalised_names_compare_equal(build) -> None:
    """pip may print a different case or separator than the pins file; the comparison normalises both."""
    build.check_pilot_environment(FREEZE.replace("Pygments", "PYGMENTS"), PINS.read_text())
