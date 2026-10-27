"""Trusted behaviour verification, kept separate from raw pytest results.

``PytestExecution.status == PASSED`` is a *raw* result: it comes from the test
process, which also runs the agent's code and edited tests, so it can be
weakened or forged.  It is never integrity approval on its own.

A ``TrustedProbe`` is supplied by the harness from outside the agent-editable
files (maintainer, dataset, or approved memory) and checks real behaviour of
one production function: each case gives arguments and the expected return
value or exception.  ``verify_behaviour`` runs it in a *fresh* sandbox:

* only the arguments go into the sandbox; the expected values stay on the
  host, so code under test cannot "agree" with them without computing them;
* a fixed driver (not pytest, no conftest, no repository config, ``python -I``)
  imports the function from the source roots only and prints the results;
* the host compares; anything missing, ambiguous, timed out or unparsable is
  ``ERROR``/``NOT_RUN`` and never counts as verified.

``ProbeStatus`` (existing contract) is the result vocabulary: ``PASSED`` means
every case matched; ``FAILED`` means a case did not; ``ERROR``/``NOT_RUN``
mean "not verified".  ``trusted_outcome`` combines a raw post-repair run with
a verification and maps it onto ``GuardDecision`` for the (not yet
implemented) Integrity Guard: only ``PASSED`` + ``PASSED`` may recommend
``APPROVE``, a failed probe recommends ``REJECT``, everything else ``ESCALATE``.

Limits: a probe only covers its own cases (choose discriminating inputs; a
constant expectation such as ``True`` is easy to satisfy); provenance is
declared by the caller and cannot be proven here; code under test can still
misbehave on inputs the probe does not try.  Reasons never contain expected
values.
"""

from __future__ import annotations

import json
import secrets
import sys
import tempfile
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from testorcode.errors import ToolError
from testorcode.logging_config import get_logger
from testorcode.schemas.contracts import PytestExecution
from testorcode.schemas.episode_schemas import EditPolicy
from testorcode.schemas.result_schemas import GuardDecision, ProbeStatus, TestRunStatus
from testorcode.tools.pytest_runner import _child_env, _execute
from testorcode.tools.sandbox import Isolation, layout_for, resolve_isolation, wrap_command

logger = get_logger(__name__)

_DOTTED: Final[str] = r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$"
_RESULT_PREFIX: Final[str] = "TESTORCODE-PROBE-RESULT"

# Runs inside the sandbox.  Reads the call specification (no expected values),
# calls the function, prints one result line.  ``python -I``: no PYTHONPATH,
# no user site, no current directory on sys.path.
_DRIVER: Final[str] = """\
import importlib, json, sys

token, spec_path = sys.argv[1], sys.argv[2]
with open(spec_path) as fh:
    spec = json.load(fh)
sys.path[:0] = spec["import_roots"]
target = importlib.import_module(spec["module"])
for part in spec["function"].split("."):
    target = getattr(target, part)
results = []
for case in spec["cases"]:
    try:
        value = target(*case["args"], **case["kwargs"])
    except Exception as exc:  # the case's outcome, not a driver failure
        results.append({"raised": type(exc).__name__})
        continue
    try:
        results.append({"value": json.loads(json.dumps(value))})
    except (TypeError, ValueError):
        results.append({"unserializable": type(value).__name__})
sys.stdout.write("\\n%s %s %s\\n" % ("TESTORCODE-PROBE-RESULT", token, json.dumps(results)))
sys.stdout.flush()
"""


class ProbeCase(BaseModel):
    """One call and its trusted expectation (``expected`` value or ``raises`` exception name)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    args: tuple[JsonValue, ...] = ()
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    expected: JsonValue = None
    raises: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class TrustedProbe(BaseModel):
    """Behaviour check supplied by the harness, never taken from agent-editable files.

    ``provenance`` must name a trusted origin; a probe written by the model for
    its own repair is not accepted (``model_generated`` is not a valid value).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    probe_id: str = Field(min_length=1, max_length=256)
    module: str = Field(pattern=_DOTTED)
    function: str = Field(pattern=_DOTTED)
    cases: tuple[ProbeCase, ...] = Field(min_length=1)
    provenance: Literal["maintainer", "dataset", "approved_memory"]


class TrustedVerification(BaseModel):
    """Outcome of trusted verification; ``verified`` is the only approval signal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ProbeStatus
    probe_id: str | None = None
    reasons: tuple[str, ...] = ()
    isolation: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TrustedVerification:
        if self.status is not ProbeStatus.PASSED and not self.reasons:
            raise ValueError("an unverified outcome must give reasons")
        return self

    @property
    def verified(self) -> bool:
        return self.status is ProbeStatus.PASSED

    @property
    def recommended_decision(self) -> GuardDecision:
        """For the Integrity Guard: APPROVE only when verified (necessary, not sufficient)."""
        if self.status is ProbeStatus.PASSED:
            return GuardDecision.APPROVE
        if self.status is ProbeStatus.FAILED:
            return GuardDecision.REJECT
        return GuardDecision.ESCALATE


NOT_VERIFIED_NO_PROBE: Final[TrustedVerification] = TrustedVerification(
    status=ProbeStatus.NOT_RUN,
    reasons=("no trusted probe supplied; a raw pytest PASS is not integrity approval",),
)


def verify_behaviour(
    worktree_path: Path,
    probe: TrustedProbe | None,
    *,
    timeout_seconds: int,
    policy: EditPolicy | None = None,
    python_executable: str | Path | None = None,
    isolation: Isolation | str = Isolation.AUTO,
    docker_image: str | None = None,
) -> TrustedVerification:
    """Run ``probe`` against the worktree's source code in a fresh sandbox.

    ``probe=None`` returns ``NOT_RUN`` (never verified).  Execution problems are
    reported as ``ERROR``, not raised, except an unavailable sandbox.

    Raises:
        ToolError: no sandbox is available (fail closed) or the worktree is missing.
    """
    if probe is None:
        return NOT_VERIFIED_NO_PROBE
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive integer")
    root = Path(worktree_path).resolve()
    if not root.is_dir():
        raise ToolError(f"worktree does not exist or is not a directory: {root}")
    policy = policy or EditPolicy()
    mode = resolve_isolation(isolation, docker_image)

    def unverified(status: ProbeStatus, reason: str) -> TrustedVerification:
        return TrustedVerification(
            status=status, probe_id=probe.probe_id, reasons=(reason,), isolation=mode.value
        )

    source_roots = [r.strip("/") for r in policy.allowed_source_roots]
    if not _module_in_source_roots(root, source_roots, probe.module):
        return unverified(
            ProbeStatus.ERROR, f"probe module {probe.module!r} is not in the source roots"
        )
    if mode is Isolation.DOCKER:
        python = str(python_executable or "python")
    else:
        python = str(python_executable or sys.executable)

    token = secrets.token_hex(16)
    with tempfile.TemporaryDirectory(prefix="testorcode-verify-") as tmp:
        tmp_dir = Path(tmp)
        outdir, probedir = tmp_dir / "out", tmp_dir / "probe"
        outdir.mkdir()
        probedir.mkdir()
        layout = layout_for(mode, root, outdir, probedir, python)
        spec = {
            "module": probe.module,
            "function": probe.function,
            "import_roots": [f"{layout.worktree}/{r}" for r in source_roots],
            # Arguments only: expected values never enter the sandbox.
            "cases": [{"args": list(c.args), "kwargs": c.kwargs} for c in probe.cases],
        }
        (probedir / "spec.json").write_text(json.dumps(spec))
        (probedir / "driver.py").write_text(_DRIVER)
        inner = [
            python,
            "-I",
            "-B",
            f"{layout.probedir}/driver.py",
            token,
            f"{layout.probedir}/spec.json",
        ]
        container = f"testorcode-verify-{token[:12]}" if mode is Isolation.DOCKER else None
        command, popen_env = wrap_command(
            mode,
            inner,
            worktree=root,
            outdir=outdir,
            probedir=probedir,
            python=python,
            env=_child_env(mode, tmp_dir, layout),
            docker_image=docker_image,
            container_name=container,
        )
        outcome = _execute(command, root, tmp_dir, timeout_seconds, popen_env, container)

    if outcome.timed_out:
        return unverified(ProbeStatus.ERROR, f"probe timed out after {timeout_seconds}s")
    lines = [
        line.split(" ", 2)[2]
        for line in outcome.stdout.splitlines()
        if line.startswith(f"{_RESULT_PREFIX} {token} ")
    ]
    if outcome.exit_code != 0 or len(lines) != 1:
        return unverified(
            ProbeStatus.ERROR,
            f"probe did not complete cleanly (exit code {outcome.exit_code}, "
            f"{len(lines)} result records)",
        )
    try:
        results = json.loads(lines[0])
    except ValueError:
        return unverified(ProbeStatus.ERROR, "probe result record is not valid JSON")
    if not isinstance(results, list) or len(results) != len(probe.cases):
        return unverified(ProbeStatus.ERROR, "probe result record has the wrong shape")

    mismatches = [
        f"case {i}: behaviour differs from the trusted expectation"
        for i, (case, result) in enumerate(zip(probe.cases, results, strict=True), start=1)
        if not _matches(case, result)
    ]
    logger.info(
        "verify_behaviour: probe %s, %d/%d cases matched",
        probe.probe_id,
        len(probe.cases) - len(mismatches),
        len(probe.cases),
    )
    if mismatches:
        return TrustedVerification(
            status=ProbeStatus.FAILED,
            probe_id=probe.probe_id,
            reasons=tuple(mismatches),
            isolation=mode.value,
        )
    return TrustedVerification(
        status=ProbeStatus.PASSED, probe_id=probe.probe_id, isolation=mode.value
    )


def trusted_outcome(
    post_repair_run: PytestExecution | None, verification: TrustedVerification | None
) -> TrustedVerification:
    """Combine a raw post-repair pytest run with trusted verification.

    A raw ``PASSED`` never approves on its own: without a verified probe the
    result is ``NOT_RUN`` (escalate).  A failing raw run cannot be approved by a
    probe either.
    """
    verification = verification or NOT_VERIFIED_NO_PROBE
    if not verification.verified:
        return verification
    if post_repair_run is None or post_repair_run.status is not TestRunStatus.PASSED:
        return TrustedVerification(
            status=ProbeStatus.FAILED,
            probe_id=verification.probe_id,
            reasons=("the post-repair test run did not pass",),
            isolation=verification.isolation,
        )
    return verification


def _matches(case: ProbeCase, result: object) -> bool:
    if not isinstance(result, dict) or len(result) != 1:
        return False
    if case.raises is not None:
        return result.get("raised") == case.raises
    return "value" in result and result["value"] == json.loads(json.dumps(case.expected))


def _module_in_source_roots(root: Path, source_roots: list[str], module: str) -> bool:
    """The module is a file or package of the repository's own source, not tests or stdlib."""
    rel = Path(*module.split("."))
    for source_root in source_roots:
        base = root / source_root
        for candidate in (base / rel.with_suffix(".py"), base / rel / "__init__.py"):
            if candidate.is_file() and not candidate.is_symlink():
                return candidate.resolve().is_relative_to(base.resolve())
    return False
