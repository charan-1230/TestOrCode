from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from testorcode.logging_config import (
    PACKAGE_LOGGER_NAME,
    RedactingFormatter,
    configure_logging,
    episode_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _restore_logger():
    pkg = logging.getLogger(PACKAGE_LOGGER_NAME)
    saved = (list(pkg.handlers), pkg.level, pkg.propagate)
    yield
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
        h.close()
    for h in saved[0]:
        pkg.addHandler(h)
    pkg.setLevel(saved[1])
    pkg.propagate = saved[2]


def _ours() -> list[logging.Handler]:
    return [
        h
        for h in logging.getLogger(PACKAGE_LOGGER_NAME).handlers
        if getattr(h, "_testorcode_handler", False)
    ]


def test_repeated_configuration_does_not_duplicate_handlers() -> None:
    for _ in range(3):
        configure_logging("INFO", stream=io.StringIO())
    assert len(_ours()) == 1


def test_root_logger_is_untouched() -> None:
    before = list(logging.getLogger().handlers)
    configure_logging("DEBUG", stream=io.StringIO())
    assert logging.getLogger().handlers == before


def test_level_is_applied_and_changeable() -> None:
    buf = io.StringIO()
    configure_logging("WARNING", stream=buf)
    log = get_logger("orchestrator")
    log.info("hidden")
    log.warning("shown")
    configure_logging("DEBUG", stream=buf)
    log.debug("now visible")
    out = buf.getvalue()
    assert "hidden" not in out and "shown" in out and "now visible" in out


def test_invalid_level_rejected() -> None:
    with pytest.raises(ValueError, match="invalid log level"):
        configure_logging("LOUD")


def test_get_logger_namespacing() -> None:
    assert get_logger("agents.triage").name == "testorcode.agents.triage"
    assert get_logger("testorcode.tools.x").name == "testorcode.tools.x"


def test_episode_id_and_component_in_output() -> None:
    buf = io.StringIO()
    configure_logging("INFO", stream=buf)
    log = get_logger("agents.triage")
    log.info("outside")
    with episode_logging("ep-42"):
        log.info("inside")
    lines = buf.getvalue().splitlines()
    assert "[ep=-]" in lines[0] and "[ep=ep-42]" in lines[1]
    assert "testorcode.agents.triage" in lines[1]


def test_secrets_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_SERVICE_TOKEN", "tok-abcdefgh12345")
    buf = io.StringIO()
    configure_logging("INFO", stream=buf, secrets=["literal-secret-value"])
    log = get_logger("llm")
    log.info("env tok-abcdefgh12345 literal literal-secret-value")
    log.info("headers Authorization: Bearer abc123def456ghi")
    log.info("config api_key=hunter2hunter2 and sk-abcdefghijklmnopqrstuvwx")
    out = buf.getvalue()
    for leaked in ("tok-abcdefgh12345", "literal-secret-value", "abc123def456ghi", "hunter2", "abcdefghijklmnopqrstuvwx"):
        assert leaked not in out
    assert "[REDACTED]" in out


def test_long_messages_are_truncated() -> None:
    buf = io.StringIO()
    configure_logging("INFO", stream=buf)
    get_logger("x").info("A" * 10_000)
    assert len(buf.getvalue()) < 2500 and "truncated" in buf.getvalue()


def test_exception_text_is_redacted() -> None:
    fmt = RedactingFormatter(secrets=["topsecret-12345"])
    try:
        raise RuntimeError("boom topsecret-12345")
    except RuntimeError:
        import sys

        rec = logging.LogRecord("testorcode.x", logging.ERROR, "f", 1, "failed", None, sys.exc_info())
    rec.episode_id = "-"
    assert "topsecret" not in fmt.format(rec)


def test_no_file_created_unless_requested(tmp_path: Path) -> None:
    configure_logging("INFO", stream=io.StringIO())
    assert list(tmp_path.iterdir()) == []
    log_file = tmp_path / "logs" / "run.log"
    configure_logging("INFO", stream=io.StringIO(), log_file=log_file)
    assert not log_file.exists()  # opened lazily
    get_logger("x").info("hello")
    for h in _ours():
        h.flush()
    assert "hello" in log_file.read_text()
