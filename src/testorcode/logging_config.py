"""Logging setup for TestOrCode (standard library only).

Usage::

    # entry point (script, orchestrator CLI, test fixture) - once:
    configure_logging("INFO")

    # any module:
    logger = get_logger(__name__)            # "testorcode.agents.triage"
    logger.info("grouped %d failures", n)

    # orchestrator, around one episode:
    with episode_logging("ep-0042"):
        ...                                   # every record carries ep=ep-0042

Design:

* Only the ``testorcode`` package logger is configured; the root logger is never
  touched and nothing happens at import time.
* ``configure_logging`` is idempotent: it replaces the handlers it installed
  earlier and leaves foreign handlers (pytest ``caplog``) alone.
* Output goes to stderr by default; a file is opened only if ``log_file`` is given.
* Secrets are redacted from every formatted record, and over-long messages are
  truncated.  Do not log prompts, source files, diffs or patches: log sizes,
  ids and counts instead.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import IO, Final

PACKAGE_LOGGER_NAME: Final[str] = "testorcode"
MAX_MESSAGE_CHARS: Final[int] = 2000
REDACTED: Final[str] = "[REDACTED]"
_HANDLER_MARK: Final[str] = "_testorcode_handler"
_FORMAT: Final[str] = "%(asctime)s %(levelname)-7s %(name)s [ep=%(episode_id)s] %(message)s"

_episode_id: contextvars.ContextVar[str] = contextvars.ContextVar("testorcode_episode_id", default="-")

_SECRET_ENV_HINTS: Final[tuple[str, ...]] = ("KEY", "TOKEN", "SECRET", "PASSWORD")
_MIN_SECRET_LENGTH: Final[int] = 8
_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(bearer\s+|basic\s+)?[^\s,;'\"]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[:=]\s*)['\"]?[^\s,;'\"&]+"),
    re.compile(r"\b(?:sk|nvapi|ghp|gho|pa)[-_][A-Za-z0-9_-]{16,}"),
)


class EpisodeFilter(logging.Filter):
    """Adds ``record.episode_id`` from the current episode context ('-' if none)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "episode_id"):
            record.episode_id = _episode_id.get()
        return True


class RedactingFormatter(logging.Formatter):
    """Formats a record, then masks secrets and truncates the result."""

    def __init__(self, secrets: Iterable[str] = (), max_chars: int = MAX_MESSAGE_CHARS) -> None:
        super().__init__(_FORMAT)
        self._secrets = sorted({s for s in secrets if len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True)
        self._max_chars = max_chars

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTED)
        for pattern in _PATTERNS:
            if pattern.groups:
                text = pattern.sub(lambda m: m.group(1) + REDACTED, text)
            else:
                text = pattern.sub(REDACTED, text)
        return text

    def format(self, record: logging.LogRecord) -> str:
        text = self.redact(super().format(record))
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + f"... [truncated {len(text) - self._max_chars} chars]"
        return text


def _environment_secrets() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if any(hint in name.upper() for hint in _SECRET_ENV_HINTS) and len(value) >= _MIN_SECRET_LENGTH
    ]


def configure_logging(
    level: int | str = "INFO",
    *,
    stream: IO[str] | None = None,
    log_file: Path | str | None = None,
    secrets: Iterable[str] = (),
) -> logging.Logger:
    """Configure the ``testorcode`` logger; safe to call repeatedly.

    Args:
        level: Level name or number.
        stream: Console stream (default ``sys.stderr``).
        log_file: If given, also append to this file (parent directories are
            created now, the file itself on first write).
        secrets: Extra literal values to redact (e.g. ``Settings.secret_values()``).
            Values of environment variables whose names contain KEY, TOKEN,
            SECRET or PASSWORD are always redacted.

    Returns:
        The package logger.

    Raises:
        ValueError: ``level`` is not a valid level.
    """
    resolved = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(resolved, int):
        msg = f"invalid log level: {level!r}"
        raise ValueError(msg)  # noqa: TRY004 - a bad level name is a bad value

    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in list(package_logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            package_logger.removeHandler(handler)
            handler.close()

    formatter = RedactingFormatter([*_environment_secrets(), *secrets])
    handlers: list[logging.Handler] = [logging.StreamHandler(stream)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8", delay=True))
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(EpisodeFilter())
        setattr(handler, _HANDLER_MARK, True)
        package_logger.addHandler(handler)

    package_logger.setLevel(resolved)
    package_logger.propagate = False
    return package_logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``testorcode`` namespace.

    Pass ``__name__`` (already namespaced) or a short component name such as
    ``"orchestrator"``.  Never creates handlers.
    """
    if name == PACKAGE_LOGGER_NAME or name.startswith(PACKAGE_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


@contextlib.contextmanager
def episode_logging(episode_id: str) -> Iterator[None]:
    """Tag every record logged inside the block with ``ep=<episode_id>``."""
    token = _episode_id.set(episode_id)
    try:
        yield
    finally:
        _episode_id.reset(token)
