"""The single LLM interface.  No agent creates its own provider client.

The orchestrator builds one client with ``create_llm_client(settings)`` and hands
it to every agent, so all agents share one provider, model and budget.

Errors (see ``testorcode.errors``): ``LLMRateLimitError`` and
``LLMProviderOutageError`` are logged but never scored against the agent;
``LLMStructuredOutputError`` means the reply did not parse into ``response_model``;
any other provider failure is ``LLMError``.

Structured output is always validated client-side with ``response_model``.
``structured_mode`` only decides what the provider is asked for:

* ``"json_schema"``: ``response_format`` carries the model's JSON schema (non-strict).
* ``"json_object"``: JSON mode; the schema goes into the system prompt.
* ``"prompt"``: no ``response_format`` at all, for endpoints without JSON mode.

Never log prompts, replies or keys: only role, model, attempt numbers and token counts.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from testorcode.config import Settings
from testorcode.errors import (
    LLMError,
    LLMProviderOutageError,
    LLMRateLimitError,
    LLMStructuredOutputError,
)
from testorcode.logging_config import get_logger
from testorcode.schemas.result_schemas import AgentRole, TokenUsage

logger = get_logger(__name__)

StructuredMode = Literal["json_schema", "json_object", "prompt"]

# Sent as the key to keyless OpenAI-compatible endpoints (the SDK insists on one).
_KEYLESS_PLACEHOLDER = "not-needed"
_MAX_BACKOFF_SECONDS = 30.0
_RETRYABLE_4XX = frozenset({408})  # Request Timeout; 429 has its own branch


class LLMResponse[T: BaseModel](BaseModel):
    """Parsed model output plus the usage the provider reported."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parsed: T
    model_id: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class LLMClient(Protocol):
    """What agents depend on.  Tests inject a fake implementing this protocol."""

    def complete_structured[T: BaseModel](
        self,
        *,
        role: AgentRole,
        system: str,
        user: str,
        response_model: type[T],
    ) -> LLMResponse[T]:
        """Return a reply validated against ``response_model``.

        Raises:
            LLMRateLimitError, LLMProviderOutageError, LLMError: provider failure.
            LLMStructuredOutputError: the reply does not validate.
        """
        ...


# --------------------------------------------------------------------------- #
# Structured-output helpers (shared by the real and the fake client)
# --------------------------------------------------------------------------- #


def extract_json_text(text: str) -> str:
    """Strip Markdown fences and surrounding prose from a reply that should be one JSON object."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            stripped = stripped[start : end + 1]
    return stripped


def describe_validation_error(exc: ValidationError, limit: int = 5) -> str:
    """Short, input-free summary (the input may echo prompt content)."""
    parts = []
    for err in exc.errors(include_input=False, include_url=False)[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    more = exc.error_count() - limit
    if more > 0:
        parts.append(f"... and {more} more")
    return "; ".join(parts)


def parse_structured[T: BaseModel](text: str | None, response_model: type[T]) -> T:
    """Validate a raw reply against ``response_model``.

    Raises:
        LLMStructuredOutputError: empty reply, malformed JSON, missing fields or a
            failed model validator (e.g. UPDATE_TEST without evidence).
    """
    if text is None or not text.strip():
        raise LLMStructuredOutputError(f"empty reply for {response_model.__name__}")
    try:
        return response_model.model_validate_json(extract_json_text(text))
    except ValidationError as exc:
        raise LLMStructuredOutputError(
            f"reply does not validate as {response_model.__name__}: {describe_validation_error(exc)}"
        ) from None


def schema_instruction(response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    return (
        "Reply with exactly one JSON object and nothing else. "
        f"It must conform to this JSON schema:\n{schema}"
    )


def merge_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    """Sum two usages; unreported counts stay unreported rather than becoming zero."""
    if not a.reported:
        return b.model_copy() if b.reported else TokenUsage(llm_calls=_add(a.llm_calls, b.llm_calls))
    if not b.reported:
        return a.model_copy(update={"llm_calls": _add(a.llm_calls, b.llm_calls)})
    return TokenUsage(
        reported=True,
        input_tokens=_add(a.input_tokens, b.input_tokens),
        output_tokens=_add(a.output_tokens, b.output_tokens),
        total_tokens=_add(a.total_tokens, b.total_tokens),
        llm_calls=_add(a.llm_calls, b.llm_calls),
    )


def _add(x: int | None, y: int | None) -> int | None:
    if x is None and y is None:
        return None
    return (x or 0) + (y or 0)


# --------------------------------------------------------------------------- #
# OpenAI-compatible client
# --------------------------------------------------------------------------- #


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-compatible endpoint.

    Retries rate limits, timeouts, connection errors, 408 and 5xx responses with
    exponential backoff (honouring ``Retry-After``); other 4xx responses fail at
    once.  An unparsable reply is re-asked ``repair_attempts`` times with the
    validation errors, then raised as ``LLMStructuredOutputError`` whose ``usage``
    attribute holds the tokens all those replies consumed.

    ``sdk_client`` and ``sleep`` exist for tests: pass an object with
    ``chat.completions.create`` to avoid any network access.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        repair_attempts: int = 1,
        structured_mode: StructuredMode = "json_schema",
        temperature: float | None = 0.0,
        sdk_client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0 or repair_attempts < 0:
            raise ValueError("max_retries and repair_attempts must be >= 0")
        self._settings = settings
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._repair_attempts = repair_attempts
        self._mode = structured_mode
        self._temperature = temperature
        self._sdk = sdk_client
        self._sleep = sleep

    def complete_structured[T: BaseModel](
        self,
        *,
        role: AgentRole,
        system: str,
        user: str,
        response_model: type[T],
    ) -> LLMResponse[T]:
        self._settings.require_llm()  # ConfigError before any network access
        model = self._settings.llm_model
        if self._mode != "json_schema":
            system = f"{system}\n\n{schema_instruction(response_model)}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        usage = TokenUsage()
        model_id = model
        for repair in range(self._repair_attempts + 1):
            response = self._create_with_retries(role, messages, response_model)
            usage = merge_usage(usage, _usage_of(response))
            model_id = getattr(response, "model", None) or model_id
            try:
                text = _reply_text(response)
            except LLMStructuredOutputError as exc:
                exc.usage = usage  # the truncated reply still cost tokens
                raise
            try:
                parsed = parse_structured(text, response_model)
            except LLMStructuredOutputError as exc:
                logger.warning(
                    "role=%s structured output invalid (attempt %d/%d)",
                    role, repair + 1, self._repair_attempts + 1,
                )
                if repair == self._repair_attempts:
                    exc.usage = usage  # every reply so far still cost tokens
                    raise
                messages = [
                    *messages,
                    {"role": "assistant", "content": text or ""},
                    {
                        "role": "user",
                        "content": f"Your reply was invalid: {exc}. "
                        "Reply again with only the corrected JSON object.",
                    },
                ]
                continue
            logger.info(
                "role=%s model=%s tokens=%s", role, model_id,
                usage.total_tokens if usage.reported else "unreported",
            )
            return LLMResponse[response_model](parsed=parsed, model_id=model_id, usage=usage)
        raise AssertionError("unreachable")  # pragma: no cover

    # ----------------------------------------------------------------- #

    def _client(self) -> Any:
        if self._sdk is None:
            from openai import OpenAI

            key = self._settings.llm_api_key
            self._sdk = OpenAI(
                api_key=key.get_secret_value() if key is not None else _KEYLESS_PLACEHOLDER,
                base_url=self._settings.llm_base_url or None,
                timeout=self._timeout,
                max_retries=0,  # retries are ours, so they are logged and testable
            )
        return self._sdk

    def _request(self, messages: list[dict[str, str]], response_model: type[BaseModel]) -> dict:
        kwargs: dict[str, Any] = {"model": self._settings.llm_model, "messages": messages}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                    "strict": False,
                },
            }
        elif self._mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _create_with_retries(
        self, role: AgentRole, messages: list[dict[str, str]], response_model: type[BaseModel]
    ) -> Any:
        import openai

        kwargs = self._request(messages, response_model)
        for attempt in range(self._max_retries + 1):
            try:
                return self._client().chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                error: LLMError = LLMRateLimitError(_describe(exc))
                retry_after = _retry_after(exc)
            except (openai.APITimeoutError, openai.APIConnectionError) as exc:
                error = LLMProviderOutageError(_describe(exc))
                retry_after = None
            except openai.APIStatusError as exc:
                if exc.status_code < 500 and exc.status_code not in _RETRYABLE_4XX:
                    raise LLMError(_describe(exc)) from None
                error = LLMProviderOutageError(_describe(exc))
                retry_after = _retry_after(exc)
            except openai.OpenAIError as exc:
                raise LLMError(_describe(exc)) from None
            if attempt == self._max_retries:
                logger.error("role=%s %s; giving up after %d attempts", role, error, attempt + 1)
                raise error from None
            delay = retry_after if retry_after is not None else min(2.0**attempt, _MAX_BACKOFF_SECONDS)
            logger.warning(
                "role=%s %s; retry %d/%d in %.1fs", role, error, attempt + 1, self._max_retries, delay
            )
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover


def _describe(exc: Exception) -> str:
    """Error class and status only: provider messages may echo the prompt or key fragments."""
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status is not None else "")


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return min(max(value, 0.0), _MAX_BACKOFF_SECONDS)


def _reply_text(response: Any) -> str | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    choice = choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        raise LLMStructuredOutputError("reply was truncated (finish_reason=length)")
    return getattr(choice.message, "content", None)


def _usage_of(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    counts = {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    if all(v is None for v in counts.values()):
        return TokenUsage(llm_calls=1)
    return TokenUsage(reported=True, llm_calls=1, **counts)


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the one shared client.

    Raises:
        ConfigError: ``settings.require_llm()`` fails.
    """
    settings.require_llm()
    return OpenAICompatibleClient(settings)


# --------------------------------------------------------------------------- #
# Deterministic fake
# --------------------------------------------------------------------------- #

FakeReply = BaseModel | Mapping[str, Any] | str | Exception


class FakeLLMClient:
    """Scripted ``LLMClient`` for tests: no network, fully deterministic.

    ``replies`` maps a role to the replies it returns, in order.  A reply may be
    a model instance (returned as is if it has the requested type), a dict or
    JSON string (validated exactly like a live reply), or an exception (raised).
    Running out of replies raises ``LLMError``.  ``calls`` records every request.
    """

    def __init__(
        self,
        replies: Mapping[AgentRole, Iterable[FakeReply]] | None = None,
        *,
        model_id: str = "fake-model-1",
        usage: TokenUsage | None = None,
    ) -> None:
        self._replies: dict[AgentRole, deque[FakeReply]] = defaultdict(deque)
        for role, items in (replies or {}).items():
            self._replies[role].extend(items)
        self.model_id = model_id
        self._usage = usage or TokenUsage(llm_calls=1)
        self.calls: list[dict[str, Any]] = []

    def add(self, role: AgentRole, *replies: FakeReply) -> None:
        self._replies[role].extend(replies)

    def remaining(self, role: AgentRole) -> int:
        return len(self._replies[role])

    def complete_structured[T: BaseModel](
        self,
        *,
        role: AgentRole,
        system: str,
        user: str,
        response_model: type[T],
    ) -> LLMResponse[T]:
        self.calls.append(
            {"role": role, "system": system, "user": user, "response_model": response_model}
        )
        if not self._replies[role]:
            raise LLMError(f"FakeLLMClient has no scripted reply for role {role}")
        reply = self._replies[role].popleft()
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, response_model):
            parsed = reply
        elif isinstance(reply, BaseModel):
            raise LLMStructuredOutputError(
                f"scripted {type(reply).__name__} is not a {response_model.__name__}"
            )
        else:
            text = reply if isinstance(reply, str) else json.dumps(reply)
            parsed = parse_structured(text, response_model)
        return LLMResponse[response_model](
            parsed=parsed, model_id=self.model_id, usage=self._usage.model_copy()
        )
