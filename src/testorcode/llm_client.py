"""The single LLM interface.  No agent creates its own provider client.

The orchestrator builds one client with ``create_llm_client(settings)`` and hands
it to every agent, so all agents share one provider, model and budget.

Errors (see ``testorcode.errors``): ``LLMRateLimitError`` and
``LLMProviderOutageError`` are logged but never scored against the agent;
``LLMStructuredOutputError`` means the reply did not parse into ``response_model``;
any other provider failure is ``LLMError``.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from testorcode.config import Settings
from testorcode.schemas.result_schemas import AgentRole, TokenUsage


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


class OpenAICompatibleClient:
    """Planned implementation over the OpenAI SDK (any OpenAI-compatible endpoint)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def complete_structured[T: BaseModel](
        self,
        *,
        role: AgentRole,
        system: str,
        user: str,
        response_model: type[T],
    ) -> LLMResponse[T]:
        raise NotImplementedError("live provider calls are not implemented yet")


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the one shared client.

    Raises:
        ConfigError: ``settings.require_llm()`` fails.
    """
    settings.require_llm()
    return OpenAICompatibleClient(settings)
