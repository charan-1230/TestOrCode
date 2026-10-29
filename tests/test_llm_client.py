"""LLM client: structured output, retries and error mapping, all against a fake SDK (no network)."""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

from testorcode import llm_client
from testorcode.config import Settings
from testorcode.errors import (
    ConfigError,
    LLMError,
    LLMProviderOutageError,
    LLMRateLimitError,
    LLMStructuredOutputError,
)
from testorcode.llm_client import (
    FakeLLMClient,
    OpenAICompatibleClient,
    create_llm_client,
    extract_json_text,
    merge_usage,
)
from testorcode.schemas.contracts import Hypothesis, IntentAnalysis
from testorcode.schemas.episode_schemas import DecisionLabel, HypothesisKind
from testorcode.schemas.result_schemas import AgentRole, TokenUsage

KEY = "sk-test-secret-0123456789"
SETTINGS = Settings.from_env({"LLM_MODEL": "test-model-2026-01-01", "LLM_API_KEY": KEY})
ROLE = AgentRole.INTENT_ANALYST

VALID = {
    "group_id": "g1",
    "hypothesis": {"kind": "outdated_test", "explanation": "changelog says so"},
    "cited_evidence_ids": ["ev-1"],
    "confidence": 0.8,
    "recommended_decision": "UPDATE_TEST",
}


def reply(content: str | None, *, usage=(10, 5, 15), finish_reason: str = "stop", model="m-served"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        usage=None if usage is None else SimpleNamespace(
            prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=usage[2]
        ),
    )


class FakeSDK:
    """Mimics ``openai.OpenAI().chat.completions.create``: returns or raises scripted items."""

    def __init__(self, *items) -> None:
        self.items = list(items)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def status_error(cls, code: int, headers: dict | None = None, message: str = "boom"):
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    return cls(message, response=httpx.Response(code, request=request, headers=headers or {}), body=None)


def timeout_error():
    return openai.APITimeoutError(request=httpx.Request("POST", "https://llm.example/v1"))


def client(sdk: FakeSDK, **kw) -> tuple[OpenAICompatibleClient, list[float]]:
    sleeps: list[float] = []
    return OpenAICompatibleClient(SETTINGS, sdk_client=sdk, sleep=sleeps.append, **kw), sleeps


def ask(c: OpenAICompatibleClient, model=IntentAnalysis):
    return c.complete_structured(role=ROLE, system="sys", user="usr", response_model=model)


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #


class TestStructuredOutput:
    __test__ = True

    def test_valid_reply_is_parsed_with_usage_and_model(self) -> None:
        sdk = FakeSDK(reply(json.dumps(VALID)))
        c, _ = client(sdk)
        r = ask(c)
        assert isinstance(r.parsed, IntentAnalysis)
        assert r.parsed.recommended_decision is DecisionLabel.UPDATE_TEST
        assert r.model_id == "m-served"
        assert r.usage == TokenUsage(reported=True, input_tokens=10, output_tokens=5, total_tokens=15, llm_calls=1)
        req = sdk.requests[0]
        assert req["model"] == "test-model-2026-01-01"
        assert req["response_format"]["type"] == "json_schema"
        assert req["response_format"]["json_schema"]["name"] == "IntentAnalysis"
        assert KEY not in json.dumps(req, default=str)

    def test_fenced_reply_with_prose_is_accepted(self) -> None:
        text = "Here you go:\n```json\n" + json.dumps(VALID) + "\n```"
        c, _ = client(FakeSDK(reply(text)))
        assert ask(c).parsed.group_id == "g1"
        assert extract_json_text("noise " + json.dumps(VALID) + " trailing") == json.dumps(VALID)

    @pytest.mark.parametrize("mode, fmt", [("json_object", {"type": "json_object"}), ("prompt", None)])
    def test_modes_without_native_schema_put_schema_in_prompt(self, mode, fmt) -> None:
        sdk = FakeSDK(reply(json.dumps(VALID)))
        c, _ = client(sdk, structured_mode=mode)
        ask(c)
        req = sdk.requests[0]
        assert req.get("response_format") == fmt
        assert "JSON schema" in req["messages"][0]["content"]

    def test_malformed_json_is_re_asked_once_and_usage_summed(self) -> None:
        sdk = FakeSDK(reply("{not json"), reply(json.dumps(VALID)))
        c, _ = client(sdk)
        r = ask(c)
        assert r.parsed.group_id == "g1"
        assert r.usage.total_tokens == 30 and r.usage.llm_calls == 2
        follow_up = sdk.requests[1]["messages"]
        assert follow_up[-2] == {"role": "assistant", "content": "{not json"}
        assert "invalid" in follow_up[-1]["content"]

    def test_persistent_malformed_json_raises_without_echoing_reply(self) -> None:
        c, _ = client(FakeSDK(reply("SECRET-REPLY {"), reply("SECRET-REPLY {")))
        with pytest.raises(LLMStructuredOutputError) as info:
            ask(c)
        assert "SECRET-REPLY" not in str(info.value)

    @pytest.mark.parametrize(
        "payload, fragment",
        [
            ({k: v for k, v in VALID.items() if k != "confidence"}, "confidence"),
            ({**VALID, "recommended_decision": "MAYBE"}, "recommended_decision"),
            ({**VALID, "cited_evidence_ids": []}, "UPDATE_TEST requires"),
            (
                {**VALID, "hypothesis": {"kind": "regression", "explanation": "x"}},
                "cannot recommend UPDATE_TEST",
            ),
            ({**VALID, "confidence": 1.5}, "confidence"),
        ],
    )
    def test_invalid_outputs_are_rejected(self, payload, fragment) -> None:
        c, _ = client(FakeSDK(reply(json.dumps(payload))), repair_attempts=0)
        with pytest.raises(LLMStructuredOutputError, match=fragment):
            ask(c)

    def test_empty_and_truncated_replies(self) -> None:
        c, _ = client(FakeSDK(reply(None)), repair_attempts=0)
        with pytest.raises(LLMStructuredOutputError, match="empty"):
            ask(c)
        c, _ = client(FakeSDK(reply(json.dumps(VALID)[:20], finish_reason="length")))
        with pytest.raises(LLMStructuredOutputError, match="truncated"):
            ask(c)

    def test_missing_usage_is_unreported_not_zero(self) -> None:
        c, _ = client(FakeSDK(reply(json.dumps(VALID), usage=None)))
        usage = ask(c).usage
        assert usage.reported is False and usage.total_tokens is None and usage.llm_calls == 1


# --------------------------------------------------------------------------- #
# Retries and provider errors
# --------------------------------------------------------------------------- #


class TestRetries:
    __test__ = True

    def test_rate_limit_is_retried_honouring_retry_after(self) -> None:
        sdk = FakeSDK(status_error(openai.RateLimitError, 429, {"retry-after": "3"}), reply(json.dumps(VALID)))
        c, sleeps = client(sdk)
        assert ask(c).parsed.group_id == "g1"
        assert sleeps == [3.0] and len(sdk.requests) == 2

    def test_persistent_rate_limit_raises_and_is_not_counted_against_agent(self) -> None:
        sdk = FakeSDK(*[status_error(openai.RateLimitError, 429) for _ in range(3)])
        c, sleeps = client(sdk, max_retries=2)
        with pytest.raises(LLMRateLimitError) as info:
            ask(c)
        assert len(sdk.requests) == 3 and sleeps == [1.0, 2.0]
        assert info.value.to_error_record("e1").counted_against_agent is False

    @pytest.mark.parametrize(
        "error", [timeout_error, lambda: status_error(openai.InternalServerError, 503)]
    )
    def test_timeouts_and_5xx_become_outages_after_retries(self, error) -> None:
        sdk = FakeSDK(error(), error())
        c, sleeps = client(sdk, max_retries=1)
        with pytest.raises(LLMProviderOutageError):
            ask(c)
        assert len(sdk.requests) == 2 and len(sleeps) == 1

    def test_transient_timeout_then_success(self) -> None:
        c, sleeps = client(FakeSDK(timeout_error(), reply(json.dumps(VALID))))
        assert ask(c).parsed.group_id == "g1" and sleeps == [1.0]

    @pytest.mark.parametrize("cls, code", [(openai.AuthenticationError, 401), (openai.BadRequestError, 400)])
    def test_client_errors_fail_fast_without_leaking_details(self, cls, code) -> None:
        sdk = FakeSDK(status_error(cls, code, message=f"Incorrect API key provided: {KEY}"))
        c, sleeps = client(sdk)
        with pytest.raises(LLMError) as info:
            ask(c)
        assert type(info.value) is LLMError and sleeps == [] and len(sdk.requests) == 1
        assert KEY not in str(info.value) and str(code) in str(info.value)

    def test_logs_never_contain_the_key_or_prompt(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        log = logging.getLogger("testorcode")
        old_level = log.level
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            sdk = FakeSDK(status_error(openai.RateLimitError, 429), reply("{bad"), reply(json.dumps(VALID)))
            c, _ = client(sdk)
            c.complete_structured(role=ROLE, system="PROMPT-SYS", user="PROMPT-USR", response_model=IntentAnalysis)
        finally:
            log.removeHandler(handler)
            log.setLevel(old_level)
        text = stream.getvalue()
        assert text and KEY not in text and "PROMPT-" not in text


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


class TestConstruction:
    __test__ = True

    def test_missing_model_fails_before_any_request(self) -> None:
        sdk = FakeSDK()
        c = OpenAICompatibleClient(Settings.from_env({}), sdk_client=sdk)
        with pytest.raises(ConfigError, match="LLM_MODEL"):
            ask(c)
        assert sdk.requests == []

    def test_create_llm_client_builds_sdk_lazily_from_settings(self) -> None:
        settings = Settings.from_env(
            {"LLM_MODEL": "m-1", "LLM_API_KEY": KEY, "LLM_BASE_URL": "https://llm.example/v1"}
        )
        c = create_llm_client(settings)
        assert isinstance(c, OpenAICompatibleClient)
        sdk = c._client()  # no network: the SDK only stores configuration
        assert str(sdk.base_url).startswith("https://llm.example/v1")
        assert sdk.max_retries == 0
        assert KEY not in repr(c)

    def test_keyless_endpoint_is_allowed(self) -> None:
        settings = Settings.from_env({"LLM_MODEL": "m-1", "LLM_BASE_URL": "http://localhost:8000/v1"})
        assert create_llm_client(settings)._client().api_key == "not-needed"

    def test_merge_usage_keeps_unreported_distinct_from_zero(self) -> None:
        unreported = TokenUsage(llm_calls=1)
        reported = TokenUsage(reported=True, total_tokens=7, llm_calls=1)
        assert merge_usage(unreported, unreported) == TokenUsage(llm_calls=2)
        assert merge_usage(unreported, reported).total_tokens == 7
        assert merge_usage(reported, reported).total_tokens == 14


# --------------------------------------------------------------------------- #
# FakeLLMClient
# --------------------------------------------------------------------------- #


class TestFakeLLMClient:
    __test__ = True

    def test_replies_in_order_per_role_and_records_calls(self) -> None:
        h1 = Hypothesis(kind=HypothesisKind.UNKNOWN, explanation="first")
        fake = FakeLLMClient({ROLE: [h1, {"kind": "regression", "explanation": "second"}]})
        r1 = fake.complete_structured(role=ROLE, system="s", user="u", response_model=Hypothesis)
        r2 = fake.complete_structured(role=ROLE, system="s", user="u", response_model=Hypothesis)
        assert r1.parsed is h1 and r2.parsed.kind is HypothesisKind.REGRESSION
        assert r1.model_id == "fake-model-1" and r1.usage.llm_calls == 1
        assert [c["role"] for c in fake.calls] == [ROLE, ROLE] and fake.remaining(ROLE) == 0

    def test_string_replies_are_validated_like_live_replies(self) -> None:
        fake = FakeLLMClient({ROLE: ["```json\n" + json.dumps(VALID) + "\n```", "{oops"]})
        assert fake.complete_structured(role=ROLE, system="", user="", response_model=IntentAnalysis).parsed
        with pytest.raises(LLMStructuredOutputError):
            fake.complete_structured(role=ROLE, system="", user="", response_model=IntentAnalysis)

    def test_errors_wrong_types_and_exhaustion(self) -> None:
        fake = FakeLLMClient({ROLE: [LLMRateLimitError("429"), Hypothesis(kind="unknown", explanation="x")]})
        with pytest.raises(LLMRateLimitError):
            fake.complete_structured(role=ROLE, system="", user="", response_model=Hypothesis)
        with pytest.raises(LLMStructuredOutputError, match="not a IntentAnalysis"):
            fake.complete_structured(role=ROLE, system="", user="", response_model=IntentAnalysis)
        with pytest.raises(LLMError, match="no scripted reply"):
            fake.complete_structured(role=AgentRole.REPAIRER, system="", user="", response_model=Hypothesis)

    def test_invalid_decision_is_rejected(self) -> None:
        bad = {**VALID, "hypothesis": {"kind": "unknown", "explanation": "?"}}
        fake = FakeLLMClient({ROLE: [bad]})
        with pytest.raises(LLMStructuredOutputError, match="cannot recommend"):
            fake.complete_structured(role=ROLE, system="", user="", response_model=IntentAnalysis)


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #


class TestRegressions:
    __test__ = True

    def test_http_408_is_retried_then_succeeds(self) -> None:
        sdk = FakeSDK(status_error(openai.APIStatusError, 408), reply(json.dumps(VALID)))
        c, sleeps = client(sdk)
        assert ask(c).parsed.group_id == "g1"
        assert len(sdk.requests) == 2 and sleeps == [1.0]

    def test_http_408_exhausts_retry_limit_as_outage(self) -> None:
        sdk = FakeSDK(*[status_error(openai.APIStatusError, 408, message=KEY) for _ in range(2)])
        c, sleeps = client(sdk, max_retries=1)
        with pytest.raises(LLMProviderOutageError) as info:
            ask(c)
        assert len(sdk.requests) == 2 and len(sleeps) == 1
        assert "408" in str(info.value) and KEY not in str(info.value)

    def test_invalid_replies_keep_their_token_usage(self) -> None:
        sdk = FakeSDK(reply("{bad", usage=(10, 5, 15)), reply('{"group_id": "g1"}', usage=(20, 4, 24)))
        c, _ = client(sdk, repair_attempts=1)
        with pytest.raises(LLMStructuredOutputError) as info:
            ask(c)
        assert info.value.usage == TokenUsage(
            reported=True, input_tokens=30, output_tokens=9, total_tokens=39, llm_calls=2
        )

    def test_invalid_replies_without_usage_stay_unreported(self) -> None:
        c, _ = client(FakeSDK(reply("{bad", usage=None), reply("{bad", usage=None)))
        with pytest.raises(LLMStructuredOutputError) as info:
            ask(c)
        assert info.value.usage == TokenUsage(llm_calls=2)  # no invented zeros

    def test_truncated_reply_keeps_its_token_usage(self) -> None:
        c, _ = client(FakeSDK(reply('{"group_id"', finish_reason="length", usage=(10, 50, 60))))
        with pytest.raises(LLMStructuredOutputError, match="truncated") as info:
            ask(c)
        assert info.value.usage.total_tokens == 60


# --------------------------------------------------------------------------- #
# Several API keys, used one after another
# --------------------------------------------------------------------------- #

KEYS = ("sk-pool-key-one-0001", "sk-pool-key-two-0002", "sk-pool-key-three-0003")
POOL = Settings.from_env({"LLM_MODEL": "m-1", **{f"LLM_API_KEY_{i}": k for i, k in enumerate(KEYS, 1)}})


def daily_quota():  # the shape of Gemini's per-day quota 429 (the body is never logged)
    request = httpx.Request("POST", "https://llm.example/v1/chat/completions")
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                      "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    return openai.RateLimitError("quota", response=httpx.Response(429, request=request), body=body)


def pool_client(per_key: dict[str, list], **kw):
    used: list[str] = []
    sdks = {k: FakeSDK(*items) for k, items in per_key.items()}

    def factory(key: str):
        used.append(key)
        return sdks[key]
    sleeps: list[float] = []
    return OpenAICompatibleClient(POOL, sdk_factory=factory, sleep=sleeps.append, **kw), used, sleeps


@pytest.fixture(autouse=True)
def _fresh_quota_memo(monkeypatch):
    """Every test starts with no key known to be out of quota (the memo is process-wide)."""
    monkeypatch.setattr(llm_client, "_EXHAUSTED", {})


class TestKeyPool:
    __test__ = True

    def test_a_new_client_skips_keys_known_to_be_out_of_daily_quota(self) -> None:
        # The experiment runner builds one client per episode: the second must not re-spend a request on key 1.
        first, used, _ = pool_client({KEYS[0]: [daily_quota()], KEYS[1]: [reply(json.dumps(VALID))]})
        ask(first)
        assert used == [KEYS[0], KEYS[1]]
        second, used2, _ = pool_client({KEYS[1]: [reply(json.dumps(VALID))]})
        assert ask(second).parsed.group_id == "g1" and used2 == [KEYS[1]]

    def test_a_per_minute_limit_is_not_remembered(self) -> None:
        first, _, _ = pool_client({KEYS[0]: [status_error(openai.RateLimitError, 429), reply(json.dumps(VALID))]})
        ask(first)
        second, used, _ = pool_client({KEYS[0]: [reply(json.dumps(VALID))]})
        ask(second)
        assert used == [KEYS[0]]

    def test_a_remembered_quota_expires(self, monkeypatch) -> None:
        first, _, _ = pool_client({KEYS[0]: [daily_quota()], KEYS[1]: [reply(json.dumps(VALID))]})
        ask(first)
        monkeypatch.setattr(llm_client, "_EXHAUSTED_TTL_SECONDS", 0.0)  # the day's quota may have reset
        second, used, _ = pool_client({KEYS[0]: [reply(json.dumps(VALID))]})
        ask(second)
        assert used == [KEYS[0]]

    def test_every_key_remembered_exhausted_still_fails_as_an_api_failure(self) -> None:
        first, _, _ = pool_client({k: [daily_quota()] for k in KEYS})
        with pytest.raises(LLMRateLimitError):
            ask(first)
        second, used, _ = pool_client({KEYS[2]: [daily_quota()]})
        with pytest.raises(LLMRateLimitError):
            ask(second)
        assert used == [KEYS[2]]  # one confirming request on the last key, none on the known-exhausted ones
        assert not any(k in repr(llm_client._EXHAUSTED) for k in KEYS)  # hashes only, never the keys

    def test_keys_are_used_in_order_and_a_daily_quota_switches_at_once(self) -> None:
        c, used, sleeps = pool_client({KEYS[0]: [daily_quota()], KEYS[1]: [reply(json.dumps(VALID))]})
        assert ask(c).parsed.group_id == "g1"
        assert used == [KEYS[0], KEYS[1]] and sleeps == []  # no waiting on a used-up day quota
        assert ask_again_uses_same_key(c, used)

    def test_per_minute_limit_is_retried_on_the_same_key(self) -> None:
        c, used, sleeps = pool_client({KEYS[0]: [status_error(openai.RateLimitError, 429), reply(json.dumps(VALID))]})
        assert ask(c).parsed.group_id == "g1"
        assert used == [KEYS[0]] and sleeps == [1.0]

    def test_rate_limits_outlasting_the_retries_move_to_the_next_key(self) -> None:
        limited = [status_error(openai.RateLimitError, 429) for _ in range(2)]
        c, used, sleeps = pool_client({KEYS[0]: limited, KEYS[1]: [reply(json.dumps(VALID))]}, max_retries=1)
        assert ask(c).parsed.group_id == "g1"
        assert used == [KEYS[0], KEYS[1]] and sleeps == [1.0]

    def test_every_key_exhausted_is_an_api_failure_not_an_agent_error(self) -> None:
        c, used, _ = pool_client({k: [daily_quota()] for k in KEYS})
        with pytest.raises(LLMRateLimitError) as info:
            ask(c)
        assert used == list(KEYS)
        assert info.value.to_error_record("e1").counted_against_agent is False

    def test_switch_logs_key_positions_never_keys(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        log = logging.getLogger("testorcode")
        log.addHandler(handler)
        try:
            c, _, _ = pool_client({KEYS[0]: [daily_quota()], KEYS[1]: [reply(json.dumps(VALID))]})
            ask(c)
        finally:
            log.removeHandler(handler)
        text = stream.getvalue()
        assert "API key 1/3 is out of quota; switching to key 2/3" in text
        assert not any(k in text for k in KEYS) and "PerDay" not in text


def ask_again_uses_same_key(c: OpenAICompatibleClient, used: list[str]) -> bool:
    c._sdk.items.append(reply(json.dumps(VALID)))  # the switched-to key stays in use for later calls
    ask(c)
    return used == [KEYS[0], KEYS[1]]
