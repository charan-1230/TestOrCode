"""Provider check: parse and correct-action rates per structured-output mode, recommended mode, returned model id."""

from __future__ import annotations

import json

from testorcode.config import Settings
from testorcode.llm_client import FakeLLMClient
from testorcode.provider_check import INTENT_ID, PROBES, run_check
from testorcode.schemas.result_schemas import AgentRole

NO_TOOL = {"query": "", "path": "", "start_line": 1, "end_line": 0, "selectors": []}
READ = {"action": "read_file", "decisions": [], "rationale": "look", **NO_TOOL, "path": "units/convert.py", "end_line": 40}
FINISH = {"action": "finish", "rationale": "done", **NO_TOOL, "decisions": [{
    "test_ids": ["tests/test_units.py::test_to_meters"], "decision": "UPDATE_TEST", "hypothesis": "outdated_test",
    "explanation": "intended", "cited_evidence_ids": [INTENT_ID], "confidence": 0.9,
    "edits": [{"path": "tests/test_units.py", "old": "== 0.3", "new": "== 0.3048"}]}]}
INTENT = {"group_id": "g1", "hypothesis": {"kind": "outdated_test", "explanation": "intended"},
          "expected_behaviour": "four decimals", "cited_evidence_ids": [INTENT_ID], "contradicting_evidence_ids": [],
          "confidence": 0.9, "recommended_decision": "UPDATE_TEST"}
GUARD = {"verdict": "consistent", "test_meaning_preserved": True, "reasons": []}


def good_client(model_id: str = "vendor/model-2026-10-01") -> FakeLLMClient:
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [READ, FINISH], AgentRole.INTENT_ANALYST: [INTENT],
                         AgentRole.INTEGRITY_GUARD: [GUARD]}, model_id=model_id)
    return llm


def test_the_first_fully_parsing_mode_is_recommended_and_the_returned_id_recorded() -> None:
    settings = Settings.from_env({"LLM_MODEL": "vendor/model", "LLM_PROVIDER": "nim",
                                  "LLM_BASE_URL": "https://integrate.api.nvidia.com/v1?x=1"})
    seen_modes = []

    def client_for(s: Settings) -> FakeLLMClient:
        seen_modes.append(s.llm_structured_mode)
        if s.llm_structured_mode == "json_schema":  # this provider rejects the schema format: nothing parses
            return FakeLLMClient({role: ["not json"] * 2 for role in AgentRole})
        return good_client()

    result = run_check(settings, client_for, repeats=1)
    assert seen_modes == ["json_schema", "json_object", "prompt"]
    assert result["modes"]["json_schema"]["parse_rate"] == 0.0
    assert result["modes"]["json_object"]["parse_rate"] == 1.0 and result["modes"]["json_object"]["correct_rate"] == 1.0
    assert result["recommended_structured_mode"] == "json_object"
    assert result["model_ids_returned"] == ["vendor/model-2026-10-01"]
    assert result["endpoint_host"] == "integrate.api.nvidia.com"  # no path, query or key
    json.dumps(result)


def test_a_wrong_but_valid_reply_parses_and_is_not_correct() -> None:
    wrong_tool = {**READ, "path": "units/other.py"}
    llm = FakeLLMClient({AgentRole.SINGLE_AGENT: [wrong_tool, FINISH], AgentRole.INTENT_ANALYST: [INTENT],
                         AgentRole.INTEGRITY_GUARD: [GUARD]})
    result = run_check(Settings(llm_model="m"), lambda s: llm, modes=["prompt"], repeats=1)
    rows = result["modes"]["prompt"]["probes"]
    assert rows["tool_step"] == {**rows["tool_step"], "parsed": 1, "correct": 0}
    assert result["modes"]["prompt"]["correct_rate"] == 0.75


def test_probes_are_synthetic() -> None:
    from testorcode.dataset import episode_path, load_episodes

    texts = [e.agent_input.ci_event.commit_message or "" for split in ("dev", "test")
             for e in load_episodes(episode_path(split))]
    for probe in PROBES:
        assert not any(t and t[:30] in probe.user for t in texts)
