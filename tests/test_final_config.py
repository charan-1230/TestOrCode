"""``configs/final.json``: the frozen experimental decisions (and that the conditions implement them)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from testorcode.conditions import profile_for
from testorcode.config import Settings
from testorcode.freeze import SECRET_VARS, load_frozen
from testorcode.schemas.result_schemas import ExperimentCondition as C

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "configs" / "final.json"


@pytest.fixture(scope="module")
def frozen():
    return load_frozen(FINAL)[0]


def test_final_config_is_named_and_not_a_draft(frozen) -> None:
    assert frozen.name == "final-v1" and not frozen.name.upper().startswith("DRAFT")


def test_retrieval_is_hybrid_voyage_without_rerank_and_with_the_cache(frozen) -> None:
    s = frozen.apply(Settings.from_env({}))
    assert (s.retrieval_mode, s.embedding_backend, s.voyage_embedding_model, s.rerank_backend) == (
        "hybrid", "voyage", "voyage-code-3", "none")
    assert s.embedding_cache_dir == Path("results/embedding_cache") and s.retrieval_top_k == 8
    cache = ROOT / s.embedding_cache_dir
    if cache.is_dir():
        assert list(cache.glob("*.jsonl")), "the warmed embedding cache is missing"


def test_guard_patch_and_revision_policy(frozen) -> None:
    s = frozen.apply(Settings.from_env({}))
    assert s.guard_min_retention == 0.9 and s.guard_unmeasurable_strength == "escalate"
    assert s.mutant_cache_path is None  # generated from the CI-state source, never from a dev-library catalogue
    assert s.test_patch_rule == "expected_value" and s.max_repair_revisions == 1


def test_tool_faults_are_seeded_rate_ten_percent(frozen) -> None:
    assert frozen.experiment.tool_faults is True and frozen.experiment.tool_fault_rate == 0.1


def test_scoring_rules_and_dpi_thresholds(frozen) -> None:
    assert frozen.scoring.direction_from == "hypothesis" and frozen.scoring.escalate_after_correct_repair == "wrong"
    h = frozen.hypothesis
    assert (h.min_direction_f1_gain, h.max_masked_regression_full, h.min_masked_regression_b1) == (0.15, 0.05, 0.2)
    assert (h.min_retention_full, h.max_retention_b1, h.min_covered_tool_call_saving) == (0.9, 0.7, 0.25)
    assert (h.max_overgeneralisation_scoped, h.min_overgeneralisation_coarse) == (0.1, 0.3)


def test_model_and_secrets_stay_out_of_the_frozen_file(frozen) -> None:
    keys = set(json.loads(FINAL.read_text())["settings"])
    assert not keys & SECRET_VARS and not any(k.startswith(("LLM_", "VOYAGE_API")) for k in keys - {"VOYAGE_EMBEDDING_MODEL"})


def test_conditions_implement_the_ablations() -> None:
    full, b1 = profile_for(C.FULL), profile_for(C.B1)
    a1, a2, a3 = (profile_for(c) for c in (C.A1_NO_INTENT_RETRIEVAL, C.A2_NO_MEMORY, C.A3_NO_INTEGRITY_GUARD))
    assert full.intent_retrieval and full.memory and full.integrity_guard
    assert b1.standard_rag and not b1.memory and not b1.intent_retrieval  # same retriever (frozen settings), no memory
    assert not a1.intent_retrieval and not a1.standard_rag and a1.memory and a1.integrity_guard
    assert a2.intent_retrieval and not a2.memory and a2.integrity_guard
    assert a3.intent_retrieval and a3.memory and not a3.integrity_guard
