from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.evaluate_fever_lora_policy import (  # noqa: E402
    score_case,
    select_prediction,
)
from scripts.benchmarks.compose_hotpot_batch1 import compose_case  # noqa: E402
from scripts.benchmarks.compose_specialist_full import compose_rows  # noqa: E402
from scripts.benchmarks.train_evaluate_fever_evidence_selector import (  # noqa: E402
    local_label,
)
from scripts.benchmarks.train_evaluate_nq_action_selector import (  # noqa: E402
    FEATURE_NAMES as NQ_ACTION_FEATURE_NAMES,
    apply_policy as apply_nq_action_policy,
    features as nq_action_features,
)


def candidate(document_id: str, entailment: float, contradiction: float) -> dict:
    return {
        "document_id": document_id,
        "entailment": entailment,
        "neutral": 1.0 - entailment - contradiction,
        "contradiction": contradiction,
    }


def test_fever_policy_maps_empty_retrieval_to_nei_without_citations():
    assert select_prediction({"case_id": "fever:empty", "candidates": []}) == (
        "not_enough_info",
        [],
    )


def test_fever_policy_selects_relation_and_thresholded_same_relation_evidence():
    raw = {
        "case_id": "fever:supports",
        "candidates": [
            candidate("gold-a", 0.90, 0.01),
            candidate("gold-b", 0.05, 0.02),
            candidate("noise", 0.04, 0.80),
        ],
    }
    # The citation floor is 0.90 / 20 = 0.045, so the lower-scoring gold
    # document is retained but the contradiction-only distractor is not.
    assert select_prediction(raw) == ("supports", ["gold-a", "gold-b"])


def test_fever_policy_abstains_when_neither_direction_clears_threshold():
    raw = {
        "case_id": "fever:uncertain",
        "candidates": [candidate("document", 0.74, 0.10)],
    }
    assert select_prediction(raw) == ("not_enough_info", [])


def test_fever_policy_rejects_duplicate_candidate_documents():
    raw = {
        "case_id": "fever:duplicate",
        "candidates": [
            candidate("same", 0.90, 0.01),
            candidate("same", 0.80, 0.02),
        ],
    }
    with pytest.raises(ValueError, match="duplicate candidate documents"):
        select_prediction(raw)


def test_fever_scoring_requires_one_complete_gold_evidence_set():
    raw = {
        "case_id": "fever:complete",
        "candidates": [
            candidate("gold-a", 0.90, 0.01),
            candidate("gold-b", 0.05, 0.02),
        ],
        "retrieved_document_ids": ["gold-a", "gold-b"],
        "gold_document_ids": ["gold-a", "gold-b"],
        "gold_evidence_sets": [["gold-a", "gold-b"]],
        "gold": {
            "answerability": "answerable",
            "answers": [],
            "label": "supports",
        },
    }
    result = score_case(raw)
    assert result["label_correct"] is True
    assert result["evidence_complete"] is True
    assert result["joint_correct"] is True
    assert result["fail_closed"] is False


def hotpot_result(prediction: str, citations: list[str], exact: float = 1.0) -> dict:
    return {
        "case_id": "hotpotqa:test",
        "benchmark_id": "hotpotqa",
        "prediction": prediction,
        "fail_closed": False,
        "gold": {"answerability": "answerable", "answers": ["Bangkok"], "label": None},
        "retrieval_any_gold": True,
        "retrieval_complete_gold": True,
        "citation_precision": 1.0,
        "citation_recall": 0.5,
        "cited_document_ids": citations,
        "answer_exact_match": exact,
        "answer_f1": exact,
        "answerability_correct": True,
        "retrieval_any_complete_evidence_set": True,
        "evidence_complete": False,
        "legacy_union_evidence_complete": False,
        "joint_correct": False,
        "legacy_union_joint_correct": False,
    }


def test_hotpot_composition_unions_citations_only_when_answers_agree():
    work = {"gold_document_ids": ["bridge", "answer"]}
    selected = compose_case(
        hotpot_result("Bangkok", ["answer"]),
        hotpot_result("  bangkok ", ["bridge"]),
        work,
    )
    assert selected["cited_document_ids"] == ["answer", "bridge"]
    assert selected["evidence_complete"] is True
    assert selected["joint_correct"] is True

    disagreement = compose_case(
        hotpot_result("Bangkok", ["answer"]),
        hotpot_result("Paris", ["bridge"], exact=0.0),
        work,
    )
    assert disagreement["cited_document_ids"] == ["answer"]
    assert disagreement["joint_correct"] is False


def test_full_specialist_composition_rejects_fail_closed_case(monkeypatch):
    import scripts.benchmarks.compose_specialist_full as module

    monkeypatch.setattr(
        module,
        "EXPECTED_COUNTS",
        {"hotpotqa": 1, "natural_questions": 1, "fever": 1},
    )
    baseline = [
        {"case_id": benchmark, "benchmark_id": benchmark}
        for benchmark in module.EXPECTED_COUNTS
    ]
    specialists = {
        benchmark: [
            {
                "case_id": benchmark,
                "benchmark_id": benchmark,
                "fail_closed": benchmark == "fever",
            }
        ]
        for benchmark in module.EXPECTED_COUNTS
    }
    with pytest.raises(ValueError, match="fail-closed"):
        compose_rows(baseline, specialists)


def test_nq_action_features_are_fixed_and_policy_can_only_abstain():
    raw = {
        "case_id": "natural_questions:test",
        "query": "Where was Ada born?",
        "gold": {"answerability": "unanswerable", "answers": [], "label": None},
        "gold_document_ids": [],
        "candidates": [
            {"answer": "Paris", "score": 0.4},
            {"answer": "", "score": 0.8},
        ],
    }
    baseline = {
        "case_id": raw["case_id"],
        "benchmark_id": "natural_questions",
        "prediction": "Paris",
        "cited_document_ids": ["d1"],
        "fail_closed": False,
        "gold": raw["gold"],
        "retrieval_any_gold": None,
        "retrieval_complete_gold": None,
        "citation_precision": 0.0,
        "citation_recall": None,
        "answer_exact_match": 0.0,
        "answer_f1": 0.0,
        "answerability_correct": False,
        "retrieval_any_complete_evidence_set": True,
        "evidence_complete": True,
        "legacy_union_evidence_complete": True,
        "joint_correct": False,
        "legacy_union_joint_correct": False,
    }
    assert len(nq_action_features(raw, baseline)) == len(NQ_ACTION_FEATURE_NAMES)
    selected = apply_nq_action_policy(
        [raw], {raw["case_id"]: baseline}, {raw["case_id"]: 0.9}, 0.45
    )[0]
    assert selected["prediction"] == "__UNANSWERABLE__"
    assert selected["cited_document_ids"] == []
    assert selected["joint_correct"] is True


def test_fever_local_label_uses_frozen_threshold_and_relation_direction():
    assert local_label({"candidates": []}) == ("not_enough_info", None, 0.0)
    below = {
        "candidates": [
            {"entailment": 0.74, "contradiction": 0.10, "neutral": 0.16}
        ]
    }
    assert local_label(below)[0] == "not_enough_info"
    refutes = {
        "candidates": [
            {"entailment": 0.01, "contradiction": 0.95, "neutral": 0.04}
        ]
    }
    assert local_label(refutes) == ("refutes", "contradiction", 0.95)
