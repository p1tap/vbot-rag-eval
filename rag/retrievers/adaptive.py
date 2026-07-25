"""Fail-closed policy for choosing dense or reranked retrieval per query."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


REQUIRED_FEATURES = (
    "dense_top1",
    "dense_top1_top2_margin",
    "dense_top4_mean",
    "dense_top4_min",
)


def dense_confidence_features(scores) -> dict[str, float]:
    values = [float(score) for score in scores]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("dense confidence scores must be finite and non-empty")
    top1 = values[0]
    top2 = values[1] if len(values) > 1 else top1
    top = values[: min(4, len(values))]
    return {
        "dense_top1": top1,
        "dense_top1_top2_margin": top1 - top2,
        "dense_top4_mean": sum(top) / len(top),
        "dense_top4_min": min(top),
    }


def validate_features(features: Mapping[str, float]) -> dict[str, float]:
    resolved = {}
    for name in REQUIRED_FEATURES:
        value = features.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"adaptive retrieval feature is invalid: {name}")
        resolved[name] = float(value)
    return resolved


@dataclass(frozen=True)
class AdaptiveRetrievalPolicy:
    """Route low-confidence dense results through the bounded reranker.

    Unknown or malformed features fail closed to the more conservative
    reranked path. A policy is scoped to a benchmark and cutoff so thresholds
    are never silently generalized to a different retrieval distribution.
    """

    benchmark_id: str
    k: int
    top1_threshold: float
    margin_threshold: float
    policy_version: str = "1.0.0"

    @classmethod
    def from_dict(cls, payload: Mapping) -> "AdaptiveRetrievalPolicy":
        if payload.get("policy_version") != "1.0.0":
            raise ValueError("unsupported adaptive retrieval policy version")
        routing = payload.get("routing", {})
        return cls(
            benchmark_id=str(payload["benchmark_id"]),
            k=int(payload["k"]),
            top1_threshold=float(routing["dense_top1_lte"]),
            margin_threshold=float(routing["dense_top1_top2_margin_lte"]),
        )

    def route(self, features: Mapping[str, float]) -> str:
        try:
            values = validate_features(features)
        except ValueError:
            return "rerank"
        if (
            values["dense_top1"] <= self.top1_threshold
            or values["dense_top1_top2_margin"] <= self.margin_threshold
        ):
            return "rerank"
        return "dense"

    def choose(self, row: Mapping) -> tuple[str, tuple[str, ...]]:
        route = self.route(row.get("features", {}))
        field = (
            "reranked_document_ids"
            if route == "rerank"
            else "dense_ranked_document_ids"
        )
        ranking = row.get(field)
        if not isinstance(ranking, list) or not all(
            isinstance(document_id, str) and document_id for document_id in ranking
        ):
            raise ValueError(f"adaptive retrieval row has invalid {field}")
        return route, tuple(ranking)
