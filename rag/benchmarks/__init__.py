"""Adapters for independently annotated public evaluation benchmarks."""

from .base import BenchmarkAdapter, canonical_json_sha256, stable_hash_sample
from .hotpotqa import HotpotQAAdapter
from .natural_questions import NaturalQuestionsAdapter
from .fever import FeverAdapter

__all__ = [
    "BenchmarkAdapter",
    "HotpotQAAdapter",
    "NaturalQuestionsAdapter",
    "FeverAdapter",
    "canonical_json_sha256",
    "stable_hash_sample",
]
