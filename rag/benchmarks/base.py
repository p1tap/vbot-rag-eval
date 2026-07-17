from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stable_hash_sample(
    records: Sequence[Mapping[str, Any]], limit: int | None, id_field: str
) -> list[Mapping[str, Any]]:
    """Select a reproducible, order-independent pseudo-random slice."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    ranked = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(str(record[id_field]).encode("utf-8")).hexdigest(),
            str(record[id_field]),
        ),
    )
    return ranked if limit is None else ranked[:limit]


class BenchmarkAdapter(ABC):
    benchmark_id: str
    source_version: str
    source_id_field: str

    @abstractmethod
    def normalize(self, record: Mapping[str, Any], source_split: str) -> dict[str, Any]:
        """Convert one source record without weakening its gold annotations."""

    def normalize_many(
        self,
        records: Sequence[Mapping[str, Any]],
        source_split: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if limit is not None and limit > len(records):
            raise ValueError(
                f"requested {limit} cases but source contains only {len(records)}"
            )
        seen: set[str] = set()
        for record in records:
            source_id = str(record.get(self.source_id_field, ""))
            if not source_id:
                raise ValueError(f"record missing {self.source_id_field}")
            if source_id in seen:
                raise ValueError(f"duplicate source record ID: {source_id}")
            seen.add(source_id)

        selected = stable_hash_sample(records, limit, self.source_id_field)
        return [self.normalize(record, source_split) for record in selected]


def render_jsonl(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)
