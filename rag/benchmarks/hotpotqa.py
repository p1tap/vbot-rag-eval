from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BenchmarkAdapter, canonical_json_sha256


class HotpotQAAdapter(BenchmarkAdapter):
    benchmark_id = "hotpotqa"
    source_version = "HotpotQA v1.0"
    source_id_field = "_id"

    def normalize(self, record: Mapping[str, Any], source_split: str) -> dict[str, Any]:
        required = {
            "_id",
            "question",
            "answer",
            "type",
            "level",
            "supporting_facts",
            "context",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"HotpotQA record missing fields: {', '.join(missing)}")

        source_id = str(record["_id"])
        question = str(record["question"]).strip()
        answer = str(record["answer"]).strip()
        if not question or not answer:
            raise ValueError(f"HotpotQA record {source_id} has an empty question or answer")

        documents: list[dict[str, Any]] = []
        title_to_document: dict[str, dict[str, Any]] = {}
        for position, raw_document in enumerate(record["context"]):
            if not isinstance(raw_document, list) or len(raw_document) != 2:
                raise ValueError(f"HotpotQA record {source_id} has malformed context")
            title = str(raw_document[0]).strip()
            sentences = [str(sentence) for sentence in raw_document[1]]
            if not title or not sentences:
                raise ValueError(f"HotpotQA record {source_id} has empty context")
            if title in title_to_document:
                raise ValueError(
                    f"HotpotQA record {source_id} has duplicate context title: {title}"
                )
            document_id = f"hotpotqa:{source_id}:doc:{position}"
            document = {
                "id": document_id,
                "title": title,
                "sentences": sentences,
                "source_position": position,
                "content_sha256": canonical_json_sha256(
                    {"title": title, "sentences": sentences}
                ),
            }
            documents.append(document)
            title_to_document[title] = document

        supporting_evidence: list[dict[str, Any]] = []
        supporting_document_ids: set[str] = set()
        seen_evidence: set[tuple[str, int]] = set()
        for raw_fact in record["supporting_facts"]:
            if not isinstance(raw_fact, list) or len(raw_fact) != 2:
                raise ValueError(
                    f"HotpotQA record {source_id} has malformed supporting fact"
                )
            title = str(raw_fact[0])
            sentence_index = raw_fact[1]
            if title not in title_to_document:
                raise ValueError(
                    f"HotpotQA record {source_id} references missing document: {title}"
                )
            if not isinstance(sentence_index, int):
                raise ValueError(
                    f"HotpotQA record {source_id} has non-integer sentence index"
                )
            document = title_to_document[title]
            if sentence_index < 0 or sentence_index >= len(document["sentences"]):
                raise ValueError(
                    f"HotpotQA record {source_id} references missing sentence "
                    f"{title}[{sentence_index}]"
                )
            evidence_key = (document["id"], sentence_index)
            if evidence_key in seen_evidence:
                raise ValueError(
                    f"HotpotQA record {source_id} has duplicate supporting fact "
                    f"{title}[{sentence_index}]"
                )
            seen_evidence.add(evidence_key)
            supporting_document_ids.add(document["id"])
            supporting_evidence.append(
                {
                    "document_id": document["id"],
                    "sentence_index": sentence_index,
                    "text": document["sentences"][sentence_index],
                }
            )

        if not supporting_evidence:
            raise ValueError(f"HotpotQA record {source_id} has no supporting facts")

        return {
            "schema_version": "1.0.0",
            "id": f"hotpotqa:{source_split}:{source_id}",
            "benchmark_id": self.benchmark_id,
            "task_type": "multi_hop_qa",
            "source_split": source_split,
            "query": question,
            "gold": {
                "answers": [answer],
                "label": None,
                "answerability": "answerable",
            },
            "documents": documents,
            "supporting_evidence": supporting_evidence,
            "distractor_document_ids": [
                document["id"]
                for document in documents
                if document["id"] not in supporting_document_ids
            ],
            "annotation_votes": [
                {
                    "id": "hotpotqa-gold",
                    "answerability": "answerable",
                    "answers": [answer],
                    "evidence": [
                        {
                            "document_id": evidence["document_id"],
                            "sentence_index": evidence["sentence_index"],
                        }
                        for evidence in supporting_evidence
                    ],
                }
            ],
            "provenance": {
                "source_dataset": "HotpotQA",
                "source_version": self.source_version,
                "source_record_id": source_id,
                "source_record_sha256": canonical_json_sha256(record),
                "annotation_method": "inherited_human_annotation",
                "human_annotated": True,
                "transformation_version": "1.0.0",
            },
            "metadata": {
                "question_type": str(record["type"]),
                "difficulty": str(record["level"]),
            },
        }
