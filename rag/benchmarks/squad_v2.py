from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BenchmarkAdapter, canonical_json_sha256


class SquadV2Adapter(BenchmarkAdapter):
    """Normalize one SQuAD 2.0 question with its oracle paragraph."""

    benchmark_id = "squad_v2"
    source_version = "SQuAD 2.0 dev (official Stanford release)"
    source_id_field = "id"

    def normalize(self, record: Mapping[str, Any], source_split: str) -> dict[str, Any]:
        required = {"id", "title", "context", "question", "answers", "is_impossible"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"SQuAD record missing fields: {', '.join(missing)}")

        source_id = str(record["id"]).strip()
        title = str(record["title"]).strip()
        context = str(record["context"]).strip()
        question = str(record["question"]).strip()
        if not source_id or not title or not context or not question:
            raise ValueError("SQuAD record contains an empty required text field")

        impossible = record["is_impossible"]
        if not isinstance(impossible, bool):
            raise ValueError(f"SQuAD record {source_id} has non-boolean is_impossible")
        answers = record["answers"]
        if not isinstance(answers, list):
            raise ValueError(f"SQuAD record {source_id} answers must be a list")

        variants: list[str] = []
        votes: list[dict[str, Any]] = []
        document_id = f"squad_v2:{source_id}:paragraph"
        for position, answer in enumerate(answers):
            if not isinstance(answer, Mapping) or set(answer) != {"text", "answer_start"}:
                raise ValueError(f"SQuAD record {source_id} has malformed answer")
            text = str(answer["text"]).strip()
            start = answer["answer_start"]
            if not text or not isinstance(start, int) or start < 0:
                raise ValueError(f"SQuAD record {source_id} has invalid answer span")
            if context[start : start + len(answer["text"])] != answer["text"]:
                raise ValueError(f"SQuAD record {source_id} answer span does not resolve")
            if text not in variants:
                variants.append(text)
            votes.append(
                {
                    "id": f"{source_id}:answer:{position}",
                    "answerability": "answerable",
                    "answers": [text],
                    "evidence": [{"document_id": document_id, "sentence_index": 0}],
                }
            )

        if impossible and answers:
            raise ValueError(f"SQuAD record {source_id} is impossible but has answers")
        if not impossible and not answers:
            raise ValueError(f"SQuAD record {source_id} is answerable but has no answers")
        if impossible:
            votes = [
                {
                    "id": f"{source_id}:unanswerable",
                    "answerability": "unanswerable",
                    "answers": [],
                    "evidence": [],
                }
            ]

        document = {
            "id": document_id,
            "title": title,
            "sentences": [context],
            "source_position": 0,
            "content_sha256": canonical_json_sha256(
                {"title": title, "context": context}
            ),
        }
        return {
            "schema_version": "1.0.0",
            "id": f"squad_v2:{source_split}:{source_id}",
            "benchmark_id": self.benchmark_id,
            "task_type": "extractive_qa",
            "source_split": source_split,
            "query": question,
            "gold": {
                "answers": [] if impossible else variants,
                "label": None,
                "answerability": "unanswerable" if impossible else "answerable",
            },
            "documents": [document],
            "supporting_evidence": (
                []
                if impossible
                else [
                    {
                        "document_id": document_id,
                        "sentence_index": 0,
                        "text": context,
                    }
                ]
            ),
            "distractor_document_ids": [],
            "annotation_votes": votes,
            "provenance": {
                "source_dataset": "SQuAD 2.0",
                "source_version": self.source_version,
                "source_record_id": source_id,
                "source_record_sha256": canonical_json_sha256(record),
                "annotation_method": "inherited_crowd_answer_or_adversarial_null_annotation",
                "human_annotated": True,
                "transformation_version": "1.0.0",
            },
            "metadata": {
                "oracle_context": True,
                "answer_annotation_count": len(answers),
            },
        }
