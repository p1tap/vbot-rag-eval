from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BenchmarkAdapter, canonical_json_sha256


LABELS = {
    "SUPPORTS": "supports",
    "REFUTES": "refutes",
    "NOT ENOUGH INFO": "not_enough_info",
}


class FeverAdapter(BenchmarkAdapter):
    benchmark_id = "fever"
    source_version = "FEVER 1.0 shared-task dev / June 2017 Wikipedia dump"
    source_id_field = "id"

    def normalize(self, record: Mapping[str, Any], source_split: str) -> dict[str, Any]:
        required = {"id", "claim", "label", "evidence", "_pages"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"FEVER record missing fields: {', '.join(missing)}")
        source_id = str(record["id"])
        claim = str(record["claim"]).strip()
        source_label = str(record["label"])
        if not claim:
            raise ValueError(f"FEVER record {source_id} has an empty claim")
        if source_label not in LABELS:
            raise ValueError(f"FEVER record {source_id} has unknown label {source_label}")

        evidence_pages = sorted(
            {
                item[2]
                for evidence_set in record["evidence"]
                for item in evidence_set
                if item[2] is not None
            }
        )
        pages = record["_pages"]
        missing_pages = sorted(set(evidence_pages) - pages.keys())
        if missing_pages:
            raise ValueError(
                f"FEVER record {source_id} is missing evidence pages: "
                f"{', '.join(missing_pages)}"
            )

        documents: list[dict[str, Any]] = []
        page_to_document: dict[str, dict[str, Any]] = {}
        for position, page_id in enumerate(evidence_pages):
            sentences = [str(sentence) for sentence in pages[page_id]["sentences"]]
            if not sentences:
                raise ValueError(f"FEVER evidence page {page_id} has no sentences")
            document_id = f"fever:{source_id}:page:{position}"
            document = {
                "id": document_id,
                "title": page_id.replace("_", " "),
                "sentences": sentences,
                "source_position": position,
                "content_sha256": canonical_json_sha256(
                    {"page_id": page_id, "sentences": sentences}
                ),
            }
            documents.append(document)
            page_to_document[page_id] = document

        annotation_votes: list[dict[str, Any]] = []
        supporting_evidence: list[dict[str, Any]] = []
        seen_support: set[tuple[str, int]] = set()
        for set_index, evidence_set in enumerate(record["evidence"]):
            vote_evidence: list[dict[str, Any]] = []
            annotation_ids: list[str] = []
            for item in evidence_set:
                if not isinstance(item, list) or len(item) != 4:
                    raise ValueError(
                        f"FEVER record {source_id} has malformed evidence tuple"
                    )
                annotation_id, _, page_id, sentence_index = item
                annotation_ids.append(str(annotation_id))
                if page_id is None and sentence_index is None:
                    continue
                if page_id not in page_to_document or not isinstance(sentence_index, int):
                    raise ValueError(
                        f"FEVER record {source_id} has unresolved evidence {page_id}"
                    )
                document = page_to_document[page_id]
                if sentence_index < 0 or sentence_index >= len(document["sentences"]):
                    raise ValueError(
                        f"FEVER record {source_id} references missing sentence "
                        f"{page_id}[{sentence_index}]"
                    )
                reference = {
                    "document_id": document["id"],
                    "sentence_index": sentence_index,
                }
                vote_evidence.append(reference)
                key = (document["id"], sentence_index)
                if key not in seen_support:
                    seen_support.add(key)
                    supporting_evidence.append(
                        {
                            **reference,
                            "text": document["sentences"][sentence_index],
                        }
                    )
            vote_id = ",".join(dict.fromkeys(annotation_ids)) or f"set-{set_index}"
            annotation_votes.append(
                {
                    "id": f"fever-evidence-set:{vote_id}:{set_index}",
                    "answerability": (
                        "unanswerable"
                        if source_label == "NOT ENOUGH INFO"
                        else "answerable"
                    ),
                    "answers": [],
                    "evidence": vote_evidence,
                }
            )

        answerable = source_label != "NOT ENOUGH INFO"
        if answerable and not supporting_evidence:
            raise ValueError(f"FEVER record {source_id} has no resolvable evidence")
        if not annotation_votes:
            raise ValueError(f"FEVER record {source_id} has no annotation sets")

        return {
            "schema_version": "1.0.0",
            "id": f"fever:{source_split}:{source_id}",
            "benchmark_id": self.benchmark_id,
            "task_type": "fact_verification",
            "source_split": source_split,
            "query": claim,
            "gold": {
                "answers": [],
                "label": LABELS[source_label],
                "answerability": "answerable" if answerable else "unanswerable",
            },
            "documents": documents,
            "supporting_evidence": supporting_evidence,
            "distractor_document_ids": [],
            "annotation_votes": annotation_votes,
            "provenance": {
                "source_dataset": "FEVER",
                "source_version": self.source_version,
                "source_record_id": source_id,
                "source_record_sha256": (
                    str(record["_source_record_sha256"])
                    if "_source_record_sha256" in record
                    else canonical_json_sha256(
                        {key: value for key, value in record.items() if key != "_pages"}
                    )
                ),
                "annotation_method": "inherited_human_verified_claim_and_evidence",
                "human_annotated": True,
                "transformation_version": "1.0.0",
            },
            "metadata": {
                "evidence_set_count": len(annotation_votes),
                "evidence_set_semantics": "any_complete_set",
                "global_corpus_required": True,
            },
        }
