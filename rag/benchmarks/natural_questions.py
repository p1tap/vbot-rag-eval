from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .base import BenchmarkAdapter, canonical_json_sha256


YES_NO_LABELS = {-1: None, 0: "no", 1: "yes"}


def _parallel_rows(columns: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("Natural Questions parallel-list fields have unequal lengths")
    return [
        {name: values[position] for name, values in columns.items()}
        for position in range(next(iter(lengths), 0))
    ]


class NaturalQuestionsAdapter(BenchmarkAdapter):
    benchmark_id = "natural_questions"
    source_version = (
        "Natural Questions v1.0 dev / Hugging Face commit "
        "e8103d566bef4154c2c12b17c6095ec5275840cc"
    )
    source_id_field = "id"

    def normalize(self, record: Mapping[str, Any], source_split: str) -> dict[str, Any]:
        required = {
            "id",
            "document",
            "question",
            "long_answer_candidates",
            "annotations",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(
                f"Natural Questions record missing fields: {', '.join(missing)}"
            )

        source_id = str(record["id"])
        question = str(record["question"]["text"]).strip()
        if not question:
            raise ValueError(f"Natural Questions record {source_id} has empty question")

        document = record["document"]
        token_columns = document["tokens"]
        tokens = list(token_columns["token"])
        html_flags = list(token_columns["is_html"])
        if len(tokens) != len(html_flags):
            raise ValueError(f"Natural Questions record {source_id} has token drift")

        candidates = _parallel_rows(record["long_answer_candidates"])
        documents: list[dict[str, Any]] = []
        candidate_ids: dict[int, str] = {}
        for candidate_index, candidate in enumerate(candidates):
            start = candidate["start_token"]
            end = candidate["end_token"]
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(
                    f"Natural Questions record {source_id} has non-integer candidate span"
                )
            if start < 0 or end <= start or end > len(tokens):
                raise ValueError(
                    f"Natural Questions record {source_id} has invalid candidate span "
                    f"{candidate_index}: [{start}, {end})"
                )
            text = " ".join(
                token
                for token, is_html in zip(tokens[start:end], html_flags[start:end])
                if not is_html
            ).strip()
            if not text:
                text = "[structural candidate with no visible text]"
            document_id = f"natural_questions:{source_id}:candidate:{candidate_index}"
            candidate_ids[candidate_index] = document_id
            title = str(document["title"]).strip() or "Untitled Wikipedia page"
            documents.append(
                {
                    "id": document_id,
                    "title": f"{title} — candidate {candidate_index}",
                    "sentences": [text],
                    "source_position": candidate_index,
                    "content_sha256": canonical_json_sha256(
                        {
                            "candidate_index": candidate_index,
                            "start_token": start,
                            "end_token": end,
                            "text": text,
                        }
                    ),
                }
            )

        annotation_columns = record["annotations"]
        annotation_count = len(annotation_columns["id"])
        if annotation_count != 5:
            raise ValueError(
                f"Natural Questions dev record {source_id} has {annotation_count} "
                "annotations; expected 5"
            )
        annotations = _parallel_rows(annotation_columns)
        votes: list[dict[str, Any]] = []
        positive_candidate_counts: Counter[int] = Counter()
        all_positive_candidates: set[int] = set()
        short_answer_votes = 0
        yes_no_votes = 0
        short_answer_variants: list[str] = []
        yes_no_variants: list[str] = []

        for annotation in annotations:
            long_answer = annotation["long_answer"]
            candidate_index = long_answer["candidate_index"]
            evidence: list[dict[str, Any]] = []
            if candidate_index >= 0:
                if candidate_index not in candidate_ids:
                    raise ValueError(
                        f"Natural Questions record {source_id} references missing "
                        f"candidate {candidate_index}"
                    )
                positive_candidate_counts[candidate_index] += 1
                all_positive_candidates.add(candidate_index)
                evidence.append(
                    {
                        "document_id": candidate_ids[candidate_index],
                        "sentence_index": 0,
                    }
                )

            short_answers = annotation["short_answers"]
            short_texts = [str(text).strip() for text in short_answers["text"]]
            short_texts = list(dict.fromkeys(text for text in short_texts if text))
            yes_no_value = annotation["yes_no_answer"]
            if yes_no_value not in YES_NO_LABELS:
                raise ValueError(
                    f"Natural Questions record {source_id} has unknown yes/no label "
                    f"{yes_no_value}"
                )
            yes_no_answer = YES_NO_LABELS[yes_no_value]
            answers = short_texts or ([yes_no_answer] if yes_no_answer else [])
            if short_texts:
                short_answer_votes += 1
            if yes_no_answer:
                yes_no_votes += 1
            for answer in short_texts:
                if answer not in short_answer_variants:
                    short_answer_variants.append(answer)
            if yes_no_answer and yes_no_answer not in yes_no_variants:
                yes_no_variants.append(yes_no_answer)
            votes.append(
                {
                    "id": str(annotation["id"]),
                    "answerability": (
                        "answerable" if candidate_index >= 0 else "unanswerable"
                    ),
                    "answers": answers,
                    "evidence": evidence,
                }
            )

        long_answer_votes = sum(positive_candidate_counts.values())
        answerable = long_answer_votes >= 2
        consensus_candidates = {
            index for index, count in positive_candidate_counts.items() if count >= 2
        }
        if answerable and not consensus_candidates:
            consensus_candidates = set(positive_candidate_counts)

        supporting_evidence = [
            {
                "document_id": candidate_ids[index],
                "sentence_index": 0,
                "text": documents[index]["sentences"][0],
            }
            for index in sorted(consensus_candidates)
        ]
        if short_answer_votes >= 2:
            answer_variants = short_answer_variants
        elif yes_no_votes >= 2:
            answer_variants = yes_no_variants
        else:
            answer_variants = [evidence["text"] for evidence in supporting_evidence]

        return {
            "schema_version": "1.0.0",
            "id": f"natural_questions:{source_split}:{source_id}",
            "benchmark_id": self.benchmark_id,
            "task_type": "extractive_qa",
            "source_split": source_split,
            "query": question,
            "gold": {
                "answers": answer_variants if answerable else [],
                "label": None,
                "answerability": "answerable" if answerable else "unanswerable",
            },
            "documents": documents,
            "supporting_evidence": supporting_evidence,
            "distractor_document_ids": [
                document["id"]
                for index, document in enumerate(documents)
                if index not in all_positive_candidates
            ],
            "annotation_votes": votes,
            "provenance": {
                "source_dataset": "Natural Questions",
                "source_version": self.source_version,
                "source_record_id": source_id,
                "source_record_sha256": (
                    str(record["_source_record_sha256"])
                    if "_source_record_sha256" in record
                    else canonical_json_sha256(record)
                ),
                "annotation_method": "inherited_five_way_human_annotation",
                "human_annotated": True,
                "transformation_version": "1.0.0",
            },
            "metadata": {
                "document_url": str(document["url"]),
                "long_answer_votes": long_answer_votes,
                "short_answer_votes": short_answer_votes,
                "yes_no_votes": yes_no_votes,
                "candidate_count": len(documents),
            },
        }
