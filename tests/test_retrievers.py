from __future__ import annotations

import unittest
from contextlib import closing
import json
import sqlite3
from unittest.mock import patch

import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.retrieval_eval import aggregate_results, evaluate_case
from rag.retrieve import load_index, retrieve
from scripts.evaluate_v2_retrieval import confidence_features, rank_headings, score_case
from rag.retrievers import (
    AdaptiveRetrievalPolicy,
    BM25Retriever,
    CrossEncoderReranker,
    RetrievalDocument,
    RetrievalHit,
    SQLiteFTSBM25Index,
    TwoStageFeverBM25Index,
    reciprocal_rank_fusion,
)


class StubPairScorer:
    name = "stub"

    def __init__(self, scores):
        self.scores = scores

    def score(self, query, documents):
        return [self.scores[document.id] for document in documents]


class RetrieverTests(unittest.TestCase):
    def test_index_loader_validates_count_identity_and_normalization(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [
                {"id": "a", "heading_key": "doc::a", "heading": "A", "text": "x"},
                {"id": "b", "heading_key": "doc::b", "heading": "B", "text": "y"},
            ]
            (root / "meta.json").write_text(
                json.dumps({"n_chunks": 2}), encoding="utf-8"
            )
            (root / "chunks.jsonl").write_text(
                "\n".join(json.dumps(chunk) for chunk in chunks), encoding="utf-8"
            )
            np.save(root / "embeddings.npy", np.eye(2, dtype=np.float32))
            vectors, loaded = load_index(root)
            self.assertEqual(vectors.shape, (2, 2))
            self.assertEqual([chunk["id"] for chunk in loaded], ["a", "b"])

            np.save(
                root / "embeddings.npy",
                np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "not L2-normalized"):
                load_index(root)

    def test_v2_retrieval_metrics_keep_complete_evidence_strict(self) -> None:
        case = {
            "acceptable_evidence": ["doc::a::span-1", "doc::b::span-2"],
            "required_claims": [
                {"id": "c1", "evidence_ids": ["doc::a::span-1"]},
                {"id": "c2", "evidence_ids": ["doc::a::span-1", "doc::b::span-2"]},
            ],
        }
        ranked = [
            {"heading_key": "doc::a", "score": 0.9},
            {"heading_key": "doc::distractor", "score": 0.8},
            {"heading_key": "doc::b", "score": 0.7},
        ]
        at_two = score_case(case, ranked, 2)
        at_three = score_case(case, ranked, 3)
        self.assertTrue(at_two["any_gold"])
        self.assertFalse(at_two["complete_gold"])
        self.assertEqual(at_two["required_claim_coverage"], 0.5)
        self.assertTrue(at_three["complete_required_claims"])

    def test_confidence_features_record_duplicate_heading_pressure(self) -> None:
        chunks = [
            {"id": "a0", "heading_key": "doc::a"},
            {"id": "a1", "heading_key": "doc::a"},
            {"id": "b0", "heading_key": "doc::b"},
        ]
        ranked = rank_headings(
            np.array([0.9, 0.85, 0.8]), chunks, 3, deduplicate=False
        )
        unique = rank_headings(
            np.array([0.9, 0.85, 0.8]), chunks, 2, deduplicate=True
        )
        features = confidence_features(unique, ranked, k=2)
        self.assertEqual(features["raw_chunks_needed_for_distinct_topk"], 3)
        self.assertAlmostEqual(features["top1_top2_margin"], 0.1)

    @patch("rag.retrieve.embed_queries")
    def test_v2_heading_dedup_uses_next_distinct_result(self, embed_queries) -> None:
        embed_queries.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        vectors = np.array(
            [[1.0, 0.0], [0.99, 0.0], [0.8, 0.0]], dtype=np.float32
        )
        chunks = [
            {"id": "a0", "heading_key": "doc::a", "heading": "A", "text": "first"},
            {"id": "a1", "heading_key": "doc::a", "heading": "A", "text": "second"},
            {"id": "b0", "heading_key": "doc::b", "heading": "B", "text": "third"},
        ]
        ordinary = retrieve("q", k=2, index=(vectors, chunks))
        distinct = retrieve("q", k=2, index=(vectors, chunks), unique_headings=True)
        self.assertEqual([row["id"] for row in ordinary], ["a0", "a1"])
        self.assertEqual([row["id"] for row in distinct], ["a0", "b0"])

    def test_bm25_ranks_lexical_match_and_breaks_ties_by_id(self) -> None:
        documents = [
            RetrievalDocument("b", "Other", "unrelated words"),
            RetrievalDocument("a", "Gateway", "circuit breaker cooldown"),
            RetrievalDocument("c", "Other", "unrelated words"),
        ]
        hits = BM25Retriever().rank("circuit breaker", documents, 3)
        self.assertEqual([hit.document.id for hit in hits], ["a", "b", "c"])

    def test_case_metrics_keep_any_and_all_evidence_separate(self) -> None:
        case = {
            "id": "case-1",
            "query": "alpha",
            "documents": [
                {"id": "d1", "title": "Alpha", "sentences": ["alpha"]},
                {"id": "d2", "title": "Beta", "sentences": ["beta"]},
            ],
            "supporting_evidence": [
                {"document_id": "d1", "sentence_index": 0, "text": "alpha"},
                {"document_id": "d2", "sentence_index": 0, "text": "beta"},
            ],
        }
        result = evaluate_case(case, BM25Retriever(), max_k=2)
        metrics = aggregate_results([result], [1, 2])
        self.assertEqual(metrics["1"]["case_recall_any"], 1.0)
        self.assertEqual(metrics["1"]["case_recall_all"], 0.0)
        self.assertEqual(metrics["1"]["evidence_recall"], 0.5)
        self.assertEqual(metrics["2"]["case_recall_all"], 1.0)

    def test_rrf_combines_rankings_deterministically(self) -> None:
        a = RetrievalDocument("a", "A", "")
        b = RetrievalDocument("b", "B", "")
        fused = reciprocal_rank_fusion(
            [
                [RetrievalHit(a, 1.0, 1), RetrievalHit(b, 0.5, 2)],
                [RetrievalHit(b, 1.0, 1), RetrievalHit(a, 0.5, 2)],
            ],
            k=2,
        )
        self.assertEqual([hit.document.id for hit in fused], ["a", "b"])

    def test_rrf_weights_can_favor_the_stronger_ranking(self) -> None:
        a = RetrievalDocument("a", "A", "")
        b = RetrievalDocument("b", "B", "")
        fused = reciprocal_rank_fusion(
            [
                [RetrievalHit(a, 1.0, 1), RetrievalHit(b, 0.5, 2)],
                [RetrievalHit(b, 1.0, 1), RetrievalHit(a, 0.5, 2)],
            ],
            k=2,
            weights=[1.0, 3.0],
        )
        self.assertEqual([hit.document.id for hit in fused], ["b", "a"])

    def test_rrf_rejects_invalid_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "match the number"):
            reciprocal_rank_fusion([[]], k=1, weights=[1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            reciprocal_rank_fusion([[], []], k=1, weights=[0.0, 0.0])

    def test_cross_encoder_reranks_only_bounded_first_stage_candidates(self) -> None:
        documents = [
            RetrievalDocument("a", "alpha", "matching lexical text"),
            RetrievalDocument("b", "beta", "matching lexical text"),
            RetrievalDocument("c", "gamma", "unrelated"),
        ]
        reranker = CrossEncoderReranker(
            BM25Retriever(),
            StubPairScorer({"a": 0.1, "b": 0.9}),
            candidate_k=2,
        )
        hits = reranker.rank("matching lexical text", documents, 2)
        self.assertEqual([hit.document.id for hit in hits], ["b", "a"])

    def test_cross_encoder_keeps_first_stage_order_for_equal_scores(self) -> None:
        documents = [
            RetrievalDocument("a", "alpha", "match"),
            RetrievalDocument("b", "beta", "match"),
        ]
        reranker = CrossEncoderReranker(
            BM25Retriever(),
            StubPairScorer({"a": 1.0, "b": 1.0}),
            candidate_k=2,
        )
        hits = reranker.rank("match", documents, 2)
        self.assertEqual([hit.document.id for hit in hits], ["a", "b"])

    def test_adaptive_policy_routes_low_confidence_and_fails_closed(self) -> None:
        policy = AdaptiveRetrievalPolicy(
            benchmark_id="fixture",
            k=4,
            top1_threshold=0.7,
            margin_threshold=0.1,
        )
        confident = {
            "dense_top1": 0.9,
            "dense_top1_top2_margin": 0.2,
            "dense_top4_mean": 0.7,
            "dense_top4_min": 0.6,
        }
        uncertain = {**confident, "dense_top1_top2_margin": 0.05}
        self.assertEqual(policy.route(confident), "dense")
        self.assertEqual(policy.route(uncertain), "rerank")
        self.assertEqual(policy.route({"dense_top1": float("nan")}), "rerank")

    def test_adaptive_policy_selects_only_the_declared_ranking(self) -> None:
        policy = AdaptiveRetrievalPolicy(
            benchmark_id="fixture",
            k=4,
            top1_threshold=0.7,
            margin_threshold=0.1,
        )
        row = {
            "features": {
                "dense_top1": 0.6,
                "dense_top1_top2_margin": 0.2,
                "dense_top4_mean": 0.5,
                "dense_top4_min": 0.4,
            },
            "dense_ranked_document_ids": ["dense"],
            "reranked_document_ids": ["reranked"],
        }
        route, ranking = policy.choose(row)
        self.assertEqual(route, "rerank")
        self.assertEqual(ranking, ("reranked",))

    def test_fever_runner_refuses_evidence_only_scope(self) -> None:
        from scripts.benchmarks.run_retrieval import run

        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-global-index.sqlite"
            with self.assertRaisesRegex(ValueError, "global index"):
                run("fever", [1], missing)

    def test_sqlite_fts_ranks_global_documents(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "CREATE VIRTUAL TABLE pages USING fts5("
                    "page_id UNINDEXED, title, text, tokenize='porter unicode61')"
                )
                connection.executemany(
                    "INSERT INTO pages(page_id, title, text) VALUES (?, ?, ?)",
                    [
                        ("other", "Other", "unrelated document"),
                        ("target", "Circuit breaker", "cooldown and failover"),
                    ],
                )
                connection.execute(
                    "CREATE TABLE page_lookup ("
                    "page_id TEXT PRIMARY KEY, fts_rowid INTEGER NOT NULL UNIQUE) "
                    "WITHOUT ROWID"
                )
                connection.execute(
                    "INSERT INTO page_lookup(page_id, fts_rowid) "
                    "SELECT page_id, rowid FROM pages"
                )
                connection.commit()
            hits = SQLiteFTSBM25Index(path).rank("circuit breaker cooldown", 2)
            self.assertEqual(hits[0].document.id, "target")
            with TwoStageFeverBM25Index(path, candidate_limit=10) as index:
                two_stage_hits = index.rank("circuit breaker cooldown", 2)
            self.assertEqual(two_stage_hits[0].document.id, "target")

    def test_published_retrieval_evidence_drops_machine_latency(self) -> None:
        from scripts.benchmarks.publish_retrieval import publish

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "local.json"
            source.write_text(
                json.dumps(
                    {
                        "report_version": "1.0.0",
                        "benchmark_id": "fixture",
                        "retriever": {"name": "bm25"},
                        "source": {"sha256": "abc"},
                        "cases": {"total": 1},
                        "metrics": {"1": {"mrr": 1.0}},
                        "latency_ms_per_query": {"median": 9.0},
                    }
                ),
                encoding="utf-8",
            )
            output = publish(source, root / "published")
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotIn("latency_ms_per_query", result)
            self.assertEqual(result["status"], "retrieval_only_generation_pending")

    def test_retrieval_gate_promotes_e5_and_rejects_hybrid_chain_regression(self) -> None:
        from scripts.benchmarks.compare_retrieval_gate import compare

        root = Path(__file__).resolve().parents[1] / "reports" / "public-benchmarks" / "retrieval"
        bm25 = json.loads((root / "hotpotqa-bm25.json").read_text(encoding="utf-8"))
        e5 = json.loads(
            (root / "hotpotqa-e5-small-v2.json").read_text(encoding="utf-8")
        )
        hybrid = json.loads(
            (root / "hotpotqa-bm25-e5-rrf.json").read_text(encoding="utf-8")
        )
        promoted, _ = compare(bm25, e5, k=4, tolerance=0.005)
        rejected, rows = compare(e5, hybrid, k=4, tolerance=0.005)
        self.assertEqual(promoted, 0)
        self.assertEqual(rejected, 2)
        self.assertIn(
            "guardrail regressed",
            {verdict for _, _, _, _, verdict in rows},
        )


if __name__ == "__main__":
    unittest.main()
