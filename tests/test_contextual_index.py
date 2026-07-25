from __future__ import annotations

import unittest

from rag.ingest import load_chunks, load_document_contexts, provenance_context_header
from rag.late_chunking import assemble_document, token_indexes_for_span


class ContextualIndexTests(unittest.TestCase):
    def test_default_chunks_remain_context_free(self) -> None:
        chunks = load_chunks()
        self.assertTrue(chunks)
        self.assertTrue(all("context_header" not in chunk for chunk in chunks))

    def test_provenance_headers_are_hash_bound_and_factual(self) -> None:
        contexts, manifest = load_document_contexts()
        chunks = load_chunks(context_mode="provenance_header")
        self.assertEqual(
            {chunk["doc_id"] for chunk in chunks},
            set(contexts),
        )
        self.assertEqual(manifest["corpus_id"], "vbot-public-safe")
        first = chunks[0]
        self.assertEqual(
            first["context_header"],
            provenance_context_header(first, contexts[first["doc_id"]]),
        )
        self.assertIn("Repository:", first["context_header"])
        self.assertIn("Section:", first["context_header"])

    def test_context_header_rejects_cross_document_metadata(self) -> None:
        chunk = {"doc_id": "a", "heading": "Section"}
        context = {
            "document_id": "b",
            "repository": "repo",
            "source_path": "path",
            "status": "active",
            "access_class": "public",
        }
        with self.assertRaisesRegex(ValueError, "do not match"):
            provenance_context_header(chunk, context)

    def test_late_chunk_document_spans_round_trip_to_source_pieces(self) -> None:
        chunks = [
            {"doc_id": "doc", "heading": "First", "text": "alpha beta"},
            {"doc_id": "doc", "heading": "Second", "text": "gamma delta"},
        ]
        text, spans = assemble_document(chunks)
        self.assertEqual(text[slice(*spans[0])], "First\nalpha beta")
        self.assertEqual(text[slice(*spans[1])], "Second\ngamma delta")

    def test_late_chunk_token_alignment_ignores_special_token_offsets(self) -> None:
        offsets = [(0, 0), (0, 5), (6, 10), (0, 0)]
        self.assertEqual(token_indexes_for_span(offsets, (0, 10)), [1, 2])
        with self.assertRaisesRegex(ValueError, "did not align"):
            token_indexes_for_span(offsets, (20, 25))


if __name__ == "__main__":
    unittest.main()
