"""SQLite FTS5 BM25 index for large global document corpora."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import re
import sqlite3
import unicodedata

from .base import RetrievalDocument, RetrievalHit
from .bm25 import BM25Retriever


QUERY_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
TITLE_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
STOPWORDS = frozenset(
    "a an and are as at be been by for from had has have he her his in is it "
    "its of on or she that the their there they this to was were which who will "
    "with".split()
)


def fts_terms(text: str) -> list[str]:
    terms = []
    for term in QUERY_TOKEN_RE.findall(text.casefold()):
        if len(term) <= 2 or term in STOPWORDS or term in terms:
            continue
        terms.append(term)
    if not terms:
        terms = QUERY_TOKEN_RE.findall(text.casefold())[:8]
    return terms


def quoted_terms(terms: list[str], operator: str) -> str:
    return f" {operator} ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
    )


def fts_query(text: str) -> str:
    return quoted_terms(fts_terms(text), "OR")


def title_candidates(text: str, max_words: int = 6) -> list[str]:
    tokens = TITLE_TOKEN_RE.findall(text)
    cleaned = [
        token[:-2] if token.casefold().endswith(("'s", "’s")) else token
        for token in tokens
    ]
    candidates = set()
    for start in range(len(cleaned)):
        for width in range(1, min(max_words, len(cleaned) - start) + 1):
            page_id = "_".join(cleaned[start : start + width]).strip("_")
            if page_id:
                candidates.add(unicodedata.normalize("NFC", page_id))
    return sorted(candidates)


class SQLiteFTSBM25Index:
    name = "bm25_fts5"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> "SQLiteFTSBM25Index":
        self._connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        return self.rank_expression(fts_query(query), k)

    def rank_expression(self, expression: str, k: int) -> list[RetrievalHit]:
        if k < 1:
            raise ValueError("retrieval k must be at least one")
        if not expression:
            return []
        if self._connection is not None:
            rows = self._rank(self._connection, expression, k)
        else:
            with closing(
                sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            ) as connection:
                rows = self._rank(connection, expression, k)
        return [
            RetrievalHit(
                RetrievalDocument(page_id, title, text),
                -float(score),
                rank,
            )
            for rank, (page_id, title, text, score) in enumerate(rows, start=1)
        ]

    def lookup_exact(self, page_ids: list[str]) -> list[RetrievalDocument]:
        if not page_ids:
            return []
        if self._connection is not None:
            return self._lookup_exact(self._connection, page_ids)
        with closing(
            sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        ) as connection:
            return self._lookup_exact(connection, page_ids)

    @staticmethod
    def _rank(
        connection: sqlite3.Connection, expression: str, k: int
    ) -> list[tuple[str, str, str, float]]:
        rows = connection.execute(
            """SELECT page_id, title, text, bm25(pages, 0.0, 3.0, 1.0) AS score
            FROM pages WHERE pages MATCH ? ORDER BY score, page_id LIMIT ?""",
            (expression, k),
        ).fetchall()
        return rows

    @staticmethod
    def _lookup_exact(
        connection: sqlite3.Connection, page_ids: list[str]
    ) -> list[RetrievalDocument]:
        documents = []
        for start in range(0, len(page_ids), 500):
            batch = page_ids[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                f"""SELECT pages.page_id, pages.title, pages.text
                FROM page_lookup JOIN pages ON pages.rowid = page_lookup.fts_rowid
                WHERE page_lookup.page_id IN ({placeholders})""",
                batch,
            ).fetchall()
            documents.extend(
                RetrievalDocument(page_id, title, text)
                for page_id, title, text in rows
            )
        return documents


class TwoStageFeverBM25Index:
    """Generate global candidates cheaply, then rerank them with local BM25."""

    name = "bm25_fts5_two_stage"

    def __init__(self, path: Path, *, candidate_limit: int = 100) -> None:
        self.fts = SQLiteFTSBM25Index(path)
        self.candidate_limit = candidate_limit
        self.reranker = BM25Retriever()

    def __enter__(self) -> "TwoStageFeverBM25Index":
        self.fts.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        self.fts.close()

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        terms = fts_terms(query)
        if not terms:
            return []
        candidates = {
            document.id: document
            for document in self.fts.lookup_exact(title_candidates(query))
        }
        focused = sorted(terms, key=lambda term: (-len(term), term))[:8]
        strict_expression = quoted_terms(focused, "AND")
        for hit in self.fts.rank_expression(strict_expression, self.candidate_limit):
            candidates[hit.document.id] = hit.document
        if len(candidates) < k and len(focused) > 2:
            relaxed_expressions = (
                quoted_terms(focused[:position] + focused[position + 1 :], "AND")
                for position in range(len(focused))
            )
        else:
            relaxed_expressions = ()
        for expression in relaxed_expressions:
            for hit in self.fts.rank_expression(expression, self.candidate_limit):
                candidates[hit.document.id] = hit.document
            if len(candidates) >= k:
                break
        return self.reranker.rank(query, list(candidates.values()), k)
