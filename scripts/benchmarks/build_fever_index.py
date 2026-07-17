"""Build a durable global FEVER Wikipedia BM25 index from the pinned ZIP."""

from __future__ import annotations

import argparse
from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3
import sys
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.audit_suite import file_sha256  # noqa: E402


DEFAULT_SOURCE = (
    ROOT / "artifacts" / "benchmarks" / "fever" / "raw" / "wiki-pages.zip"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "benchmarks" / "fever" / "indexes" / "wikipedia-bm25.sqlite"
)


def build(source: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        "CREATE VIRTUAL TABLE pages USING fts5("
        "page_id UNINDEXED, title, text, tokenize='porter unicode61')"
    )
    connection.execute(
        "CREATE TABLE page_lookup ("
        "page_id TEXT PRIMARY KEY, fts_rowid INTEGER NOT NULL UNIQUE) WITHOUT ROWID"
    )
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    page_count = 0
    member_count = 0
    batch: list[tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(source) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.startswith("wiki-pages/wiki-") and name.endswith(".jsonl")
            )
            for member_name in members:
                member_count += 1
                with closing(archive.open(member_name)) as raw_handle:
                    with io.TextIOWrapper(raw_handle, encoding="utf-8") as handle:
                        for line in handle:
                            record = json.loads(line)
                            page_id = unicodedata.normalize("NFC", str(record["id"]))
                            text = str(record["text"]).strip()
                            if not page_id or not text:
                                continue
                            title = page_id.replace("_", " ")
                            batch.append((page_id, title, text))
                            if len(batch) >= 5000:
                                indexed_batch = [
                                    (page_count + offset, page_id, title, text)
                                    for offset, (page_id, title, text) in enumerate(
                                        batch, start=1
                                    )
                                ]
                                connection.executemany(
                                    "INSERT INTO pages(rowid, page_id, title, text) "
                                    "VALUES (?, ?, ?, ?)",
                                    indexed_batch,
                                )
                                connection.executemany(
                                    "INSERT INTO page_lookup(page_id, fts_rowid) "
                                    "VALUES (?, ?)",
                                    [
                                        (page_id, rowid)
                                        for rowid, page_id, _, _ in indexed_batch
                                    ],
                                )
                                page_count += len(batch)
                                batch.clear()
                                if page_count % 100000 == 0:
                                    connection.commit()
                                    print(f"indexed FEVER pages: {page_count}", flush=True)
            if batch:
                indexed_batch = [
                    (page_count + offset, page_id, title, text)
                    for offset, (page_id, title, text) in enumerate(batch, start=1)
                ]
                connection.executemany(
                    "INSERT INTO pages(rowid, page_id, title, text) VALUES (?, ?, ?, ?)",
                    indexed_batch,
                )
                connection.executemany(
                    "INSERT INTO page_lookup(page_id, fts_rowid) VALUES (?, ?)",
                    [(page_id, rowid) for rowid, page_id, _, _ in indexed_batch],
                )
                page_count += len(batch)
        source_sha256 = file_sha256(source)
        metadata = {
            "builder_version": "1.0.0",
            "source_path": source.relative_to(ROOT).as_posix(),
            "source_sha256": source_sha256,
            "page_count": page_count,
            "archive_member_count": member_count,
            "fts_tokenizer": "porter unicode61",
            "status": "complete",
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        temporary.replace(output)
        return metadata
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    metadata = build(args.source, args.output)
    print(
        "FEVER global BM25 index complete: "
        f"pages={metadata['page_count']} members={metadata['archive_member_count']}"
    )


if __name__ == "__main__":
    main()
