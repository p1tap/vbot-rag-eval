"""Build stable, line-addressable semantic blocks for human evidence review.

IDs derive from document ID, the current heading key, and normalized block
content. Unrelated line-offset edits therefore do not rename an unchanged
block. Duplicate identical blocks under one heading receive a deterministic
occurrence suffix.

Run:
  python scripts/build_evidence_catalog.py --write
  python scripts/build_evidence_catalog.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_MANIFEST = ROOT / "corpus" / "manifest.json"
DEFAULT_OUT = ROOT / "corpus" / "evidence-catalog-v1.jsonl"
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|[0-9]+\.\s+)")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def normalize(lines: list[str]) -> str:
    return "\n".join(line.rstrip() for line in lines).strip()


def block_type(text: str) -> str:
    lines = text.splitlines()
    first = lines[0].lstrip()
    if FENCE_RE.match(first):
        return "code"
    if first.startswith(">"):
        return "quote"
    if LIST_RE.match(first):
        return "list"
    if len(lines) >= 2 and "|" in lines[0] and re.search(r"\|?\s*:?-{3,}", lines[1]):
        return "table"
    return "prose"


def make_record(
    *,
    document_id: str,
    document_sha256: str,
    heading_path: list[str],
    heading_key: str,
    kind: str,
    start_line: int,
    end_line: int,
    text: str,
    occurrences: Counter,
) -> dict:
    content_sha256 = sha256_bytes(text.encode("utf-8"))
    base_id = f"{heading_key}::span-{content_sha256[:16]}"
    occurrences[base_id] += 1
    suffix = "" if occurrences[base_id] == 1 else f"-{occurrences[base_id]}"
    return {
        "schema_version": "1.0.0",
        "evidence_id": f"{base_id}{suffix}",
        "document_id": document_id,
        "document_sha256": document_sha256,
        "heading_path": list(heading_path),
        "heading_key": heading_key,
        "block_type": kind,
        "start_line": start_line,
        "end_line": end_line,
        "text": text,
        "content_sha256": content_sha256,
        "word_count": len(text.split()),
    }


def parse_document(document: dict) -> list[dict]:
    path = ROOT / document["path"]
    lines = path.read_text(encoding="utf-8").splitlines()
    document_id = document["document_id"]
    document_sha256 = document["content_sha256"]
    path_stack: list[str] = []
    current_heading = "intro"
    current_heading_path = ["intro"]
    current_heading_key = f"{document_id}::intro"
    records: list[dict] = []
    occurrences: Counter = Counter()
    buffer: list[str] = []
    buffer_start = 0
    in_fence = False

    def flush(end_line: int) -> None:
        nonlocal buffer, buffer_start
        text = normalize(buffer)
        if text and not text.startswith("<!--"):
            records.append(
                make_record(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    heading_path=current_heading_path,
                    heading_key=current_heading_key,
                    kind=block_type(text),
                    start_line=buffer_start,
                    end_line=end_line,
                    text=text,
                    occurrences=occurrences,
                )
            )
        buffer = []
        buffer_start = 0

    for line_number, line in enumerate(lines, 1):
        fence = FENCE_RE.match(line)
        if in_fence:
            buffer.append(line)
            if fence:
                in_fence = False
                flush(line_number)
            continue
        if fence:
            flush(line_number - 1)
            buffer_start = line_number
            buffer = [line]
            in_fence = True
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush(line_number - 1)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            path_stack = path_stack[: level - 1]
            path_stack.append(title)
            current_heading = title
            current_heading_path = list(path_stack)
            current_heading_key = f"{document_id}::{slug(current_heading)}"
            records.append(
                make_record(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    heading_path=current_heading_path,
                    heading_key=current_heading_key,
                    kind="heading",
                    start_line=line_number,
                    end_line=line_number,
                    text=title,
                    occurrences=occurrences,
                )
            )
            continue

        if not line.strip():
            flush(line_number - 1)
            continue
        if LIST_RE.match(line):
            # Each top-level list item is its own evidence span. Continuation
            # lines remain attached until the next item.
            flush(line_number - 1)
            buffer_start = line_number
            buffer = [line]
            continue
        if buffer and LIST_RE.match(buffer[0]) and not line.startswith((" ", "\t")):
            flush(line_number - 1)
        if not buffer:
            buffer_start = line_number
        buffer.append(line)
    flush(len(lines))
    return records


def build_records() -> list[dict]:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    records = []
    for document in sorted(manifest["documents"], key=lambda item: item["document_id"]):
        if document["status"] == "active" and document["ingestion"]["allowed"]:
            records.extend(parse_document(document))
    ids = [record["evidence_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("evidence IDs are not unique")
    return records


def render(records: list[dict]) -> bytes:
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    ) + "\n"
    return text.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = build_records()
    expected = render(records)
    output = args.out.resolve()
    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(expected)
        action = "wrote"
    else:
        if not output.is_file() or output.read_bytes() != expected:
            raise ValueError(
                "evidence catalog is missing or stale; regenerate with "
                "python scripts/build_evidence_catalog.py --write"
            )
        action = "verified"
    counts = Counter(record["block_type"] for record in records)
    print(f"{action} {len(records)} evidence spans")
    print(f"types: {dict(sorted(counts.items()))}")
    print(f"output sha256: {sha256_bytes(expected)}")


if __name__ == "__main__":
    main()
