"""Build a separate long-context late-chunked Vbot retrieval index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ingest import load_chunks  # noqa: E402
from rag.late_chunking import LateChunkingEmbedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "indexes" / "vbot-late-chunking",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--strategy",
        choices=("late_chunking", "pre_chunked_control"),
        default="late_chunking",
    )
    args = parser.parse_args()
    chunks = load_chunks()
    embedder = LateChunkingEmbedder(device=args.device)
    if args.strategy == "late_chunking":
        vectors = embedder.embed_chunks(chunks)
    else:
        vectors = embedder.embed_passages(
            [f"{chunk['heading']}. {chunk['text']}" for chunk in chunks]
        )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", vectors)
    (args.output / "chunks.jsonl").write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in chunks),
        encoding="utf-8",
    )
    metadata = {
        "embed_model": embedder.model_name,
        "embed_model_revision": embedder.model_revision,
        "remote_code_revision": embedder.code_revision,
        "index_strategy": args.strategy,
        "query_encoder": "jina_v2_small_symmetric",
        "document_view": "ordered_heading_and_chunk_text",
        "max_tokens_per_document": embedder.max_length,
        "n_chunks": len(chunks),
        "status": "experimental_not_promoted",
        "limitations": (
            [
                "Stable overlapping chunk windows are assembled into the document view, so overlap text may be repeated.",
                "This index requires separate query embeddings from the same pinned model.",
            ]
            if args.strategy == "late_chunking"
            else [
                "This control isolates encoder-family effects from late chunk pooling.",
                "Passages are independently encoded and truncated to 512 tokens.",
            ]
        ),
    }
    (args.output / "meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"long-context experiment: strategy={args.strategy} "
        f"chunks={len(chunks)} output={args.output}"
    )


if __name__ == "__main__":
    main()
