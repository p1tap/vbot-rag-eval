"""Build a separate provenance-contextualized Vbot retrieval index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.ingest import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "indexes" / "vbot-provenance-context",
    )
    args = parser.parse_args()
    chunks = build_index(
        index_dir=args.output,
        context_mode="provenance_header",
    )
    print(f"contextual index: chunks={len(chunks)} output={args.output}")


if __name__ == "__main__":
    main()
