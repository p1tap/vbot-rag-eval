"""Verify hash-locked authentic PROMOTE and REJECT decision examples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402

MANIFEST = ROOT / "docs" / "promotion-evidence.json"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decisions = set()
    for example in manifest["examples"]:
        expected = example["expected_decision"]
        decisions.add(expected)
        files = [example["decision_artifact"], *example["inputs"]]
        for artifact in files:
            path = (ROOT / artifact["path"]).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
                raise ValueError(f"missing or unsafe promotion artifact: {artifact['path']}")
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"promotion artifact drifted: {artifact['path']}")
        decision = json.loads(
            (ROOT / example["decision_artifact"]["path"]).read_text(encoding="utf-8")
        )
        if decision.get("decision") != expected:
            raise ValueError(f"decision mismatch for {example['id']}")
        if expected == "REJECT" and not decision.get("rejection_reasons"):
            raise ValueError(f"rejected example lacks reasons: {example['id']}")
    if decisions != {"PROMOTE", "REJECT"}:
        raise ValueError("promotion evidence must contain PROMOTE and REJECT")
    print("promotion examples verified: PROMOTE=1 REJECT=1")


if __name__ == "__main__":
    main()
