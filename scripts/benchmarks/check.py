from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.benchmarks import FeverAdapter, HotpotQAAdapter, NaturalQuestionsAdapter  # noqa: E402
from scripts.benchmarks.prepare import (  # noqa: E402
    load_records,
    load_registry,
    validate_normalized,
)


def main() -> None:
    registry = load_registry()
    target = sum(entry["target_cases"] for entry in registry["benchmarks"])
    if target < 10_000:
        raise ValueError(f"public benchmark target fell below 10,000: {target}")

    fixture = load_records(
        ROOT / "benchmarks" / "fixtures" / "hotpotqa-dev-sample.json"
    )
    normalized = HotpotQAAdapter().normalize_many(fixture, "dev_distractor")
    validation = validate_normalized(normalized)
    if validation["normalized_cases"] != 2:
        raise ValueError("HotpotQA adapter fixture count drifted")
    nq_fixture = load_records(
        ROOT / "benchmarks" / "fixtures" / "natural-questions-dev-sample.json"
    )
    nq_normalized = NaturalQuestionsAdapter().normalize_many(nq_fixture, "dev")
    nq_validation = validate_normalized(nq_normalized)
    if nq_validation["normalized_cases"] != 2:
        raise ValueError("Natural Questions adapter fixture count drifted")
    fever_fixture = load_records(
        ROOT / "benchmarks" / "fixtures" / "fever-dev-sample.json"
    )
    fever_normalized = FeverAdapter().normalize_many(fever_fixture, "dev")
    fever_validation = validate_normalized(fever_normalized)
    if fever_validation["normalized_cases"] != 2:
        raise ValueError("FEVER adapter fixture count drifted")
    print(
        "public benchmark contracts valid: "
        f"registered={len(registry['benchmarks'])} target={target} "
        f"fixture_evidence="
        f"{validation['evidence_references_resolved'] + nq_validation['evidence_references_resolved'] + fever_validation['evidence_references_resolved']}"
    )


if __name__ == "__main__":
    main()
