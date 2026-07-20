"""Fail closed on high-confidence credential material in the public corpus."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
PATTERNS = {
    "openai_or_openrouter_key": re.compile(r"\bsk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{16,}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
    ),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*"
        r"[\"']?([A-Za-z0-9_./+=-]{16,})"
    ),
}
PLACEHOLDER_MARKERS = {
    "changeme",
    "example",
    "placeholder",
    "redacted",
    "replace",
    "your_",
    "your-",
}


def corpus_paths(root: Path = CORPUS) -> list[Path]:
    suffixes = {".md", ".json", ".jsonl", ".yaml", ".yml"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def scan(paths: list[Path]) -> list[dict]:
    findings = []
    for path in paths:
        try:
            display_path = path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            display_path = path.name
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    value = match.group(1) if match.lastindex else match.group(0)
                    lowered = value.casefold()
                    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
                        continue
                    findings.append(
                        {
                            "path": display_path,
                            "line": line_number,
                            "pattern": label,
                            "match_sha256_only": __import__("hashlib")
                            .sha256(value.encode())
                            .hexdigest(),
                        }
                    )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    paths = corpus_paths()
    findings = scan(paths)
    report = {
        "schema_version": "1.0.0",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "public_corpus_high_confidence_credential_scan",
        "scanner": "deterministic_regex_no_external_service",
        "scanned_file_count": len(paths),
        "finding_count": len(findings),
        "status": "passed" if not findings else "failed",
        "limitations": [
            "Pattern scanning cannot prove that prose contains no sensitive information.",
            "Only high-confidence credential formats and secret assignments are detected.",
        ],
        "findings": findings,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
