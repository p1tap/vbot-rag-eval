"""Append and validate privacy-safe sealed-holdout access events.

The ledger is a hash chain. It records access metadata, never case content,
private queries, raw outputs, credentials, or personal identity.

Examples:
  python scripts/holdout_ledger.py --initialize --actor-id codex-session-3
  python scripts/holdout_ledger.py --check
  python scripts/holdout_ledger.py --append --actor-id owner-reviewer-01 \
    --actor-type human --event-type access_requested --split release \
    --scope case_content --outcome requested --purpose "Second-pass review"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER = ROOT / "evals" / "v2" / "holdout-access-ledger.jsonl"
SCHEMA_PATH = ROOT / "evals" / "schema" / "holdout-access-event.schema.json"
MANIFEST_PATH = ROOT / "evals" / "v2" / "dataset-manifest.json"


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def event_hash(event: dict) -> str:
    unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
    return sha256_bytes(canonical_bytes(unsigned))


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def validate_events(path: Path = DEFAULT_LEDGER) -> list[dict]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    events = load_events(path)
    if not events:
        raise ValueError("holdout access ledger is empty or missing")
    previous_hash = None
    for expected_sequence, event in enumerate(events):
        errors = sorted(validator.iter_errors(event), key=lambda error: list(error.path))
        if errors:
            location = ".".join(str(part) for part in errors[0].absolute_path)
            raise ValueError(
                f"holdout ledger event {expected_sequence}:{location}: "
                f"{errors[0].message}"
            )
        if event["sequence"] != expected_sequence:
            raise ValueError(f"holdout ledger sequence mismatch at {expected_sequence}")
        if event["previous_event_sha256"] != previous_hash:
            raise ValueError(f"holdout ledger chain mismatch at {expected_sequence}")
        expected_hash = event_hash(event)
        if event["event_sha256"] != expected_hash:
            raise ValueError(f"holdout ledger event hash mismatch at {expected_sequence}")
        previous_hash = expected_hash
    if events[0]["event_type"] != "ledger_initialized":
        raise ValueError("holdout ledger must begin with ledger_initialized")
    return events


def build_event(
    *,
    previous: dict | None,
    actor_id: str,
    actor_type: str,
    event_type: str,
    split: str | None,
    scope: str,
    purpose: str,
    authorization_reference: str | None,
    case_count: int | None,
    outcome: str,
    timestamp: str | None = None,
) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sequence = 0 if previous is None else previous["sequence"] + 1
    base = {
        "schema_version": "1.0.0",
        "sequence": sequence,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "actor_id": actor_id,
        "actor_type": actor_type,
        "event_type": event_type,
        "dataset_id": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "dataset_fingerprint_sha256": manifest["fingerprint_sha256"],
        "split": split,
        "scope": scope,
        "purpose": purpose,
        "authorization_reference": authorization_reference,
        "case_count": case_count,
        "outcome": outcome,
        "previous_event_sha256": None if previous is None else previous["event_sha256"],
    }
    identity_hash = sha256_bytes(canonical_bytes(base))
    event = {
        "schema_version": base.pop("schema_version"),
        "event_id": f"ha-{sequence:06d}-{identity_hash[:12]}",
        **base,
    }
    event["event_sha256"] = event_hash(event)
    return event


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--append", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--actor-id")
    parser.add_argument("--actor-type", choices=("human", "codex", "service"), default="codex")
    parser.add_argument(
        "--event-type",
        choices=(
            "access_requested",
            "access_granted",
            "access_denied",
            "evaluation_started",
            "evaluation_completed",
            "aggregate_released",
            "contamination_reported",
        ),
    )
    parser.add_argument("--split", choices=("release", "adversarial", "real_user", "reserve"))
    parser.add_argument(
        "--scope",
        choices=("none", "aggregate", "case_ids", "case_content", "per_case_results", "raw_outputs"),
        default="none",
    )
    parser.add_argument("--purpose")
    parser.add_argument("--authorization-reference")
    parser.add_argument("--case-count", type=int)
    parser.add_argument(
        "--outcome",
        choices=("requested", "granted", "denied", "completed", "released", "contaminated"),
    )
    args = parser.parse_args()
    path = args.ledger.resolve()

    if args.check:
        events = validate_events(path)
        print(f"holdout access ledger: OK ({len(events)} events)")
        print(f"chain head: {events[-1]['event_sha256']}")
        return

    if not args.actor_id:
        parser.error("--actor-id is required for initialization or append")
    existing = load_events(path)
    if args.initialize:
        if existing:
            raise ValueError("refusing to initialize a non-empty holdout ledger")
        event = build_event(
            previous=None,
            actor_id=args.actor_id,
            actor_type=args.actor_type,
            event_type="ledger_initialized",
            split=None,
            scope="none",
            purpose=args.purpose or "Initialize sealed-holdout access audit chain.",
            authorization_reference=None,
            case_count=0,
            outcome="initialized",
        )
    else:
        if not existing:
            raise ValueError("initialize the holdout ledger before appending")
        validate_events(path)
        required = {
            "event_type": args.event_type,
            "purpose": args.purpose,
            "outcome": args.outcome,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"append requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
        event = build_event(
            previous=existing[-1],
            actor_id=args.actor_id,
            actor_type=args.actor_type,
            event_type=args.event_type,
            split=args.split,
            scope=args.scope,
            purpose=args.purpose,
            authorization_reference=args.authorization_reference,
            case_count=args.case_count,
            outcome=args.outcome,
        )
    append_event(path, event)
    validate_events(path)
    print(f"appended holdout event {event['event_id']}")
    print(f"event sha256: {event['event_sha256']}")


if __name__ == "__main__":
    main()

