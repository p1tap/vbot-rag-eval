"""Freeze completed localhost review queues into scrubbed V2 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.provenance import sha256_file  # noqa: E402

DEFAULT_SERVER = "http://localhost:3000"
DEFAULT_CASES = ROOT / "evals" / "v2" / "reviewed" / "vbot-human-reviewed-100.jsonl"
DEFAULT_LABELS = ROOT / "evals" / "v2" / "judge-calibration" / "human-labels.jsonl"
DEFAULT_MANIFEST = ROOT / "evals" / "v2" / "review-freeze-manifest.json"
REVIEWER_ID = "owner-reviewer-1"
DATASET_VERSION = "2.0.0-dev.2"

ACTION_REASONS = {
    "abstain_absent": {"absent_from_corpus", "subjective"},
    "clarify_ambiguous": {"ambiguous", "malformed"},
    "abstain_conflict": {"conflicting_sources"},
    "abstain_obsolete": {"obsolete_only"},
    "abstain_unauthorized": {"unauthorized"},
    "reject_injection": {"prompt_injection"},
}
DEFAULT_ACTION_REASON = {
    "abstain_absent": "absent_from_corpus",
    "clarify_ambiguous": "ambiguous",
    "abstain_conflict": "conflicting_sources",
    "abstain_obsolete": "obsolete_only",
    "abstain_unauthorized": "unauthorized",
    "reject_injection": "prompt_injection",
}


def _json_get(url: str) -> dict:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - localhost operator input
        return json.loads(response.read().decode("utf-8"))


def _jsonl_get(url: str) -> list[dict]:
    with urlopen(url, timeout=30) as response:  # noqa: S310 - localhost operator input
        text = response.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _jsonl_text(rows: list[dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_timestamp(value: str) -> str:
    cleaned = value.strip().replace(" ", "T", 1)
    if cleaned.endswith("Z"):
        parsed = datetime.fromisoformat(cleaned[:-1] + "+00:00")
    else:
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalized_unanswerable(action: str, current: dict | None) -> dict | None:
    """Keep valid review metadata while making the approved action authoritative."""
    if action == "answer":
        return None
    if action not in ACTION_REASONS:
        raise ValueError(f"unsupported reviewed answer action: {action}")
    value = dict(current or {})
    if value.get("reason") not in ACTION_REASONS[action]:
        value["reason"] = DEFAULT_ACTION_REASON[action]
    value.setdefault("notes", "")
    return value


def _validate_rows(rows: list[dict], schema_path: Path, label: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for row in rows:
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.path))
        if errors:
            path = ".".join(str(item) for item in errors[0].path)
            raise ValueError(
                f"invalid {label} {row.get('id', row.get('case_id'))} at {path}: "
                f"{errors[0].message}"
            )


def _reviewed_case(source: dict, decision: dict, base: dict | None) -> dict:
    payload = decision["payload"]
    answerable = source["answerability"] == "answerable"
    required_claims = [
        {
            "id": claim["id"],
            "text": claim["text"].strip(),
            "evidence_ids": claim["evidenceIds"],
            "status": "atomic_verified",
        }
        for claim in payload["atomicClaims"]
    ]
    reviewed_at = _normalize_timestamp(decision["submittedAt"])
    notes = ["Approved in the source-grounded localhost review queue."]
    if payload.get("notes", "").strip():
        notes.append(payload["notes"].strip())

    if base is not None:
        case = dict(base)
        case.update(
            {
                "dataset_version": DATASET_VERSION,
                "lane": payload["lane"],
                "answer_action": payload["answerAction"],
                "reference_answer": (
                    " ".join(claim["text"] for claim in required_claims)
                    if answerable
                    else base["reference_answer"]
                ),
                "required_claims": required_claims,
                "acceptable_evidence": payload["acceptableEvidenceIds"],
                "distractor_evidence": payload["distractorEvidenceIds"],
                "evidence_granularity": "span" if answerable else "none",
                "intent_family_id": payload["intentFamilyId"],
                "evidence_cluster_id": payload["evidenceClusterId"],
                "severity": payload["severity"],
                "unanswerable": _normalized_unanswerable(
                    payload["answerAction"], base.get("unanswerable")
                ),
                "tags": sorted(set(base["tags"] + ["human_reviewed_seed"])),
                "review": {
                    "status": "approved",
                    "reviewers": [REVIEWER_ID],
                    "reviewed_at": reviewed_at,
                    "notes": notes + ["Visible V1 migration upgraded to atomic span labels."],
                },
            }
        )
        if answerable:
            headings = {
                evidence_id.rsplit("::span-", 1)[0]
                for evidence_id in payload["acceptableEvidenceIds"]
            }
            case["reasoning_hops"] = max(1, len(headings))
        else:
            case["reasoning_hops"] = 0
        return case

    candidate = source["candidate"]
    return {
        "schema_version": "2.0.0-dev",
        "id": candidate["id"],
        "dataset_version": DATASET_VERSION,
        "split": "dev",
        "lane": payload["lane"],
        "question": candidate["question"],
        "language": "en",
        "answerability": candidate["answerability"],
        "answer_action": payload["answerAction"],
        "reference_answer": (
            " ".join(claim["text"] for claim in required_claims)
            if answerable
            else candidate["reference_answer"]
        ),
        "required_claims": required_claims,
        "acceptable_evidence": payload["acceptableEvidenceIds"],
        "distractor_evidence": payload["distractorEvidenceIds"],
        "evidence_granularity": "span" if answerable else "none",
        "reasoning_hops": candidate["reasoning_hops"] if answerable else 0,
        "intent_family_id": payload["intentFamilyId"],
        "evidence_cluster_id": payload["evidenceClusterId"],
        "conversation_id": candidate["conversation_id"],
        "turn_index": candidate["turn_index"],
        "as_of": candidate["as_of"],
        "severity": payload["severity"],
        "traffic_weight": None,
        "tags": sorted(set(candidate["tags"] + ["human_reviewed_seed"])),
        "authoring": {
            "method": "model_proposed_human_verified",
            "identity_disclosure": "private_pseudonym",
            "independent_of_corpus_authors": None,
            "model_assistance": "candidate_generation",
        },
        "review": {
            "status": "approved",
            "reviewers": [REVIEWER_ID],
            "reviewed_at": reviewed_at,
            "notes": notes + ["Model-authored candidate verified against exact source spans."],
        },
        "unanswerable": _normalized_unanswerable(
            payload["answerAction"], candidate["unanswerable"]
        ),
    }


def freeze_reviewed_cases(server: str) -> tuple[list[dict], dict]:
    app_source = json.loads(
        (ROOT.parent / "rag-review-app" / "data" / "review-source.json").read_text(
            encoding="utf-8"
        )
    )
    source_by_id = {row["case_id"]: row for row in app_source["cases"]}
    migrated = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "evals" / "v2" / "dev" / "v1-migrated.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    candidates_value = json.loads(
        (ROOT / "evals" / "v2" / "candidates" / "phase1-new-candidates.json")
        .read_text(encoding="utf-8")
    )
    candidate_by_id = {row["id"]: row for row in candidates_value["cases"]}

    state = _json_get(f"{server}/api/review")
    queue = state["queue"]
    if len(queue) != 100 or state["progress"].get("approved") != 100:
        raise ValueError("the Vbot review queue is not exactly 100/100 approved")
    rows = []
    versions = {}
    for item in queue:
        selected = _json_get(f"{server}/api/review?caseId={quote(item['id'])}")["selected"]
        decision = selected.get("decision")
        if not decision or decision.get("status") != "approved":
            raise ValueError(f"{item['id']} is not approved")
        if decision.get("qualityAttested") is not True:
            raise ValueError(f"{item['id']} is missing quality attestation")
        source = source_by_id.get(item["id"])
        if not source or selected["source"].get("source_hash") != source["source_hash"]:
            raise ValueError(f"{item['id']} source hash does not match the frozen review seed")
        source_view = dict(source)
        if item["id"].startswith("v2-"):
            source_view["candidate"] = candidate_by_id[item["id"]]
        rows.append(_reviewed_case(source_view, decision, migrated.get(item["id"])))
        versions[item["id"]] = decision["version"]
    if {row["id"] for row in rows} != set(source_by_id):
        raise ValueError("reviewed case IDs do not exactly match the frozen review source")
    return rows, {
        "case_count": len(rows),
        "approved_count": len(rows),
        "decision_versions_sha256": _canonical_sha256(versions),
        "source_dataset_fingerprint_sha256": app_source["metadata"][
            "dataset_fingerprint_sha256"
        ],
    }


def freeze_calibration_labels(server: str) -> tuple[list[dict], dict]:
    manifest = json.loads(
        (ROOT / "evals" / "v2" / "judge-calibration" / "manifest.json")
        .read_text(encoding="utf-8")
    )
    queue = [
        json.loads(line)
        for line in (ROOT / "evals" / "v2" / "judge-calibration" / "queue.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    exported = _jsonl_get(f"{server}/api/judge-calibration/export")
    task_by_id = {task["task_id"]: task for task in queue}
    export_by_id = {row["task_id"]: row for row in exported}
    if len(exported) != 175 or set(export_by_id) != set(task_by_id):
        raise ValueError("calibration export does not exactly cover the 175 frozen tasks")
    by_case: dict[str, dict] = {}
    task_versions = {}
    decision_counts = Counter()
    for task in queue:
        row = export_by_id[task["task_id"]]
        if row["queue_sha256"] != manifest["queue"]["sha256"]:
            raise ValueError(f"{task['task_id']} targets a different queue")
        if row["answer_sha256"] != task["answer_sha256"]:
            raise ValueError(f"{task['task_id']} targets a different answer")
        task_versions[task["task_id"]] = row["version"]
        decision_counts[f"{task['task_type']}:{row['decision']}"] += 1
        record = by_case.setdefault(
            task["case_id"],
            {
                "schema_version": "1.0.0",
                "case_id": task["case_id"],
                "run_id": manifest["run_id"],
                "answer_sha256": task["answer_sha256"],
                "claim_labels": [],
                "required_claim_labels": [],
                "reviewer_id": REVIEWER_ID,
                "reviewed_at": _normalize_timestamp(row["submitted_at"]),
                "status": "final",
                "notes": "",
            },
        )
        if record["answer_sha256"] != task["answer_sha256"]:
            raise ValueError(f"{task['case_id']} has inconsistent answer hashes")
        record["reviewed_at"] = max(
            record["reviewed_at"], _normalize_timestamp(row["submitted_at"])
        )
        if row.get("note", "").strip():
            record["notes"] = " ".join(
                value for value in (record["notes"], row["note"].strip()) if value
            )
        if task["task_type"] == "generated_claim_support":
            evidence = row.get("evidence_spans", [])
            sources = {item["citation_id"]: item["text"] for item in task["cited_sources"]}
            if row["decision"] == "unsupported" and evidence:
                raise ValueError(f"{task['task_id']} unsupported label retains evidence")
            if row["decision"] != "unsupported" and not evidence:
                raise ValueError(f"{task['task_id']} needs exact evidence")
            for span in evidence:
                if span["citation_id"] not in sources or span["quote"] not in sources[span["citation_id"]]:
                    raise ValueError(f"{task['task_id']} contains a non-exact evidence span")
            record["claim_labels"].append(
                {
                    "claim_id": task["generated_claim"]["id"],
                    "status": row["decision"],
                    "evidence": evidence,
                    "rationale": row["rationale"],
                }
            )
        else:
            linked = row.get("linked_claim_ids", [])
            valid = {item["id"] for item in task["generated_claims"]}
            if not set(linked).issubset(valid):
                raise ValueError(f"{task['task_id']} links an unknown generated claim")
            if row["decision"] == "missing" and linked:
                raise ValueError(f"{task['task_id']} missing label retains links")
            if row["decision"] != "missing" and not linked:
                raise ValueError(f"{task['task_id']} needs a generated-claim link")
            record["required_claim_labels"].append(
                {
                    "claim_id": task["required_claim"]["id"],
                    "status": row["decision"],
                    "generated_claim_ids": linked,
                    "rationale": row["rationale"],
                }
            )
    rows = [by_case[case_id] for case_id in manifest["answered_case_ids"]]
    return rows, {
        "task_count": len(exported),
        "case_count": len(rows),
        "decision_counts": dict(sorted(decision_counts.items())),
        "task_versions_sha256": _canonical_sha256(task_versions),
        "queue_sha256": manifest["queue"]["sha256"],
        "run_id": manifest["run_id"],
    }


def _write_or_check(path: Path, content: str, check: bool) -> None:
    encoded = content.encode("utf-8")
    if check:
        if not path.exists() or path.read_bytes() != encoded:
            raise ValueError(f"{path} is not current")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    cases, case_summary = freeze_reviewed_cases(args.server.rstrip("/"))
    labels, label_summary = freeze_calibration_labels(args.server.rstrip("/"))
    _validate_rows(cases, ROOT / "evals" / "schema" / "case-v2.schema.json", "case")
    _validate_rows(
        labels,
        ROOT / "evals" / "schema" / "judge-calibration-label.schema.json",
        "calibration label",
    )
    cases_text = _jsonl_text(cases)
    labels_text = _jsonl_text(labels)
    manifest = {
        "schema_version": "1.0.0",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "reviewer_id": REVIEWER_ID,
            "reviewer_identity": "private; direct identifiers omitted",
            "review_kind": "owner_human_review",
            "source": "localhost D1-backed review application",
        },
        "reviewed_cases": {
            "path": DEFAULT_CASES.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(cases_text.encode("utf-8")).hexdigest(),
            **case_summary,
        },
        "judge_calibration": {
            "path": DEFAULT_LABELS.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(labels_text.encode("utf-8")).hexdigest(),
            **label_summary,
        },
    }
    if args.check and DEFAULT_MANIFEST.exists():
        existing = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        manifest["frozen_at_utc"] = existing.get("frozen_at_utc", manifest["frozen_at_utc"])
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _write_or_check(DEFAULT_CASES, cases_text, args.check)
    _write_or_check(DEFAULT_LABELS, labels_text, args.check)
    _write_or_check(DEFAULT_MANIFEST, manifest_text, args.check)
    if not args.check:
        # Re-read hashes after atomic publication so the manifest cannot point
        # at content different from the emitted artifacts.
        assert sha256_file(DEFAULT_CASES) == manifest["reviewed_cases"]["sha256"]
        assert sha256_file(DEFAULT_LABELS) == manifest["judge_calibration"]["sha256"]
    print(json.dumps({"reviewed_cases": case_summary, "judge_calibration": label_summary}, indent=2))


if __name__ == "__main__":
    main()
