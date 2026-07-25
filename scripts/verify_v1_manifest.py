"""Verify the frozen V1 manifest and its deterministic reproduction evidence."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "baselines" / "v1" / "run-manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def file_matches_checkout_hash(path: Path, expected: str) -> bool:
    """Accept the frozen Windows checkout hash on LF-only CI runners.

    The original V1 reproduction artifacts were captured with CRLF line
    endings. Git materializes the same text as LF on Linux Actions runners,
    so compare both the exact bytes and an explicit CRLF rendering.
    """
    value = path.read_bytes()
    if sha256_bytes(value) == expected:
        return True
    normalized_lf = value.replace(b"\r\n", b"\n")
    windows_checkout = normalized_lf.replace(b"\n", b"\r\n")
    return sha256_bytes(windows_checkout) == expected


def canonical_json_sha256(obj: dict) -> str:
    value = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(value)


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    accepted = manifest["accepted_run"]
    commit = accepted["source"]["commit"]
    errors: list[str] = []

    tree = subprocess.check_output(
        ["git", "show", "-s", "--format=%T", commit], cwd=ROOT, text=True
    ).strip()
    if tree != accepted["source"]["tree"]:
        errors.append(f"accepted tree mismatch: {tree}")

    for path, expected in accepted["inputs"].items():
        actual = sha256_bytes(git_blob(commit, path))
        if actual != expected:
            errors.append(f"accepted input hash mismatch: {path}")
        current = ROOT / path
        if path != "config.py" and sha256_file(current) != expected:
            errors.append(f"frozen current input changed: {path}")

    report_meta = accepted["report"]
    accepted_report_bytes = git_blob(commit, report_meta["source_path"])
    if sha256_bytes(accepted_report_bytes) != report_meta["sha256_git_blob"]:
        errors.append("accepted report Git-blob hash mismatch")
    archived_report_bytes = (ROOT / report_meta["archived_path"]).read_bytes()
    if not file_matches_checkout_hash(
        ROOT / report_meta["archived_path"],
        report_meta["sha256_checkout_before_phase0_provenance"],
    ):
        errors.append("archived accepted checkout report hash mismatch")
    accepted_report = json.loads(accepted_report_bytes)
    if canonical_json_sha256(accepted_report) != report_meta["sha256_canonical_json"]:
        errors.append("accepted report canonical hash mismatch")
    if canonical_json_sha256(json.loads(archived_report_bytes)) != report_meta[
        "sha256_canonical_json"
    ]:
        errors.append("archived accepted report semantic hash mismatch")

    current_report = json.loads(
        (ROOT / report_meta["source_path"]).read_text(encoding="utf-8")
    )
    provenance = current_report.get("provenance")
    if provenance and provenance.get("capture") == "retrospective_phase0":
        current_core = dict(current_report)
        current_core.pop("provenance")
        if canonical_json_sha256(current_core) != report_meta["sha256_canonical_json"]:
            errors.append("accepted report evidence changed beyond Phase 0 provenance")
        if provenance.get("run_manifest") != "baselines/v1/run-manifest.json":
            errors.append("accepted report is missing its V1 run-manifest link")

    reproduction = manifest["deterministic_reproduction"]
    for name in ("environment", "dependency_snapshot", "report"):
        artifact = reproduction[name]
        if not file_matches_checkout_hash(
            ROOT / artifact["path"], artifact["sha256"]
        ):
            errors.append(f"reproduction artifact hash mismatch: {artifact['path']}")
    reproduced_report = json.loads(
        (ROOT / reproduction["report"]["path"]).read_text(encoding="utf-8")
    )
    for metric, expected in reproduction["summary"].items():
        if reproduced_report["summary"].get(metric) != expected:
            errors.append(f"reproduced metric mismatch: {metric}")
    misses = [
        item["id"]
        for item in reproduced_report["items"]
        if item["answerable"] and not item["hit"]
    ]
    if misses != manifest["known_failures"]["retrieval"]:
        errors.append(f"reproduced retrieval failures changed: {misses}")

    sys.path.insert(0, str(ROOT))
    import config  # noqa: E402

    expected_revision = accepted["models"]["embedding"]["resolved_revision"]
    if config.EMBED_MODEL_REVISION != expected_revision:
        errors.append("configured embedding revision differs from V1 manifest")

    if errors:
        print("V1 manifest verification: FAIL")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("V1 manifest verification: OK")
    print(f"  accepted commit: {commit}")
    print("  clean CPU reproduction: recall@k=0.974, MRR=0.844")


if __name__ == "__main__":
    main()
