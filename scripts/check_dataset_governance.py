"""Enforce local and cross-version V2 dataset governance rules."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "evals" / "v2" / "dataset-manifest.json"
INVENTORY_PATH = ROOT / "evals" / "v2" / "review-inventory.json"
SEALED_SPLITS = {"release", "adversarial", "real_user", "reserve"}
SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\.(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)(?:-(?P<pre>[0-9A-Za-z.-]+))?$"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def semver_key(value: str) -> tuple[int, int, int, int, str]:
    match = SEMVER_RE.fullmatch(value)
    if not match:
        raise ValueError(f"invalid dataset semantic version: {value}")
    prerelease = match.group("pre")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        1 if prerelease is None else 0,
        prerelease or "",
    )


def validate_current(manifest: dict, inventory: dict) -> None:
    version = manifest["dataset_version"]
    parsed = semver_key(version)
    partitions = {item["split"]: item for item in manifest["partitions"]}
    if set(partitions) != {"dev", *SEALED_SPLITS}:
        raise ValueError("dataset governance requires every partition exactly once")

    if manifest["status"] == "development":
        if parsed[3] != 0 or "dev" not in version.split("-", 1)[1].lower():
            raise ValueError("development dataset version must use a -dev prerelease")
        if partitions["dev"]["frozen"]:
            raise ValueError("visible development data cannot be frozen as a holdout")
        for split in SEALED_SPLITS:
            partition = partitions[split]
            if partition["frozen"] and partition["expected_cases"] == 0:
                raise ValueError(f"empty sealed partition cannot be frozen: {split}")
    elif manifest["status"] == "frozen":
        if parsed[3] != 1:
            raise ValueError("frozen dataset version cannot be a prerelease")
        if not all(partition["frozen"] for partition in partitions.values()):
            raise ValueError("every partition must be frozen in a frozen dataset")
        for split in ("release", "adversarial", "real_user"):
            if partitions[split]["expected_cases"] == 0:
                raise ValueError(f"frozen dataset requires non-empty {split} split")
        if not inventory["freeze_readiness"]["ready"]:
            raise ValueError("review inventory is not ready for dataset freeze")


def read_manifest_from_git(ref: str) -> dict | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:evals/v2/dataset-manifest.json"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def compare_against_base(base: dict, current: dict) -> None:
    if base["status"] != "frozen":
        return
    base_version = base["dataset_version"]
    current_version = current["dataset_version"]
    if current_version == base_version:
        if current != base:
            raise ValueError(
                "frozen dataset changed without an explicit dataset-version change"
            )
        return
    if semver_key(current_version) <= semver_key(base_version):
        raise ValueError("replacement dataset version must increase after a freeze")
    archive = ROOT / "evals" / "v2" / "releases" / base_version / "dataset-manifest.json"
    if not archive.is_file() or load_json(archive) != base:
        raise ValueError(
            f"previous frozen manifest must be preserved at "
            f"evals/v2/releases/{base_version}/dataset-manifest.json"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref")
    args = parser.parse_args()
    manifest = load_json(MANIFEST_PATH)
    inventory = load_json(INVENTORY_PATH)
    validate_current(manifest, inventory)

    base_ref = args.base_ref
    if base_ref is None and os.environ.get("GITHUB_BASE_REF"):
        base_ref = f"origin/{os.environ['GITHUB_BASE_REF']}"
    if base_ref:
        base = read_manifest_from_git(base_ref)
        if base is None:
            print(f"no V2 manifest found at base ref {base_ref}; cross-version check skipped")
        else:
            compare_against_base(base, manifest)
            print(f"cross-version governance: OK against {base_ref}")

    print(
        f"dataset governance: OK ({manifest['dataset_version']}, "
        f"status={manifest['status']})"
    )


if __name__ == "__main__":
    main()
