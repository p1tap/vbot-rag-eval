"""Check bakeoff model availability and strict structured-output behavior."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from rag import llm  # noqa: E402
from rag.llm import chat_with_metadata  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "v2" / "judge-preflight.json"


def response_format(profile_id: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_preflight",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ok", "profile_id"],
                "properties": {
                    "ok": {"type": "boolean", "const": True},
                    "profile_id": {"type": "string", "const": profile_id},
                },
            },
        },
    }


def validate_profiles(profiles: tuple[dict, ...]) -> None:
    ids = [profile.get("id") for profile in profiles]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("every judge profile needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("judge profile ids must be unique")
    for profile in profiles:
        if set(profile) != {"id", "model", "request_options"}:
            raise ValueError(f"profile {profile['id']} has missing or extra fields")
        if not isinstance(profile["model"], str) or not profile["model"]:
            raise ValueError(f"profile {profile['id']} needs a model slug")
        options = profile["request_options"]
        if not isinstance(options, dict):
            raise ValueError(f"profile {profile['id']} request_options must be an object")
        provider = options.get("provider")
        if not isinstance(provider, dict) or provider.get("require_parameters") is not True:
            raise ValueError(f"profile {profile['id']} must require all request parameters")
        only = provider.get("only")
        if not isinstance(only, list) or len(only) != 1 or not isinstance(only[0], str):
            raise ValueError(f"profile {profile['id']} must pin exactly one provider")
        if provider.get("allow_fallbacks") is not False:
            raise ValueError(f"profile {profile['id']} must disable provider fallbacks")


def validate_smoke_content(content: str, profile_id: str) -> dict:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    expected = {"ok": True, "profile_id": profile_id}
    if value != expected:
        raise ValueError(f"strict response mismatch: expected {expected}, received {value}")
    return value


def load_catalog() -> dict[str, dict]:
    headers = {"Accept": "application/json"}
    if llm.API_KEY:
        headers["Authorization"] = f"Bearer {llm.API_KEY}"
    response = llm.requests.get(f"{llm.BASE_URL}/models", headers=headers, timeout=30)
    response.raise_for_status()
    body = response.json()
    rows = body.get("data") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        raise ValueError("model catalog response does not contain a data array")
    return {
        row["id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }


def select_profiles(value: str) -> tuple[dict, ...]:
    profiles = tuple(config.JUDGE_BAKEOFF_PROFILES)
    validate_profiles(profiles)
    if not value:
        return profiles
    wanted = {item.strip() for item in value.split(",") if item.strip()}
    selected = tuple(profile for profile in profiles if profile["id"] in wanted)
    unknown = wanted - {profile["id"] for profile in selected}
    if unknown:
        raise ValueError(f"unknown profile ids: {sorted(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="", help="comma-separated profile IDs")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.max_tokens < 64:
        raise SystemExit("--max-tokens must be at least 64")

    try:
        profiles = select_profiles(args.profiles)
        catalog = load_catalog()
    except Exception as exc:  # noqa: BLE001 - operator-facing preflight report
        raise SystemExit(f"preflight setup failed: {exc}") from exc

    results = []
    for profile in profiles:
        catalog_row = catalog.get(profile["model"])
        result = {
            "profile_id": profile["id"],
            "model": profile["model"],
            "request_options": profile["request_options"],
            "catalog_available": catalog_row is not None,
            "catalog_reasoning": catalog_row.get("reasoning") if catalog_row else None,
            "supported_parameters": catalog_row.get("supported_parameters")
            if catalog_row
            else None,
            "smoke_attempted": not args.catalog_only and catalog_row is not None,
            "smoke_valid": None,
            "error": None,
        }
        if not catalog_row:
            result["error"] = "model slug is absent from the provider catalog"
        elif not args.catalog_only:
            try:
                call = chat_with_metadata(
                    profile["model"],
                    [
                        {
                            "role": "system",
                            "content": "Return only the requested strict JSON object.",
                        },
                        {
                            "role": "user",
                            "content": f"Confirm preflight profile {profile['id']}.",
                        },
                    ],
                    max_tokens=args.max_tokens,
                    # With require_parameters enabled, sending an unsupported
                    # optional parameter correctly eliminates every endpoint.
                    temperature=(
                        0.0
                        if "temperature"
                        in (catalog_row.get("supported_parameters") or [])
                        else None
                    ),
                    retries=1,
                    response_format=response_format(profile["id"]),
                    request_options=profile["request_options"],
                )
                validate_smoke_content(call.content, profile["id"])
                result["smoke_valid"] = True
                result["call"] = call.to_record()
            except Exception as exc:  # noqa: BLE001 - preserve every profile result
                result["smoke_valid"] = False
                result["error"] = str(exc)
        results.append(result)

    report = {
        "schema_version": "1.0.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_only": args.catalog_only,
        "results": results,
        "valid": all(
            item["catalog_available"]
            and (args.catalog_only or item["smoke_valid"] is True)
            for item in results
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
