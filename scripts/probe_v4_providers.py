"""Measure pinned DeepSeek V4 strict-schema reliability before provider freeze."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.llm import chat_with_metadata  # noqa: E402
from scripts.preflight_judge_models import response_format, validate_smoke_content  # noqa: E402

MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_PROVIDERS = (
    "streamlake/fp8",
    "baidu/fp8",
    "parasail/fp8",
    "akashml/fp8",
    "digitalocean",
    "nextbit/fp8",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--providers", default=",")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "reports/v2/judge-v4-provider-probe.json"
    )
    args = parser.parse_args()
    providers = tuple(
        value.strip() for value in args.providers.split(",") if value.strip()
    ) or DEFAULT_PROVIDERS
    if args.trials < 1:
        raise SystemExit("--trials must be positive")

    rows = []
    for provider in providers:
        for effort in ("high", "xhigh"):
            profile_id = f"v4-{provider.replace('/', '-')}-{effort}"
            for trial in range(1, args.trials + 1):
                row = {
                    "provider": provider,
                    "effort": effort,
                    "trial": trial,
                    "valid": False,
                    "error": None,
                }
                try:
                    call = chat_with_metadata(
                        MODEL,
                        [
                            {"role": "system", "content": "Return only the requested strict JSON object."},
                            {"role": "user", "content": f"Confirm {profile_id}, trial {trial}."},
                        ],
                        max_tokens=512,
                        temperature=0.0,
                        retries=1,
                        response_format=response_format(profile_id),
                        request_options={
                            "reasoning": {"effort": effort, "exclude": True},
                            "provider": {
                                "only": [provider],
                                "allow_fallbacks": False,
                                "require_parameters": True,
                            },
                        },
                    )
                    validate_smoke_content(call.content, profile_id)
                    row["valid"] = True
                    row["call"] = call.to_record()
                except Exception as exc:  # noqa: BLE001 - retain failed provider evidence
                    row["error"] = str(exc)
                rows.append(row)

    summary = []
    for provider in providers:
        selected = [row for row in rows if row["provider"] == provider]
        summary.append(
            {
                "provider": provider,
                "valid": sum(row["valid"] for row in selected),
                "total": len(selected),
                "all_valid": all(row["valid"] for row in selected),
            }
        )
    report = {
        "schema_version": "1.0.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "trials_per_effort": args.trials,
        "summary": summary,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if any(row["all_valid"] for row in summary) else 1)


if __name__ == "__main__":
    main()
