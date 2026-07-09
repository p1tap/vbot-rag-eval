"""Adopt the current eval-report's summary as the committed baseline.

This is the ONLY way the baseline changes — run it, then commit
baselines/baseline.json in a reviewed PR. Keeps promotion an explicit,
diffable act rather than an ad-hoc edit.

Run: python scripts/promote_baseline.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
report = json.loads((ROOT / "eval-report.json").read_text(encoding="utf-8"))
out = {"summary": report["summary"], "config": report["config"],
       "adopted_from": report["meta"]}
(ROOT / "baselines").mkdir(exist_ok=True)
(ROOT / "baselines" / "baseline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("baseline updated:")
for k, v in report["summary"].items():
    print(f"  {k:<14} {v}")
