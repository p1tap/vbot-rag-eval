"""Promotion gate — compares a candidate eval-report against the committed
baseline and exits with the decision:

  0 PROMOTE        no metric regressed beyond tolerance
  1 NEEDS_REVIEW   a soft metric slipped within the review band
  2 REJECT         a guardrail metric regressed, or data is missing

Guardrails (regression here = REJECT): correctness (did we answer right)
and abstention (did we refuse the unanswerable). These are the "safe/correct"
guarantees, exactly like the fine-tuning gate treats calibration + general.
Everything else (recall@k, MRR, faithfulness, answer_rate) is a soft metric:
a small dip is REVIEW, a large dip is REJECT.

Promotion = replacing baselines/baseline.json with the candidate summary via
a reviewed PR (scripts/promote_baseline.py) — never an ad-hoc edit. No GPU,
no keys, no network: CI runs this against committed JSON.

Noise floor: LLM-judged metrics are not run-to-run deterministic even at
temperature 0 (provider-side inference varies) — the same config was measured
flipping one abstention item between runs. On a small lane, one item is a big
number: the abstention lane has 11 items, so a single flip moves the metric
by 0.091. The effective tolerance is therefore max(tol, 1.5/lane_n) per
metric — a regression only counts when it exceeds one item's worth of noise.
Lane sizes come from the candidate report's meta.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARD = {"correctness", "abstention"}


def summary_of(obj):
    return obj["summary"] if "summary" in obj else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "eval-report.json"))
    ap.add_argument("--baseline", default=str(ROOT / "baselines" / "baseline.json"))
    ap.add_argument("--tol", type=float, default=0.05, help="soft tolerance band")
    args = ap.parse_args()

    cand_obj = json.loads(Path(args.report).read_text(encoding="utf-8"))
    cand = summary_of(cand_obj)
    meta = cand_obj.get("meta", {})
    n_ans = meta.get("n_answerable")
    n_abst = meta.get("n") - n_ans if meta.get("n") and n_ans else None

    def eff_tol(metric):
        lane = n_abst if metric == "abstention" else n_ans
        return max(args.tol, 1.5 / lane) if lane else args.tol

    bpath = Path(args.baseline)
    if not bpath.exists():
        print("No committed baseline — this report becomes the candidate baseline.")
        print("Adopt with: python scripts/promote_baseline.py")
        sys.exit(0)
    base = summary_of(json.loads(bpath.read_text(encoding="utf-8")))

    worst, rows = 0, []
    for metric, b in base.items():
        if b is None:
            continue
        c = cand.get(metric)
        if c is None:
            rows.append((metric, b, "-", "MISSING → REJECT")); worst = max(worst, 2); continue
        delta = round(c - b, 3)
        tol = eff_tol(metric)
        if delta >= -tol:
            v = "ok"
        elif metric in HARD:
            v = "GUARDRAIL REGRESSED → REJECT"; worst = max(worst, 2)
        elif delta >= -2 * tol:
            v = "soft regression → REVIEW"; worst = max(worst, 1)
        else:
            v = "large regression → REJECT"; worst = max(worst, 2)
        rows.append((metric, b, c, f"{'+' if delta > 0 else ''}{delta}  {v}"))

    print(f"{'metric':<14}{'baseline':>10}{'candidate':>11}  verdict")
    for m, b, c, v in rows:
        print(f"{m:<14}{b:>10}{c:>11}  {v}")
    decision = ["PROMOTE", "NEEDS_REVIEW", "REJECT"][worst]
    print(f"\nDECISION: {decision}")
    sys.exit(worst)


if __name__ == "__main__":
    main()
