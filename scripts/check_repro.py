"""Reproducibility check — verifies the committed eval-report's deterministic
retrieval metrics (recall@k, MRR) can be reproduced from the corpus + config
in this PR.

This is the anti-fabrication half of the gate-on-artifact pattern: the gate
(compare_gate.py) trusts the committed eval-report.json for the expensive
LLM-judged metrics, so CI recomputes the free, deterministic half and asserts
it matches. A hand-edited report, or a config change whose report was never
regenerated, fails here.

Tolerance is 0.02: retrieval is deterministic on a given machine, but CPU
torch in CI can flip a near-tie ranking vs the local GPU run. One recall flip
costs 1/38 ≈ 0.026, so 0.02 absorbs float jitter while still catching any
real rank change.

Run: python scripts/check_repro.py --recomputed /tmp/repro-report.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETERMINISTIC = ("recall_at_k", "mrr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", default=str(ROOT / "eval-report.json"))
    ap.add_argument("--recomputed", required=True,
                    help="report from run_eval.py --no-llm on this checkout")
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args()

    committed = json.loads(Path(args.committed).read_text(encoding="utf-8"))["summary"]
    recomputed = json.loads(Path(args.recomputed).read_text(encoding="utf-8"))["summary"]

    ok = True
    print(f"{'metric':<14}{'committed':>10}{'recomputed':>12}  verdict")
    for m in DETERMINISTIC:
        c, r = committed.get(m), recomputed.get(m)
        if c is None or r is None:
            print(f"{m:<14}{c!s:>10}{r!s:>12}  MISSING → FAIL"); ok = False; continue
        d = abs(c - r)
        verdict = "ok" if d <= args.tol else "MISMATCH → FAIL"
        ok = ok and d <= args.tol
        print(f"{m:<14}{c:>10}{r:>12}  Δ{round(d, 3)}  {verdict}")

    if not ok:
        print("\nCommitted eval-report.json does not reproduce from this PR's "
              "corpus/config. Regenerate it: python scripts/run_eval.py")
        sys.exit(1)
    print("\nreproducibility: OK — committed retrieval metrics match this checkout")


if __name__ == "__main__":
    main()
