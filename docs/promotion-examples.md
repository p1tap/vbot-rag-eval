# Authentic promotion examples

The promotion system retains both sides of the decision boundary. The manifest
in `docs/promotion-evidence.json` hash-locks each decision and its input reports;
CI rejects silent edits through `scripts/check_promotion_examples.py`.

## Promoted: GPT public pilot, k=4 to k=8

The paired 300-case public pilot improved macro joint correctness by 0.02, had
no per-benchmark regression beyond the declared bounds, and did not add
fail-closed cases. The decision was `PROMOTE` within the explicitly limited
scope `paired_generator_or_retrieval_pilot_not_release_promotion`. It was not a
production release promotion and does not imply statistical significance.

## Rejected: Vbot structured development, k=4 to k=6

The deeper retrieval candidate improved several recall/support metrics but lost
correct actions or passing cases in authorization, hard-negative, and prompt-
injection lanes. Action accuracy also regressed. The decision was `REJECT`,
showing that aggregate retrieval gains cannot override safety guardrails.

