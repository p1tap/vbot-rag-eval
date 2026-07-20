# Corpus and evaluation coverage

Status: generated inventory plus review guidance

The machine-readable source of truth is `evals/v2/coverage-matrix.json`, built
by `scripts/build_coverage_matrix.py` from the pinned corpus, evidence catalog,
and dataset manifest.

The inventory reports:

- Evidence blocks and block types per document and heading.
- Legacy answerable-case coverage by document and heading.
- Headings with no current answerable case.
- Span-level review coverage, currently zero by design.
- Current and approved case counts for every proposed V2 lane.
- A provisional three-case seed minimum for exercising each lane workflow.

Three cases per lane is not a statistical target. It is the minimum review and
execution seed used to prove that the lane schema, fixtures, scoring, and
reporting path work. Release-scale counts are decided only after independence,
traffic importance, severity, observed failure rate, and uncertainty are
reviewed.

Current V1 intent-family counts are migration placeholders, not validated
independence judgments. Reviewers must merge shared intents/evidence clusters
where appropriate rather than treating one legacy ID as one independent unit.

Regenerate and verify:

```bash
python scripts/build_coverage_matrix.py --write
python scripts/build_coverage_matrix.py --check
```

