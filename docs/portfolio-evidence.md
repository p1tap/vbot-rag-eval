# Portfolio evidence and claim boundaries

## Defensible project summary

Built a corpus-agnostic RAG evaluation and serving platform with frozen data
contracts, reproducible BM25/E5 retrieval experiments, claim-level citation
verification, calibrated multi-model judging, resumable public end-to-end
evaluation, provenance-aware CI gates, and a version-reporting FastAPI runtime.

The clean 10,000-case end-to-end report passed the versioned contract audit.
The strongest concise dataset claim is:

> Evaluated the RAG pipeline across 10,000 validated, publisher-human-annotated
> public benchmark cases, plus a 100-case locally human-reviewed Vbot domain set
> covering 15 failure lanes.

Do not shorten that to “10,000 locally human-reviewed cases.” HotpotQA, Natural
Questions, and FEVER inherit human annotations from their publishers; local
work validated conversion/integrity and performed a 48-case AI spot audit.
The audited public report contains 10,000/10,000 evaluated cases, 2,500/2,500
valid batches, zero fail-closed cases, and 10,000 unique case IDs. This wording
is now a verified project claim.

## Verified evidence

- Vbot dataset: 100/100 locally owner-reviewed development cases, 15/15 lanes,
  and 19/19 corpus headings represented.
- Judge calibration: 175 frozen human tasks; selected cascade achieved 76.0%
  overall and 86.5% confirmation exact agreement with zero false accepts.
- Retrieval: E5 complete-support recall at k=4 reached 76.17% on 3,000 HotpotQA
  cases and 67.23% on 2,945 answerable Natural Questions cases, beating BM25;
  RRF was rejected where it regressed evidence completeness.
- Public generation pilots: GPT-5.4 reached 48.33% macro joint correctness on a
  fixed 300-case k=8 pilot. Local Qwen3.5 9B is weaker and is used as the
  zero-provider-cost scalability executor, not represented as the quality
  winner.
- Public full run: pinned local Qwen3.5 9B completed all 10,000 cases under
  runner `2.2.0` at 31.87% macro joint correctness. The audited run had zero
  fail-closed cases, 1.64% allowlisted normalized batches, and 0.52% isolated
  retry batches; all rates passed their frozen caps.
- Vbot structured generation: the selected local policy achieved 100% contract
  validity, 88% action accuracy, and zero false answers on 100 human-reviewed
  development cases. Its one-time 50-case same-agent AI release result was
  100%, 90%, and zero. The 12 derivative adversarial regressions rejected the
  action policy at 58.33% accuracy and remain a visible limitation.
- Service: a bounded 20-request real-Qwen test completed with 0% errors,
  0.302 requests/s, p95 3.39 s, and p99 3.44 s at concurrency one. A separate
  100-request final-container mock test reached 10.25 requests/s; it is
  explicitly not model-inference capacity.

## Limitations that should stay visible

- The Vbot 100 cases are development data, not a blinded release set.
- The public inputs are development sets and may appear in model training data.
- The selected judge cascade is conservative: zero observed false accepts came
  with a 21.1% false-reject rate and incomplete DeepSeek V4-max coverage after
  the external account returned HTTP 402.
- The local Qwen same-family judge was rejected after only 59.43% valid tasks
  and five high/critical false accepts; it cannot certify local generations.
- The adversarial action gate failed, and no genuine real-user partition exists.
- A scalar retrieval-score abstention threshold was rejected because no point
  met both false-answer and false-refusal guardrails.
- Local Qwen output sometimes needs narrowly enumerated, audited serialization
  normalization; the full run used 1.64% against a frozen 5% cap.
- The corpus secret scan detects high-confidence credential patterns, not all
  possible sensitive prose.
