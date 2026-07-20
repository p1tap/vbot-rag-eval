# Portfolio evidence and claim boundaries

## Defensible project summary

Built a corpus-agnostic RAG evaluation and serving platform with frozen data
contracts, reproducible BM25/E5 retrieval experiments, claim-level citation
verification, calibrated multi-model judging, resumable public end-to-end
evaluation, provenance-aware CI gates, and a version-reporting FastAPI runtime.

The clean 10,000-case end-to-end report passed the versioned contract audit.
The strongest concise resume claim is:

> Built a RAG evaluation system covering 10,000 human-reviewed cases; raised
> overall strict answer-and-citation correctness from 31.9% to 60.5% by
> improving answer selection and supporting-evidence retrieval.

Do not shorten that to “10,000 locally human-reviewed cases.” HotpotQA, Natural
Questions, and FEVER inherit human annotations from their publishers; local
work validated conversion/integrity and performed a 48-case AI spot audit.
The audited public report contains 10,000/10,000 evaluated cases, 2,500/2,500
valid batches, zero fail-closed cases, and 10,000 unique case IDs. This wording
is now a verified project claim.

## Verified evidence

- Vbot dataset: 100/100 locally owner-reviewed development cases, 15/15 lanes,
  and 19/19 corpus headings represented.
- Judge calibration: 175 frozen human tasks; selected cascade achieved 78.3%
  overall and 86.5% confirmation exact agreement with zero false accepts.
- Retrieval: E5 complete-support recall at k=4 reached 76.17% on 3,000 HotpotQA
  cases and 67.23% on 2,945 answerable Natural Questions cases, beating BM25;
  RRF was rejected where it regressed evidence completeness.
- Public full run: the promoted specialist pipeline completed all 10,000 cases
  at 60.50% strict macro joint answer-and-citation correctness with zero
  fail-closed cases: 60.53% HotpotQA, 57.06% Natural Questions, and 63.90%
  FEVER. It improved the original 31.87% local-Qwen full-run baseline by 28.63
  percentage points; an independent audit recomputed every metric, confirmed
  10,000 unique IDs, and passed the declared 60% macro gate.
- Separate single-hop reader: a frozen 1,000-case SQuAD 2.0 oracle-context run
  with a pinned local SQuAD-2.0-fine-tuned extractive reader reached 91.08%
  token F1, 87.50% exact match, and 94.1% answerability accuracy with zero
  fail-closed cases. It excludes retrieval, uses the public development set,
  and must remain separate from the 10,000-case RAG score.
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
- The SQuAD reader was fine-tuned on SQuAD 2.0 and evaluated on its public
  development set; this is an in-domain reader benchmark, not evidence of
  blinded or out-of-domain generalization.
- The selected judge cascade is conservative: zero observed false accepts came
  with 34 false rejects and 12 fail-closed tasks. Its GLM primary reached only
  89.14% valid-task coverage, so it remains a calibrated cascade rather than a
  sole release oracle.
- A provider-pinned GLM answer-generator pilot reached 49.0% strict joint
  correctness on the frozen 300-case sample. It did not beat the 51.67%
  task-aware DeepSeek ensemble pilot and was not promoted.
- A predeclared GLM-for-Hotpot/DeepSeek-elsewhere router reproduced a positive
  Hotpot direction on a disjoint 300-case confirmation window, raising strict
  macro joint correctness from 46.67% to 47.67%. The Hotpot comparison was only
  7 paired wins versus 4 losses (exact two-sided p=0.549), so the promotion gate
  retained the existing pipeline rather than spending a full-run claim on weak
  evidence.
- The local Qwen same-family judge was rejected after only 59.43% valid tasks
  and five high/critical false accepts; it cannot certify local generations.
- The adversarial action gate failed, and no genuine real-user partition exists.
- A scalar retrieval-score abstention threshold was rejected because no point
  met both false-answer and false-refusal guardrails.
- Local Qwen output sometimes needs narrowly enumerated, audited serialization
  normalization; the full run used 1.64% against a frozen 5% cap.
- The corpus secret scan detects high-confidence credential patterns, not all
  possible sensitive prose.
