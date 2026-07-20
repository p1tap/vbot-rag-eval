## RAG evidence gate

**Decision:** NOT_RUN

| Evidence | Result |
|---|---:|
| V1 retrieval recall@k | 97.40% |
| V1 correctness | 92.60% |
| V2 dataset | 2.0.0-dev.5 · 162 cases |
| Public suite integrity | end_to_end_evaluated · 10000 cases |
| Public end-to-end macro joint | 31.87% |
| Human-reference judge agreement | 76.00% |

Provenance: the public 10K inherits publisher human annotations; it is not locally human-reviewed. The Vbot domain seed contains 100 locally owner-reviewed cases. PROMOTE/REJECT examples are hash-locked in `docs/promotion-evidence.json`.
