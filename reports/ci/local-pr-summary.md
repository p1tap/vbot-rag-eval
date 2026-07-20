## RAG evidence gate

**Decision:** LOCAL_CHECK

| Evidence | Result |
|---|---:|
| V1 retrieval recall@k | 97.40% |
| V1 correctness | 92.60% |
| V2 dataset | 2.0.0-dev.2 · 100 cases |
| Public suite integrity | machine_validated_ai_spot_audit_passed · 10000 cases |
| Public end-to-end macro joint | n/a |
| Human-reference judge agreement | 76.00% |

Provenance: the public 10K inherits publisher human annotations; it is not locally human-reviewed. The Vbot domain seed contains 100 locally owner-reviewed cases. PROMOTE/REJECT examples are hash-locked in `docs/promotion-evidence.json`.

Public claim blockers: RAG retrieval/generation evaluation has not yet run across this suite
