"""Run the RAG eval over the frozen golden set → eval-report.json.

Retrieval metrics are deterministic and free (recall@k, MRR). Generation
metrics use an LLM judge from a different family than the answer model
(faithfulness/groundedness, answer correctness, and abstention behavior).

The expensive pass runs locally and commits eval-report.json; CI only
enforces the gate against the committed artifact (see compare_gate.py) —
the same gate-on-artifact pattern as the fine-tuning lab, so CI needs no
GPU, no keys, no network.

Run:  PYTHONNOUSERSITE=1 conda run -n ft python scripts/run_eval.py
Flags: --no-llm (retrieval only)  --limit N (smoke)  --out path
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent

import config  # noqa: E402
from rag.generate import answer, format_context, is_abstention  # noqa: E402
from rag.llm import chat  # noqa: E402
from rag.retrieve import load_index, retrieve  # noqa: E402


def _judge_int(prompt):
    out = chat(config.JUDGE_MODEL, [{"role": "user", "content": prompt}], max_tokens=6, temperature=0.0)
    m = re.search(r"\d+", out or "")
    return max(0, min(10, int(m.group()))) if m else 0


def judge_correctness(question, reference, model_answer):
    return _judge_int(
        "Score 0-10 how well the ANSWER conveys the key fact of the REFERENCE for the QUESTION. "
        "Ignore wording and extra detail; 10 = correct and complete on the key fact, 0 = wrong or missing.\n\n"
        f"QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {model_answer}\n\nReply with ONLY the integer.")


def judge_faithfulness(context, model_answer):
    return _judge_int(
        "Score 0-10 whether every factual claim in the ANSWER is supported by the CONTEXT. "
        "10 = fully grounded in the context, 0 = contains claims not in the context.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER: {model_answer}\n\nReply with ONLY the integer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="retrieval metrics only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "eval-report.json"))
    args = ap.parse_args()

    golden = [json.loads(l) for l in (ROOT / "evals" / "golden.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        golden = golden[: args.limit]
    index = load_index()

    items, hits, rrs = [], [], []
    correctness, faithfulness, answered_flags, abstains = [], [], [], []
    for g in golden:
        chunks = retrieve(g["question"], index=index)
        keys = [c["heading_key"] for c in chunks]
        rel = set(g["relevant"])
        rank = next((i + 1 for i, k in enumerate(keys) if k in rel), 0)
        rec = {"id": g["id"], "answerable": g["answerable"], "retrieved": keys,
               "hit": bool(rank), "rank": rank}

        if g["answerable"]:
            hits.append(1.0 if rank else 0.0)
            rrs.append(1.0 / rank if rank else 0.0)

        if not args.no_llm:
            ans, _ = answer(g["question"], chunks)
            abst = is_abstention(ans)
            rec["answer"] = ans
            rec["abstained"] = abst
            if g["answerable"]:
                answered_flags.append(0.0 if abst else 1.0)
                if abst:
                    rec["correctness"] = 0  # wrongly refused an answerable question
                    correctness.append(0.0)
                else:
                    c = judge_correctness(g["question"], g["answer"], ans)
                    f = judge_faithfulness(format_context(chunks), ans)
                    rec["correctness"], rec["faithfulness"] = c, f
                    correctness.append(c / 10.0)
                    faithfulness.append(f / 10.0)
            else:
                abstains.append(1.0 if abst else 0.0)
        items.append(rec)
        print(f"  {g['id']} {'✓' if rank or not g['answerable'] else '·'} "
              f"{'abstain' if not g['answerable'] else 'rank ' + str(rank)}")

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {
        "recall_at_k": mean(hits),
        "mrr": mean(rrs),
        "answer_rate": mean(answered_flags) if not args.no_llm else None,
        "correctness": mean(correctness) if not args.no_llm else None,
        "faithfulness": mean(faithfulness) if not args.no_llm else None,
        "abstention": mean(abstains) if not args.no_llm else None,
    }
    report = {
        "summary": summary,
        "config": {"chunk_max_words": config.CHUNK_MAX_WORDS,
                   "chunk_overlap_words": config.CHUNK_OVERLAP_WORDS,
                   "top_k": config.TOP_K, "embed_model": config.EMBED_MODEL,
                   "gen_model": config.GEN_MODEL},
        "meta": {"n": len(golden), "n_answerable": sum(g["answerable"] for g in golden),
                 "llm": not args.no_llm, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "items": items,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<14} {v}")
    print(f"\nreport → {args.out}")


if __name__ == "__main__":
    main()
