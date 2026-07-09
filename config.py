"""Tunable knobs for the RAG pipeline.

This is the file a promotion PR edits. Every retrieval/generation behavior
that the eval gate scores is controlled from here, so a reviewer can see the
whole change surface in one diff (chunking, k, the answer prompt) and the
gate decides whether it ships. Same discipline as the model gateway's config
and the backbone fine-tune's hyperparameters — the knob and the gate live
next to each other.
"""

# --- retrieval ---
EMBED_MODEL = "intfloat/e5-small-v2"   # mean-pooled, query/passage-prefixed
CHUNK_MAX_WORDS = 180                   # sections longer than this are windowed
CHUNK_OVERLAP_WORDS = 40               # overlap between windows of a long section
TOP_K = 1                              # chunks retrieved per query

# --- generation ---
# Answer model is routed through an OpenAI-compatible base URL. Default is
# OpenRouter directly; point RAG_LLM_BASE_URL at the Vbot gateway's virtual
# key to get the per-key spend ledger (see README "Serving path").
GEN_MODEL = "meta-llama/llama-3.1-8b-instruct"   # different family from the judge
GEN_MAX_TOKENS = 300
GEN_TEMPERATURE = 0.0                  # deterministic answers → stable eval

# The abstention contract: the generator is instructed to say exactly this
# when the retrieved context does not contain the answer. The eval's
# abstention lane checks that it fires on out-of-corpus questions and does
# NOT fire on answerable ones.
ABSTAIN_STRING = "I don't have enough information in the provided context to answer that."

ANSWER_SYSTEM = (
    "You answer strictly from the provided context passages about the Vbot "
    "AI platform (its model gateway and fine-tuning lab). Use only facts "
    "stated in the context. Do not use outside knowledge. If the context "
    "does not contain the answer, reply with exactly this sentence and "
    f"nothing else: \"{ABSTAIN_STRING}\" Keep answers to 1-3 sentences."
)

# --- judge ---
JUDGE_MODEL = "google/gemini-2.5-flash-lite"   # different family from generator
