"""RAG answer generation: retrieve → stuff context → answer or abstain."""
from config import ABSTAIN_STRING, ANSWER_SYSTEM, GEN_MAX_TOKENS, GEN_MODEL, GEN_TEMPERATURE
from rag.llm import chat
from rag.retrieve import retrieve


def format_context(chunks):
    return "\n\n".join(f"[{i + 1}] ({c['heading']})\n{c['text']}" for i, c in enumerate(chunks))


def answer(question, chunks=None):
    chunks = retrieve(question) if chunks is None else chunks
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Context passages:\n{format_context(chunks)}\n\nQuestion: {question}"},
    ]
    return chat(GEN_MODEL, messages, GEN_MAX_TOKENS, GEN_TEMPERATURE), chunks


def is_abstention(text):
    return bool(text) and "enough information" in text.lower()


__all__ = ["answer", "is_abstention", "format_context", "ABSTAIN_STRING"]
