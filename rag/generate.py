"""RAG answer generation: retrieve → stuff context → answer or abstain."""
import json
from config import (
    ABSTAIN_STRING,
    ANSWER_SYSTEM,
    GEN_MAX_TOKENS,
    GEN_MODEL,
    GEN_REQUEST_OPTIONS,
    GEN_TEMPERATURE,
    CURRENT_SUPPORTED_ACTIONS,
    CURRENT_RUNTIME_POLICY,
    STRUCTURED_ANSWER_SYSTEM,
    STRUCTURED_GEN_MAX_TOKENS,
)
from rag.llm import chat, chat_with_metadata
from rag.retrieve import retrieve
from rag.structured_answer import (
    context_sources,
    format_sources,
    parse_structured_answer,
    structured_answer_response_format,
    validate_structured_answer,
)


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


def structured_answer(
    question,
    chunks=None,
    *,
    allowed_actions=None,
    runtime_policy=None,
    model=None,
    max_tokens=None,
    temperature=None,
    request_options=None,
):
    """Generate V2 JSON and fail closed on malformed or fictional citations.

    Returns ``(answer_object, chunks, audit)``. The raw provider output is kept
    in ``audit`` even when parsing or validation fails.
    """

    chunks = retrieve(question, unique_headings=True) if chunks is None else chunks
    allowed_actions = tuple(allowed_actions or CURRENT_SUPPORTED_ACTIONS)
    runtime_policy = runtime_policy or CURRENT_RUNTIME_POLICY
    model = model or GEN_MODEL
    max_tokens = max_tokens or STRUCTURED_GEN_MAX_TOKENS
    temperature = GEN_TEMPERATURE if temperature is None and model == GEN_MODEL else temperature
    request_options = GEN_REQUEST_OPTIONS if request_options is None else request_options
    sources = context_sources(chunks)
    response_format = structured_answer_response_format(
        (source["citation_id"] for source in sources), allowed_actions
    )
    messages = [
        {"role": "system", "content": STRUCTURED_ANSWER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Allowed actions: {', '.join(allowed_actions)}\n\n"
                "Runtime policy metadata: "
                f"{json.dumps(runtime_policy, sort_keys=True)}\n\n"
                f"Retrieved sources:\n{format_sources(sources)}\n\nQuestion: {question}"
            ),
        },
    ]
    calls, raw_outputs, retry_feedback = [], [], []
    value = None
    validation_record = None
    for _ in range(2):
        result = chat_with_metadata(
            model,
            messages,
            max_tokens,
            temperature,
            response_format=response_format,
            request_options=request_options,
        )
        calls.append(result.to_record())
        raw_outputs.append(result.content)
        try:
            candidate = parse_structured_answer(result.content)
        except ValueError as exc:
            value = None
            validation_record = {
                "valid": False,
                "errors": [{"code": "invalid_json", "message": str(exc)}],
                "metrics": {},
            }
            if len(calls) == 1:
                feedback = _structured_retry_feedback(validation_record)
                retry_feedback.append(feedback)
                messages.extend(
                    [
                        {"role": "assistant", "content": result.content},
                        {"role": "user", "content": feedback},
                    ]
                )
            continue
        validation = validate_structured_answer(
            candidate,
            (source["citation_id"] for source in sources),
            allowed_actions,
        )
        validation_record = validation.to_record()
        value = candidate
        if validation.valid:
            break
        if len(calls) == 1:
            feedback = _structured_retry_feedback(validation_record)
            retry_feedback.append(feedback)
            messages.extend(
                [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": feedback},
                ]
            )
    audit = {
        "call": calls[-1],
        "calls": calls,
        "raw_output": raw_outputs[-1],
        "raw_outputs": raw_outputs,
        "sources": sources,
        "allowed_actions": list(allowed_actions),
        "runtime_policy": runtime_policy,
        "retry_feedback": retry_feedback,
        "validation": validation_record,
    }
    return value, chunks, audit


def _structured_retry_feedback(validation_record: dict) -> str:
    """Describe validator failures without repairing or inferring answer content."""

    errors = validation_record.get("errors", [])
    details = "; ".join(
        f"{item.get('code', 'invalid')}: {item.get('message', 'contract violation')}"
        for item in errors
        if isinstance(item, dict)
    ) or "invalid: output did not satisfy the response contract"
    return (
        "Your previous output failed the response contract: "
        f"{details}. Return a new complete JSON object only. Do not use Markdown "
        "fences or prose. Use exactly the declared root and claim fields. Reconsider "
        "the action rules before emitting claims; do not merely repeat the invalid output."
    )


__all__ = [
    "answer",
    "structured_answer",
    "is_abstention",
    "format_context",
    "ABSTAIN_STRING",
]
