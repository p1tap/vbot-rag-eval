"""Shared deterministic helpers for claim-judge contracts."""
from __future__ import annotations

import re


_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_PATH_RE = re.compile(
    r"(?<![\w.-])(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.-]+)?"
)
_NUMBER_RE = re.compile(
    r"(?<![\w])[$#]?\d+(?:\.\d+)*(?:\+[A-Za-z0-9_.-]+)?(?:%|[kKmMbB])?"
)
_NUMERIC_UNIT_RE = re.compile(
    r"(?P<number>(?<![\w])[$#]?\d+(?:\.\d+)*(?:%|[kKmMbB])?)"
    r"(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,2}\s+"
    r"(?P<unit>pairs?|models?|parameters?|calls?|requests?|failures?|seconds?|"
    r"minutes?|hours?|days?|tokens?|ports?|errors?|retries?|replicas?|hosts?|"
    r"workers?|users?)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|neither|without|cannot|can't|won't|doesn't|"
    r"does\s+not|do\s+not|did\s+not|is\s+not|are\s+not|must\s+not)\b",
    re.IGNORECASE,
)
_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


def namespaced_ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index}" for index in range(1, count + 1)]


def extract_exact_anchors(text: str) -> tuple[str, ...]:
    """Return explicit values whose omission cannot be repaired semantically.

    The helper intentionally stays conservative. It protects literal URLs,
    code/path spans, and numeric values while leaving ordinary prose to the
    semantic judge.
    """

    anchors: list[str] = []
    occupied: list[tuple[int, int]] = []

    def add(value: str, span: tuple[int, int]) -> None:
        normalized = _normalize_anchor(value)
        if normalized and normalized not in anchors:
            anchors.append(normalized)
        occupied.append(span)

    for match in _URL_RE.finditer(text):
        value = match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
        add(value, match.span())
    for match in _BACKTICK_RE.finditer(text):
        add(match.group(1), match.span())
    for match in _PATH_RE.finditer(text):
        if not _overlaps(match.span(), occupied):
            add(match.group(0), match.span())
    for match in _NUMBER_RE.finditer(text):
        if not _overlaps(match.span(), occupied):
            add(match.group(0), match.span())
    for match in _NUMERIC_UNIT_RE.finditer(text):
        number = _normalize_anchor(match.group("number"))
        unit = match.group("unit").casefold().rstrip("s")
        anchors.append(f"<numeric-unit:{number}:{unit}>")

    lowered = text.casefold()
    if re.search(r"\bonce\b", lowered):
        anchors.append("<quantity:once>")
    if re.search(r"\btwice\b", lowered):
        anchors.append("<quantity:twice>")
    return tuple(dict.fromkeys(anchors))


def missing_exact_anchors(reference: str, candidate: str) -> tuple[str, ...]:
    normalized_candidate = _normalize_text(candidate)
    missing: list[str] = []
    for anchor in extract_exact_anchors(reference):
        if anchor == "<quantity:once>":
            if not re.search(r"\b(?:once|one\s+time|one\s+retry)\b", normalized_candidate):
                missing.append(anchor)
        elif anchor == "<quantity:twice>":
            if not re.search(r"\b(?:twice|two\s+times|two\s+retries)\b", normalized_candidate):
                missing.append(anchor)
        elif anchor.startswith("<numeric-unit:"):
            _, number, unit = anchor[1:-1].split(":", 2)
            unit_pattern = re.escape(unit) + r"s?"
            number_pattern = re.escape(number)
            nearby = re.compile(
                rf"(?:{number_pattern}(?:\s+\S+){{0,3}}\s+{unit_pattern}\b|"
                rf"\b{unit_pattern}(?:\s+\S+){{0,3}}\s+{number_pattern})",
                re.IGNORECASE,
            )
            if not nearby.search(normalized_candidate):
                missing.append(anchor)
        elif anchor not in normalized_candidate:
            missing.append(anchor)
    return tuple(missing)


def missing_explicit_negation(reference: str, candidate: str) -> bool:
    """Return true when an explicitly negative reference loses its polarity."""
    return bool(_NEGATION_RE.search(reference) and not _NEGATION_RE.search(candidate))


def _normalize_anchor(value: str) -> str:
    value = value.strip().rstrip(_TRAILING_URL_PUNCTUATION).casefold()
    if value.startswith(("$", "#")) and len(value) > 1 and value[1].isdigit():
        value = value[1:]
    return _normalize_text(value)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


__all__ = [
    "extract_exact_anchors",
    "missing_exact_anchors",
    "missing_explicit_negation",
    "namespaced_ids",
]
