"""Shared utilities for assessment tasks."""

from __future__ import annotations

import re
from typing import Tuple

__all__ = [
    "YEAR_HYPHEN_PATTERN",
    "YEAR_COMPACT_PATTERN",
    "tokenize_mixed",
    "infer_instrument_pattern",
]

# Instrument number heuristics ------------------------------------------------- #

# Accepts values like 2020-12345 with flexible trailing digit length.
YEAR_HYPHEN_PATTERN = re.compile(r"^\d{4}-\d{5,7}$")
# Accepts compact values like 202012345 without a separator.
YEAR_COMPACT_PATTERN = re.compile(r"^\d{4}\d{5,7}$")


def tokenize_mixed(text: str) -> Tuple[str, str]:
    """Tokenize mixed instrument strings into (label, regex) components."""

    if not text:
        return "empty", ""

    segments: list[tuple[str, str]] = []
    current_type: str | None = None
    current_chars: list[str] = []

    def flush_segment() -> None:
        if current_chars and current_type:
            segments.append((current_type, "".join(current_chars)))

    for char in text:
        char_type = "N" if char.isdigit() else "A" if char.isalpha() else "X"
        if char_type == current_type:
            current_chars.append(char)
        else:
            flush_segment()
            current_type = char_type
            current_chars = [char]
    flush_segment()

    label_parts: list[str] = []
    regex_parts: list[str] = []
    for seg_type, value in segments:
        length = len(value)
        if seg_type == "N":
            label_parts.append(f"N{length}")
            regex_parts.append(rf"\d{{{length}}}")
        elif seg_type == "A":
            label_parts.append(f"A{length}")
            regex_parts.append(rf"[A-Za-z]{{{length}}}")
        else:
            label_parts.append(f"X{length}")
            regex_parts.append(re.escape(value))
    return "-".join(label_parts), "".join(regex_parts)


def infer_instrument_pattern(raw_value: str) -> tuple[str, str]:
    """
    Infer a pattern label and regex for an instrument number.

    Handles `bp`/`pb` prefixes, year-prefixed strings with or without hyphens,
    numeric-only values, and mixed alphanumeric sequences by delegating to
    `tokenize_mixed`. Returns a `(pattern_label, regex)` tuple.
    """

    value = (raw_value or "").strip()
    if not value:
        return "missing", r"^$"

    prefix = ""
    remainder = value
    prefix_candidate = remainder[:2].lower()
    if prefix_candidate in {"bp", "pb"}:
        prefix = prefix_candidate
        remainder = remainder[2:]

    prefix_label = f"{prefix}-" if prefix else ""
    prefix_regex = re.escape(prefix) if prefix else ""

    if YEAR_HYPHEN_PATTERN.fullmatch(remainder):
        return f"{prefix_label}YYYY-N{{5-7}}", rf"^{prefix_regex}\d{{4}}-\d{{5,7}}$"
    if YEAR_COMPACT_PATTERN.fullmatch(remainder):
        return f"{prefix_label}YYYYN{{5-7}}", rf"^{prefix_regex}\d{{4}}\d{{5,7}}$"
    if remainder.isdigit():
        digit_len = len(remainder)
        return (
            f"{prefix_label}digits-{digit_len}",
            rf"^{prefix_regex}\d{{{digit_len}}}$",
        )

    label, regex_body = tokenize_mixed(remainder)
    return f"{prefix_label}{label}", rf"^{prefix_regex}{regex_body}$"
