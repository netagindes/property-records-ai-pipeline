"""Shared utilities for assessment tasks."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from dotenv import load_dotenv
from openai import OpenAI

# from google import genai  # TODO

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_FOLDER = REPO_ROOT.joinpath("data")
OUTPUT_FOLDER = REPO_ROOT.joinpath("assessment_solution", "outputs")

DEFAULT_INPUT_JSONL = DATA_FOLDER.joinpath("nc_records_assessment.jsonl")
DEFAULT_OUTPUT_COUNTY_PATTERNS = OUTPUT_FOLDER.joinpath("county_patterns.json")
DEFAULT_OUTPUT_DOC_TYPE_MAPPING = OUTPUT_FOLDER.joinpath("doc_type_mapping.json")

__all__ = [
    "REPO_ROOT",
    "DATA_FOLDER",
    "OUTPUT_FOLDER",
    "DEFAULT_INPUT_JSONL",
    "DEFAULT_OUTPUT_COUNTY_PATTERNS",
    "DEFAULT_OUTPUT_DOC_TYPE_MAPPING",
    "parse_llm_json",
    "build_openai_client",
    # "build_gemini_client",  # TODO
    "CostTracker",
    "stream_jsonl",
    "normalize_doc_type",
    "safe_parse_date",
    "YEAR_HYPHEN_PATTERN",
    "YEAR_COMPACT_PATTERN",
    "tokenize_mixed",
    "infer_instrument_pattern",
]

# Pricing (per 1M tokens) for OpenAI gpt-4o-mini (as of Jan 2026).
# Reference: https://openai.com/api/pricing
GPT4O_MINI_INPUT_RATE = 0.15 / 1_000_000
GPT4O_MINI_OUTPUT_RATE = 0.60 / 1_000_000

# # TODO:
# # Pricing (per 1M tokens) for Gemini 2.5 Flash-Lite standard tier.
# # Reference: https://ai.google.dev/gemini-api/docs/pricing?utm_source=chatgpt.com#gemini-2.5-flash-lite
# FLASH_LITE_INPUT_RATE = 0.30 / 1_000_000
# FLASH_LITE_OUTPUT_RATE = 2.50 / 1_000_000
MAX_RETRIES = 3


# LLM helpers ----------------------------------------------------------------- #


def parse_llm_json(raw_text: str) -> dict[str, str]:
    """Parse JSON content from the LLM response, stripping fences if present."""

    cleaned = raw_text.strip()
    if not cleaned:
        return {}

    # Handle common Markdown fences.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        LOGGER.error(f"Failed to parse LLM JSON: {e} | {cleaned}")
        raise e


def build_openai_client() -> OpenAI:
    """
    Configure and return an OpenAI client using OPENAI_API_KEY from the environment.

    Loads `.env` for local development convenience. Raises RuntimeError if the key
    is missing to ensure explicit configuration.
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    return OpenAI(max_retries=MAX_RETRIES)


# # TODO
# def build_gemini_model(model_name: str = GEMINI_CLASSIFIER_MODEL) -> genai.Client:
#     """
#     Configure and return a Gemini model using GEMINI_API_KEY from the environment.

#     Loads `.env` for local development convenience. Raises RuntimeError if the key
#     is missing to ensure explicit configuration.
#     """
#     load_dotenv()

#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise RuntimeError("GEMINI_API_KEY is not set")

#     client = genai.Client()
#     model = client.get_model(model_name)
#     return model


@dataclass
class CostTracker:
    """Track token usage and estimated USD cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    input_rate: float = GPT4O_MINI_INPUT_RATE
    output_rate: float = GPT4O_MINI_OUTPUT_RATE

    # input_rate: float = FLASH_LITE_INPUT_RATE  # TODO
    # output_rate: float = FLASH_LITE_OUTPUT_RATE
    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)

    @property
    def cost_usd(self) -> float:
        return (self.input_tokens * self.input_rate) + (self.output_tokens * self.output_rate)

    def snapshot(self) -> dict[str, float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.cost_usd, 4),
        }


# Streaming and normalization ------------------------------------------------ #


def stream_jsonl(path: Path) -> Generator[dict[str, Any], None, None]:
    """Yield parsed JSON objects from a JSONL file, skipping malformed lines."""

    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON on line %s", idx)
                continue


def normalize_doc_type(value: str | None) -> str:
    """Normalize doc_type strings: strip, collapse whitespace, replace underscores, uppercase."""

    if value is None:
        return ""
    cleaned = value.replace("_", " ").strip()
    cleaned = " ".join(cleaned.split())
    return cleaned.upper().strip()


def safe_parse_date(raw_value: str | None) -> datetime | None:
    """Best-effort ISO date parser; returns None on failure."""

    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return None


# Instrument number heuristics ------------------------------------------------ #

# Accepts values like 2020-12345 with flexible trailing digit length.
YEAR_HYPHEN_PATTERN = re.compile(r"^\d{4}-\d{5,7}$")
# Accepts compact values like 202012345 without a separator.
YEAR_COMPACT_PATTERN = re.compile(r"^\d{4}\d{5,7}$")


def tokenize_mixed(text: str) -> tuple[str, str]:
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

    prefix_regex = re.escape(prefix) if prefix else ""

    def wrap(body: str) -> tuple[str, str]:
        """Return pattern (no anchors) and anchored regex with optional prefix."""

        full_body = f"{prefix_regex}{body}"
        return full_body, rf"^{full_body}$"

    # Year-prefixed with hyphen (separate buckets for 5/6/7 digits).
    if re.fullmatch(r"\d{4}-\d{5}", remainder):
        return wrap(r"\d{4}-\d{5}")
    if re.fullmatch(r"\d{4}-\d{6}", remainder):
        return wrap(r"\d{4}-\d{6}")
    if re.fullmatch(r"\d{4}-\d{7}", remainder):
        return wrap(r"\d{4}-\d{7}")

    # Year-prefixed compact (no separator).
    if re.fullmatch(r"\d{4}\d{5}", remainder):
        return wrap(r"\d{4}\d{5}")
    if re.fullmatch(r"\d{4}\d{6}", remainder):
        return wrap(r"\d{4}\d{6}")
    if re.fullmatch(r"\d{4}\d{7}", remainder):
        return wrap(r"\d{4}\d{7}")

    # Pure numeric.
    if remainder.isdigit():
        digit_len = len(remainder)
        return wrap(rf"\d{{{digit_len}}}")

    # Fallback to mixed tokenization.
    _, regex_body = tokenize_mixed(remainder)
    return wrap(regex_body)
