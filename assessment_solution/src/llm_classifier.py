"""Task 2: LLM-assisted document type classification.

Streams the dataset once, extracts unique normalized doc_type strings with
frequencies, samples to ~95–98% coverage (plus key rare abbreviations), and
classifies the sampled set with OpenAI `gpt-4o-mini`. Outputs a mapping JSON
# classifies the sampled set with Gemini 2.5 Flash-Lite. Outputs a mapping JSON  # TODO
used to classify all records locally without per-record LLM calls.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import tiktoken
from openai import APIConnectionError, APIError, RateLimitError

from assessment_solution.src.utils import (
    DEFAULT_INPUT_JSONL,
    OUTPUT_FOLDER,
    CostTracker,
    build_openai_client,
    # build_gemini_model,  # TODO
    normalize_doc_type,
    parse_llm_json,
    stream_jsonl,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_DOC_TYPE_MAPPING = OUTPUT_FOLDER.joinpath("doc_type_mapping.json")

ALLOWED_CATEGORIES = [
    "SALE_DEED",
    "MORTGAGE",
    "DEED_OF_TRUST",
    "RELEASE",
    "LIEN",
    "PLAT",
    "EASEMENT",
    "LEASE",
    "MISC",
]

COMMON_OFFENDERS = [
    "DEED",
    "DEEDS",
    "SATISFACTION",
    "SATISF",
    "SAT",
    "ASSIGNMENT",
    "ASSIGN",
    "UCC",
    "CANCELLATION",
    "CANCEL",
    "NOTICE",
    "AMENDMENT",
    "AMEND",
    "COVENANT",
    "COVENANTS",
    "RESTRICTIVE",
    "RESTRICTIVE COVENANT",
    "RESTRICTIVE COVENANTS",
]


DEFAULT_MODEL = "gpt-4o-mini"  # TODO
COVERAGE_TARGET = 0.98
BATCH_SIZE = 15  # TODO: 20
TEMPERATURE = 0.1  # lower temperature to reduce raw echo drift
TOP_P = 0.90
STOP_SEQUENCES: list[str] = []


# TODO
# - If unsure, choose MISC.


# CLI helpers ----------------------------------------------------------------- #


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="LLM-assisted doc_type classifier using OpenAI chat completions."
        # description="LLM-assisted doc_type classifier using Gemini 2.5 Flash-Lite."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help="Path to JSONL input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DOC_TYPE_MAPPING,
        help="Path to write the doc_type mapping JSON.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model name to use (default: gpt-4o-mini).",
        # help="Gemini model name to use.",  # TODO
    )
    parser.add_argument(
        "--coverage-target",
        type=float,
        default=COVERAGE_TARGET,
        help="Fraction of record coverage to achieve with sampled doc_types.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of doc_type strings per LLM call.",
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    """Configure logging for CLI usage."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# Core logic ------------------------------------------------------------------ #


def collect_doc_type_counts(input_path: Path) -> Counter[str]:
    """Stream the dataset and count normalized doc_type frequencies."""

    counts: Counter[str] = Counter()
    for record in stream_jsonl(input_path):
        normalized = normalize_doc_type(record.get("doc_type"))
        if not normalized:
            continue  # skip empty/whitespace doc_types to avoid blank LLM inputs
        counts[normalized] += 1
    return counts


def sample_doc_types(
    counts: Counter[str],
    coverage_target: float,
) -> list[str]:
    """
    Sample doc_types by frequency until the desired coverage is met, while
    ensuring key abbreviations/semantic types are included.
    """

    total = sum(counts.values())
    if total == 0:
        return []

    sorted_items = counts.most_common()
    selected: list[str] = []
    covered = 0
    coverage_threshold = total * coverage_target

    for doc_type, freq in sorted_items:
        selected.append(doc_type)
        covered += freq
        if covered >= coverage_threshold:
            break

    # Ensure abbreviations / important rare types are considered if present.
    must_include = {"MTG", "DOT", "REL", "LIEN", "PLAT", "EASEMENT"}
    for candidate in must_include:
        if candidate in counts and candidate not in selected:
            selected.append(candidate)

    return selected


SYSTEM_PROMPT = """
You are a classifier specialized in North Carolina property records document types.

Allowed categories (choose exactly ONE per input):
{categories}

Category guidance (choose best fit, never echo the input):
- SALE_DEED: deeds, bargain & sale, quitclaim, warranty, trustee or sheriff deeds.
- MORTGAGE: mortgages labeled MTG or MTGE.
- DEED_OF_TRUST: deeds of trust, DOT, assignments/mods/subordinations of DOT, foreclosure notices.
- RELEASE: satisfactions, cancellations, partial releases, reconveyance.
- LIEN: UCC filings, liens (tax/mechanic/JT), claims of lien.
- PLAT: plats, maps, re-plats, condo maps, surveys, dedications.
- EASEMENT: easements, right-of-way.
- LEASE: leases or memoranda of lease.
- MISC: everything else (covenants/restrictions/HOA docs, affidavits, generic “notice”/“amendment”, ambiguous strings).

Rules:
1) Return exactly one category string per item; never output raw doc_type text.
2) If multiple categories seem plausible, pick the single best fit using the guidance.
3) If unsure, choose MISC.

Example:
{example}
"""


def build_system_prompt(categories: list[str] = ALLOWED_CATEGORIES) -> str:
    """Render the classification system prompt for a batch of doc type strings."""

    categories_str = "\n".join([f"- {category}" for category in categories])
    example = json.dumps(
        {
            "DOT": "DEED_OF_TRUST",
            # "SUBORDINATION": "DEED_OF_TRUST",
            "SUBSTITUTE TRUSTEE": "DEED_OF_TRUST",
            "FORECLOSURE NOTICE": "DEED_OF_TRUST",
            "FC": "DEED_OF_TRUST",
            "AFDVT": "DEED_OF_TRUST",
            "ASGMT": "DEED_OF_TRUST",
            "C-SAT": "RELEASE",
            "REL": "RELEASE",
            "DEDICATION": "PLAT",
            "JUDGMENT": "LIEN",
            "UCC": "LIEN",
            "UCC FINANCING STATEMENT AMENDMENT": "LIEN",
            "REST": "RELEASE",
            "Q C D": "SALE_DEED",
            "D & REL": "SALE_DEED", 
            "SUB D": "SALE_DEED",
            "CORR D": "SALE_DEED",
            # "DEED": "SALE_DEED",
            # "TRUSTEES DEED": "SALE_DEED",
            "SUBSTITUTE TRUSTEE'S DEED": "SALE_DEED",
            "REL D": "SALE_DEED",
            "COVENANT/RESTRICTIVE COVENANTS": "MISC",
            # "AMENDMENT": "MISC",
            # "NOTICE": "MISC",
        }
    )  # TODO: refine illustrative examples as needed

    return SYSTEM_PROMPT.format(categories=categories_str, example=example)


USER_PROMPT = """
You are classifying real estate document types.

Rules:
- Be conservative.
- Return valid JSON only.
- Return only one label for each input.
- Do NOT echo the input text or invent new labels.
- No explanations.

Input:
{payload}

Output a single JSON object where:
- Each key is exactly one of the provided input strings.
- Each value is one of the categories above.
- Do not add or remove keys. Do not return arrays. No explanations.
"""
BOOST = 8.0  # stronger bias toward allowed labels
PENALTY = -6.0  # slightly softer penalty to reduce clashes with sub-tokens


def build_logit_bias(
    model_name: str,
    categories: list[str] = ALLOWED_CATEGORIES,
    offenders: list[str] = COMMON_OFFENDERS,
    boost: float = BOOST,
    penalty: float = PENALTY,
) -> dict[int, float]:
    """
    Optionally bias generation toward allowed category tokens.
    Falls back gracefully when tokenization helpers are unavailable.
    """

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    bias: dict[int, float] = {}
    for category in categories:
        for token_id in encoding.encode(category):
            bias[token_id] = boost

    for offender in offenders:
        for token_id in encoding.encode(offender):
            # Do not override positive boosts for allowed categories.
            if token_id not in bias:
                bias[token_id] = penalty

    # Nudge JSON structure tokens without over-constraining.
    for token in ['"', "{", "}", ":", ","]:
        for token_id in encoding.encode(token):
            bias.setdefault(token_id, boost / 2)

    return bias


def validate_mapping(
    requested: list[str],
    mapping: dict[str, str],
) -> dict[str, str]:
    """Validate and coerce an LLM mapping to allowed categories."""

    result: dict[str, str] = {}
    for item in requested:
        category = mapping.get(item)
        if category is None:
            LOGGER.warning("LLM response missing key; defaulting %s to UNKNOWN", item)
            category = "UNKNOWN"  # TODO: MISC

        if category not in ALLOWED_CATEGORIES:
            LOGGER.warning("LLM response with invalid category; defaulting %s to UNKNOWN", category)

            LOGGER.warning(f"{item} {category} not in ALLOWED_CATEGORIES")
            category = "UNKNOWN"  # TODO: "MISC"
        result[item] = category
    return result


def summarize_category_percentages(
    mapping: dict[str, str],
) -> dict[str, dict[str, float]]:
    """
    Compute count and percentage for each allowed category across a sample
    mapping. Percentages are relative to all mapped sample items.
    """

    total = len(mapping)
    counts = Counter(mapping.values())
    counts.setdefault("UNKNOWN", 0)
    summary: dict[str, dict[str, float]] = {}

    for category in ALLOWED_CATEGORIES + ["UNKNOWN"]:
        count = counts.get(category)
        percentage = (count / total * 100.0) if total else 0.0
        summary[category] = {"count": count, "percentage": percentage}

    return summary


def classify_batches(
    items: list[str],
    model_name: str,
    batch_size: int,
    cost_tracker: CostTracker,
) -> dict[str, str]:
    """Classify doc_type strings in batches via OpenAI Chat Completions."""

    client = build_openai_client()
    # model = build_gemini_model(model_name)  # TODO
    final_mapping: dict[str, str] = {}

    logit_bias = build_logit_bias(model_name)

    for idx in range(0, len(items), batch_size):
        batch = items[idx : idx + batch_size]
        # response = model.generate_content(prompt)  # TODO
        # usage = getattr(response, "usage_metadata", None)  # TODO

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": USER_PROMPT.format(payload=json.dumps(batch))},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                # stop=STOP_SEQUENCES or None,  # TODO
                logit_bias=logit_bias,
                response_format={"type": "json_object"},
            )

        except RateLimitError as e:
            LOGGER.error(f"OpenAI API request exceeded rate limit on batch {idx}: {e}")
            continue

        except APIConnectionError as e:
            LOGGER.error(f"Failed to connect to OpenAI API on batch {idx}: {e}")
            continue

        except APIError as e:
            LOGGER.error(f"OpenAI API returned an API Error on batch {idx}: {e}")
            continue

        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)  # TODO default
            cost_tracker.add(input_tokens, output_tokens)

        try:
            content = response.choices[0].message.content
            raw_mapping = parse_llm_json(content)
        except (json.JSONDecodeError, TypeError) as err:
            LOGGER.warning("Failed to parse LLM JSON for batch starting at %s: %s", idx, err)
            raw_mapping = {}

        validated = validate_mapping(batch, raw_mapping)
        final_mapping.update(validated)

    return final_mapping


def build_full_mapping(
    counts: Counter[str],
    sampled_mapping: dict[str, str],
) -> dict[str, str]:
    """Default non-sampled doc_types to MISC and merge sampled classifications."""

    full_mapping: dict[str, str] = {}
    for doc_type in counts:
        # if not doc_type.strip():  # TODO
        # continue  # defensive: ignore empty keys if they slip through
        if doc_type in sampled_mapping:
            full_mapping[doc_type] = sampled_mapping[doc_type]
        else:
            full_mapping[doc_type] = "MISC"
    return dict(sorted(full_mapping.items()))


def write_output(path: Path, mapping: dict[str, str]) -> None:
    """Write mapping to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, sort_keys=True)
    LOGGER.info("Wrote mapping to %s", path)


def main(argv: Iterable[str] | None = None) -> None:
    """Entrypoint for the doc_type classifier."""

    args = parse_args(argv)
    setup_logging()

    LOGGER.info("Collecting doc_type counts from %s", args.input)
    counts = collect_doc_type_counts(args.input)
    total_records = sum(counts.values())
    LOGGER.info("Found %s doc_type values across records", total_records)

    sampled = sample_doc_types(counts, args.coverage_target)
    LOGGER.info(
        "Sampling %s unique doc_types targeting %.2f coverage",
        len(sampled),
        args.coverage_target,
    )

    tracker = CostTracker()
    sampled_mapping = classify_batches(sampled, args.model, args.batch_size, tracker)
    mapping = build_full_mapping(counts, sampled_mapping)

    coverage = sum(counts.get(item, 0) for item in sampled) / total_records if total_records else 0
    sampled_stats = summarize_category_percentages(sampled_mapping)
    LOGGER.info(
        "Coverage achieved: %.3f; LLM calls: %s; cost snapshot: %s",
        coverage,
        (len(sampled) + args.batch_size - 1) // args.batch_size,
        tracker.snapshot(),
    )
    LOGGER.info(
        "Sampled category distribution (count, percentage of samples): %s",
        {
            cat: (stats["count"], f"{stats['percentage']:.2f}%")
            for cat, stats in sampled_stats.items()
        },
    )

    write_output(args.output, mapping)


if __name__ == "__main__":
    main()
