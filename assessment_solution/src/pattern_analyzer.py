"""Streaming county pattern analyzer for Task 1.

This module streams `data/nc_records_assessment.jsonl` line-by-line, aggregates
per-county statistics, and emits `assessment_solution/outputs/county_patterns.json`. It follows the
Dono project rules: defensive parsing, logging-based progress, and PEP8/typing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Iterable

from assessment_solution.src.utils import (
    DEFAULT_INPUT_JSONL,
    OUTPUT_FOLDER,
    infer_instrument_pattern,
    normalize_doc_type,
    safe_parse_date,
    stream_jsonl,
    tokenize_mixed,
)

LOGGER = logging.getLogger(__name__)
DATE_LOWER_BOUND = datetime(1900, 1, 1)

DEFAULT_OUTPUT_COUNTY_PATTERNS = OUTPUT_FOLDER.joinpath("county_patterns.json")


@dataclass
class DateRange:
    """Track date range and anomalies."""

    earliest: datetime = None
    latest: datetime = None
    anomalies: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "earliest": self.earliest.date().isoformat() if self.earliest else None,
            "latest": self.latest.date().isoformat() if self.latest else None,
            "anomalies": self.anomalies,
        }


@dataclass
class DocTypeStats:
    """Track doc_type distributions and cross-tab with doc_category."""

    counts: Counter[str] = field(default_factory=Counter)
    cross_tab: DefaultDict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def add(self, doc_type: str, doc_category: str) -> None:
        sanitized_category = (doc_category or "UNKNOWN").strip()
        self.counts[doc_type] += 1
        self.cross_tab[doc_type][sanitized_category] += 1


@dataclass
class FieldPattern:
    """Normalized shape for a single field (book/page)."""

    label: str
    regex: str
    is_numeric: bool
    numeric_value: int | None
    raw_value: str | None


@dataclass
class BookPagePatternSummary:
    """Aggregate pattern details for a (book, page) shape."""

    book_pattern: str
    book_regex: str
    page_pattern: str
    page_regex: str
    example: dict[str, str | None]
    count: int = 0
    book_numeric_min: int | None = None
    book_numeric_max: int | None = None
    page_numeric_min: int | None = None
    page_numeric_max: int | None = None

    def to_json(self, record_count: int) -> dict[str, Any]:
        percentage = 0.0
        if record_count:
            percentage = round((self.count / record_count) * 100, 3)
        return {
            "book_pattern": self.book_pattern,
            "book_regex": self.book_regex,
            "page_pattern": self.page_pattern,
            "page_regex": self.page_regex,
            "example": self.example,
            "count": self.count,
            "percentage": percentage,
        }


@dataclass
class NumericFieldStats:
    """Track numeric ranges and null/non-numeric rates for a field."""

    numeric_min: int | None = None
    numeric_max: int | None = None
    null_count: int = 0
    non_numeric_count: int = 0

    def update(self, pattern: FieldPattern) -> None:
        if pattern.raw_value is None:
            self.null_count += 1
            return
        if pattern.is_numeric and pattern.numeric_value is not None:
            self.numeric_min, self.numeric_max = update_bounds(
                self.numeric_min,
                self.numeric_max,
                pattern.numeric_value,
            )
            return

        self.non_numeric_count += 1

    def to_json(self, record_count: int) -> dict[str, Any]:
        denominator = record_count or 1  # avoid division by zero
        return {
            "numeric_min": self.numeric_min,
            "numeric_max": self.numeric_max,
            "null_pct": round((self.null_count / denominator) * 100, 3),
            "non_numeric_pct": round((self.non_numeric_count / denominator) * 100, 3),
        }


@dataclass
class BookPageStats:
    """Per-county statistics for book/page fields."""

    book: NumericFieldStats = field(default_factory=NumericFieldStats)
    page: NumericFieldStats = field(default_factory=NumericFieldStats)

    def to_json(self, record_count: int) -> dict[str, Any]:
        return {
            "book": self.book.to_json(record_count),
            "page": self.page.to_json(record_count),
        }


@dataclass
class InstrumentShape:
    """Canonicalized instrument shape attributes."""

    label: str
    raw: str
    regex_shape: str
    has_letters: bool
    has_year_prefix: bool
    separator: str
    digit_length_bucket: str
    prefix: str | None
    numeric_value: int | None
    year_value: int | None

    @property
    def pattern_label(self) -> str:
        """Human-readable pattern string."""

        return self.label

    @property
    def group_key(self) -> str:
        """Grouping key for aggregation."""

        return "|".join(
            [
                self.regex_shape,
                str(self.has_year_prefix),
                self.separator,
                self.digit_length_bucket,
            ]
        )


@dataclass
class InstrumentPatternSummary:
    """Aggregate stats for an instrument pattern."""

    pattern: str
    regex: str
    example: str
    count: int = 0
    numeric_min: int | None = None
    numeric_max: int | None = None
    year_min: int | None = None
    year_max: int | None = None

    def to_json(self, record_count: int) -> dict[str, Any]:
        percentage = 0.0
        if record_count:
            percentage = round((self.count / record_count) * 100, 3)

        return {
            "pattern": self.pattern,
            "regex": self.regex,
            "example": self.example,
            "count": self.count,
            "percentage": percentage,
        }


@dataclass
class InstrumentCoverageStats:
    """Coverage percentages across instrument shapes."""

    numeric_only: int = 0
    year_prefixed: int = 0
    alpha_numeric: int = 0
    null_count: int = 0

    def update(self, shape: InstrumentShape | None) -> None:
        if shape is None:
            self.null_count += 1
            return

        if shape.has_year_prefix:
            self.year_prefixed += 1
            return

        if shape.has_letters:
            self.alpha_numeric += 1
            return

        self.numeric_only += 1

    def to_json(self, record_count: int) -> dict[str, float]:
        denominator = record_count or 1
        return {
            "numeric_only_pct": round((self.numeric_only / denominator) * 100, 3),
            "year_prefixed_pct": round((self.year_prefixed / denominator) * 100, 3),
            "alpha_numeric_pct": round((self.alpha_numeric / denominator) * 100, 3),
            "null_pct": round((self.null_count / denominator) * 100, 3),
        }


@dataclass
class CountyStats:
    """Aggregate all per-county statistics."""

    record_count: int = 0
    instrument_patterns: dict[str, InstrumentPatternSummary] = field(default_factory=dict)
    instrument_stats: InstrumentCoverageStats = field(default_factory=InstrumentCoverageStats)
    book_page_patterns: dict[str, BookPagePatternSummary] = field(default_factory=dict)
    book_page_stats: BookPageStats = field(default_factory=BookPageStats)
    dates: DateRange = field(default_factory=DateRange)
    doc_types: DocTypeStats = field(default_factory=DocTypeStats)

    def to_json(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "instrument_patterns": [
                pattern.to_json(self.record_count)
                for pattern in sorted(
                    self.instrument_patterns.values(),
                    key=lambda entry: (-entry.count, entry.pattern, entry.regex),
                )
            ],
            "book_patterns": [
                pattern.to_json(self.record_count)
                for pattern in sorted(
                    self.book_page_patterns.values(),
                    key=lambda entry: (-entry.count, entry.book_pattern, entry.page_pattern),
                )
            ],
            "date_range": self.dates.to_json(),
            "doc_type_distribution": dict(self.doc_types.counts.most_common(10)),
            "unique_doc_types": len(self.doc_types.counts),
            # "doc_category_distribution": doc_category_distribution(self.doc_types),
            # "doc_type_to_category": doc_type_to_category(self.doc_types),  # TODO
        }


# Update helpers ------------------------------------------------------------- #


def update_bounds(
        current_min: int | None,
        current_max: int | None,
        candidate: int | None,
) -> tuple[int | None, int | None]:
    """Return updated numeric bounds after considering a candidate value."""

    if candidate is None:
        return current_min, current_max

    if current_min is None or candidate < current_min:
        current_min = candidate
    if current_max is None or candidate > current_max:
        current_max = candidate

    return current_min, current_max


def numeric_if_available(pattern: FieldPattern) -> int | None:
    """Convenience helper to pull numeric value only when the pattern is numeric."""

    return pattern.numeric_value if pattern.is_numeric else None


def update_instrument(county_stats: CountyStats, raw_value: str = None) -> None:
    """Update instrument pattern counters for a county."""

    normalized = canonicalize_instrument(raw_value)
    if not normalized:
        county_stats.instrument_stats.update(None)
        return

    shape = infer_instrument_shape(normalized)
    county_stats.instrument_stats.update(shape)

    bucket = county_stats.instrument_patterns.get(shape.group_key)
    if bucket is None:
        county_stats.instrument_patterns[shape.group_key] = InstrumentPatternSummary(
            pattern=shape.pattern_label,
            regex=shape.regex_shape,
            example=normalized,
            count=1,
            numeric_min=shape.numeric_value,
            numeric_max=shape.numeric_value,
            year_min=shape.year_value,
            year_max=shape.year_value,
        )
    else:
        bucket.count += 1
        bucket.numeric_min, bucket.numeric_max = update_bounds(
            bucket.numeric_min,
            bucket.numeric_max,
            shape.numeric_value,
        )
        bucket.year_min, bucket.year_max = update_bounds(
            bucket.year_min,
            bucket.year_max,
            shape.year_value,
        )


def canonicalize_instrument(value: Any) -> str | None:
    """Normalize instrument strings: trim, collapse whitespace, uppercase, preserve separators."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = " ".join(value.strip().split())
    else:
        cleaned = " ".join(str(value).strip().split())

    cleaned = cleaned.upper()
    return cleaned or None


def infer_instrument_shape(value: str) -> InstrumentShape:
    """Derive canonical shape attributes for an instrument number."""

    pattern_label, regex_shape = infer_instrument_pattern(value)

    digits_total = sum(1 for char in value if char.isdigit())
    has_letters = any(char.isalpha() for char in value)
    separator = "none"
    for candidate in ["-", "/", "_"]:
        if candidate in value:
            separator = candidate
            break

    has_year_prefix = bool(re.match(r"^(19|20)\d{2}", value))
    digit_length_bucket = (
        "short" if digits_total <= 6 else "medium" if digits_total <= 10 else "long"
    )
    prefix_match = re.match(r"^[A-Za-z]+", value)
    prefix = prefix_match.group(0) if prefix_match else None
    numeric_value = int(value) if value.isdigit() else None
    year_value = int(value[:4]) if has_year_prefix else None

    return InstrumentShape(
        label=pattern_label,
        raw=value,
        regex_shape=regex_shape,
        has_letters=has_letters,
        has_year_prefix=has_year_prefix,
        separator=separator,
        digit_length_bucket=digit_length_bucket,
        prefix=prefix,
        numeric_value=numeric_value,
        year_value=year_value,
    )


def normalize_book_page_value(value: Any) -> str | None:
    """Normalize raw book/page values to stripped strings or None."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        cleaned = str(value).strip()
    return cleaned or None


def infer_field_pattern(value: str | None) -> FieldPattern:
    """Infer a shape label/regex for a single field and capture numeric info."""

    if not value:
        return FieldPattern(
            label="missing",
            regex=r"^$",
            is_numeric=False,
            numeric_value=None,
            raw_value=None,
        )

    if value.isdigit():
        digit_len = len(value)
        return FieldPattern(
            label=f"digits-{digit_len}",
            regex=rf"^\d{{{digit_len}}}$",
            is_numeric=True,
            numeric_value=int(value),
            raw_value=value,
        )

    label, regex_body = tokenize_mixed(value)
    return FieldPattern(
        label=label,
        regex=rf"^{regex_body}$",
        is_numeric=False,
        numeric_value=None,
        raw_value=value,
    )


def coarsen_field_pattern(pattern: FieldPattern) -> tuple[str, str]:
    """Reduce field pattern into coarse buckets used for output."""

    if pattern.raw_value is None:
        return "missing", r"^$"

    if pattern.is_numeric:
        return "digits_only", r"^\d+$"

    raw = pattern.raw_value or ""
    if re.fullmatch(r"[A-Za-z0-9]+", raw):
        return "alnum", r"^[A-Za-z0-9]+$"

    return "mixed", pattern.regex


def update_book_page_patterns(
        stats: CountyStats,
        book_value_raw: Any,
        page_value_raw: Any,
) -> None:
    """Aggregate per-county book/page shape patterns and field stats."""

    book_value = normalize_book_page_value(book_value_raw)
    page_value = normalize_book_page_value(page_value_raw)

    book_pattern = infer_field_pattern(book_value)
    page_pattern = infer_field_pattern(page_value)
    book_numeric = numeric_if_available(book_pattern)
    page_numeric = numeric_if_available(page_pattern)

    stats.book_page_stats.book.update(book_pattern)
    stats.book_page_stats.page.update(page_pattern)

    book_label, book_regex = coarsen_field_pattern(book_pattern)
    page_label, page_regex = coarsen_field_pattern(page_pattern)

    pattern_key = f"{book_label}|{page_label}"
    bucket = stats.book_page_patterns.get(pattern_key)
    if bucket is None:
        stats.book_page_patterns[pattern_key] = BookPagePatternSummary(
            book_pattern=book_label,
            book_regex=book_regex,
            page_pattern=page_label,
            page_regex=page_regex,
            example={"book": book_value, "page": page_value},
            count=1,
            book_numeric_min=book_numeric,
            book_numeric_max=book_numeric,
            page_numeric_min=page_numeric,
            page_numeric_max=page_numeric,
        )
    else:
        bucket.count += 1
        bucket.book_numeric_min, bucket.book_numeric_max = update_bounds(
            bucket.book_numeric_min,
            bucket.book_numeric_max,
            book_numeric,
        )
        bucket.page_numeric_min, bucket.page_numeric_max = update_bounds(
            bucket.page_numeric_min,
            bucket.page_numeric_max,
            page_numeric,
        )


def update_dates(date_range: DateRange, now: datetime, raw_date: str = None) -> None:
    """Update earliest/latest dates and anomalies."""

    if not raw_date:
        return
    parsed = safe_parse_date(raw_date)
    if parsed is None:
        date_range.anomalies.append(f"parse_error:{raw_date}")
        return

    if date_range.earliest is None or parsed < date_range.earliest:
        date_range.earliest = parsed
    if date_range.latest is None or parsed > date_range.latest:
        date_range.latest = parsed

    if parsed < DATE_LOWER_BOUND or parsed > now:
        date_range.anomalies.append(raw_date)


def update_doc_types(
        doc_stats: DocTypeStats,
        doc_type: str = None,
        doc_category: str = None,
) -> None:
    """Track doc_type and cross-tab information."""

    normalized_doc_type = normalize_doc_type(doc_type)
    doc_stats.add(normalized_doc_type, doc_category or "")




# Ingestion and post-processing --------------------------------------------- #


def process_record(
        record: dict[str, Any],
        county_stats: dict[str, CountyStats],
        now: datetime,
) -> None:
    """Process a single JSON record into county aggregates."""

    county = (record.get("county") or "").strip().lower()
    if not county:
        return

    stats = county_stats.setdefault(county, CountyStats())
    stats.record_count += 1

    update_book_page_patterns(stats, record.get("book"), record.get("page"))

    raw_instrument = record.get("instrument_number")
    instrument_value = (
        raw_instrument if (raw_instrument is not None and str(raw_instrument).strip()) else None
    )
    update_instrument(stats, instrument_value)
    update_dates(stats.dates, now, record.get("date"))
    update_doc_types(stats.doc_types, record.get("doc_type"), record.get("doc_category"))


def build_output(county_stats: dict[str, CountyStats]) -> dict[str, Any]:
    """Convert aggregated stats into the final JSON-ready structure."""

    output: dict[str, Any] = {}
    for county, stats in sorted(county_stats.items()):
        output[county] = stats.to_json()
    return output


# CLI ----------------------------------------------------------------------- #
def parse_args(argv: Iterable[str] = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Stream and summarize county record patterns.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help="Path to JSONL file with county records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_COUNTY_PATTERNS,
        help="Path to write summarized JSON output.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Log progress every N records.",
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    """Configure module-level logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def stream_file(input_path: Path, log_interval: int) -> dict[str, CountyStats]:
    """Stream the input JSONL file and aggregate per-county stats."""

    county_stats: dict[str, CountyStats] = {}
    now = datetime.now()
    processed_records = 0

    for _, record in enumerate(stream_jsonl(input_path), start=1):
        process_record(record, county_stats, now)
        processed_records += 1
        if log_interval and processed_records % log_interval == 0:
            LOGGER.info("Processed %s records", processed_records)

    LOGGER.info(
        "Finished streaming %s total records",
        processed_records,
    )
    return county_stats


def write_output(output_path: Path, content: dict[str, Any]) -> None:
    """Write the final JSON output with sorted keys."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, sort_keys=True)
    LOGGER.info("Wrote summary to %s", output_path)


def main(argv: Iterable[str] = None) -> None:
    """Entrypoint for the streaming pattern analyzer."""

    args = parse_args(argv)
    setup_logging()
    if not args.input.exists():
        LOGGER.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    county_stats = stream_file(args.input, args.log_interval)
    output_content = build_output(county_stats)
    write_output(args.output, output_content)


if __name__ == "__main__":
    main()
