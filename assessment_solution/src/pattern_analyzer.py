"""Streaming county pattern analyzer for Task 1.

This module streams `data/nc_records_assessment.jsonl` line-by-line, aggregates
per-county statistics, and emits `assessment_solution/outputs/county_patterns.json`. It follows the
Dono project rules: defensive parsing, logging-based progress, and PEP8/typing.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Iterable

from assessment_solution.src.utils import infer_instrument_pattern

LOGGER = logging.getLogger(__name__)
DATE_LOWER_BOUND = datetime(1900, 1, 1)


@dataclass
class InstrumentPatternInfo:
    """Track a specific instrument pattern."""

    pattern: str
    regex: str
    example: str
    count: int = 0

    def to_json(self, record_count: int) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "regex": self.regex,
            "example": self.example,
            "count": self.count,
            "percentage": round((self.count / record_count) * 100, 3),
        }


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
        sanitized_type = (doc_type or "").strip() or "UNKNOWN"
        sanitized_category = (doc_category or "").strip() or "unknown"
        self.counts[sanitized_type] += 1
        self.cross_tab[sanitized_type][sanitized_category] += 1


@dataclass
class BookPageIndex:
    """Track book/page information."""

    book: str | None = None
    page: str | None = None

    @property
    def instrument_pattern(self) -> str | None:
        if self.book is not None and self.page is not None:
            return f"pb{self.book}{self.page}"

        return None

    def to_json(self) -> dict[str, Any]:
        return {"book": self.book, "page": self.page}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BookPageIndex:
        if record.get("book") is None and record.get("page") is None:
            return cls()
        book = record.get("book", None)
        page = record.get("page", None)
        return cls(book=book, page=page)


@dataclass
class CountyStats:
    """Aggregate all per-county statistics."""

    record_count: int = 0
    instrument_patterns: dict[str, InstrumentPatternInfo] = field(default_factory=dict)
    book_page_index: dict[str, BookPageIndex] = field(default_factory=dict)
    dates: DateRange = field(default_factory=DateRange)
    doc_types: DocTypeStats = field(default_factory=DocTypeStats)

    def to_json(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "instrument_patterns": [
                pattern.to_json(self.record_count)
                for pattern in sorted(
                    self.instrument_patterns.values(),
                    key=lambda entry: (-entry.count, entry.pattern),
                )
            ],
            "book_patterns": [bp_index.to_json() for bp_index in self.book_page_index.values()],
            "date_range": self.dates.to_json(),
            "doc_type_distribution": dict(self.doc_types.counts.most_common(10)),
            "unique_doc_types": len(self.doc_types.counts),
            # "doc_type_category_crosstab": {
            #     k: v.to_json() for k, v in self.doc_types.cross_tab.items()
            # },  # TODO
        }


# Update helpers ------------------------------------------------------------- #


def update_instrument(county_stats: CountyStats, raw_value: str = None) -> None:
    """Update instrument pattern counters for a county."""

    if raw_value is None:
        return
    pattern_label, regex = infer_instrument_pattern(raw_value)
    bucket = county_stats.instrument_patterns.get(pattern_label)
    if bucket is None:
        county_stats.instrument_patterns[pattern_label] = InstrumentPatternInfo(
            pattern=pattern_label,
            regex=regex,
            example=raw_value,
            count=1,
        )
    else:
        bucket.count += 1


def update_book_page_index(stats: CountyStats, bp_index: BookPageIndex) -> None:
    """Update book/page index."""

    if bp_index.instrument_pattern is None:
        return

    if bp_index.instrument_pattern in stats.book_page_index:
        return

    stats.book_page_index[bp_index.instrument_pattern] = bp_index


def update_dates(date_range: DateRange, now: datetime, raw_date: str = None) -> None:
    """Update earliest/latest dates and anomalies."""

    if not raw_date:
        return
    try:
        parsed = datetime.fromisoformat(raw_date)
    except ValueError:
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

    doc_stats.add(doc_type or "", doc_category or "")


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

    raw_instrument = record.get("instrument_number")
    instrument_value = (
        raw_instrument if (raw_instrument is not None and str(raw_instrument).strip()) else None
    )
    if not instrument_value:
        bp_index = BookPageIndex.from_record(record)
        update_book_page_index(stats, bp_index)
        instrument_value = bp_index.instrument_pattern

    update_instrument(stats, instrument_value)
    update_dates(stats.dates, now, record.get("date"))
    update_doc_types(stats.doc_types, record.get("doc_type"), record.get("doc_category"))


# TODO
# def _finalize_date_stats(stats: DateRange) -> dict[str, Any]:
#     """Convert DateStats to serializable dict."""

#     return {
#         "earliest": stats.earliest.date().isoformat() if stats.earliest else None,
#         "latest": stats.latest.date().isoformat() if stats.latest else None,
#         "anomalies": stats.anomalies,
#     }


# def _finalize_instrument_patterns(stats: CountyStats) -> list[dict[str, Any]]:
#     """Build sorted instrument pattern summaries."""

#     if stats.record_count == 0:
#         return []
#     patterns = sorted(
#         stats.instrument_patterns.values(),
#         key=lambda entry: entry.count,
#         reverse=True,
#     )
#     return [entry.to_json(stats.record_count) for entry in patterns]


# def _finalize_doc_types(
#     doc_stats: DocTypeStats,
# ) -> tuple[dict[str, int], int, dict[str, dict[str, int]]]:
#     """Return doc_type top10, unique count, and cross-tab."""

#     top_10 = dict(doc_stats.counts.most_common(10))
#     unique_count = len(doc_stats.counts)
#     cross_tab = {
#         doc_type: dict(sorted(categories.items()))
#         for doc_type, categories in doc_stats.cross_tab.items()
#     }
#     return top_10, unique_count, cross_tab


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
        default=Path("data/nc_records_assessment.jsonl"),
        help="Path to JSONL file with county records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assessment_solution/outputs/county_patterns.json"),
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

    with input_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON on line %s", idx)
                continue
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
    county_stats = stream_file(args.input, args.log_interval)
    output_content = build_output(county_stats)
    write_output(args.output, output_content)


if __name__ == "__main__":
    main()
