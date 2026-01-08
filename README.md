# Assessment Solution

## Setup
- Python >= 3.9
- Install tooling: `pip install -r assessment_solution/requirements.txt`

## How to run scripts
- Task 1 (county patterns): `python -m assessment_solution.src.pattern_analyzer --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/county_patterns.json --log-interval 1000`
- Task 2 (doc_type classification): `python -m assessment_solution.src.llm_classifier --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/doc_type_mapping.json` _(status: to be implemented; see approach below)_
- Bonus (connections/PDF/story): `python -m assessment_solution.src.connection_finder`, `python -m assessment_solution.src.pdf_analyzer`, `python -m assessment_solution.src.story_generator` _(status: planned; see bonus plan)_
- Style checks: `ruff check assessment_solution` and `black --check assessment_solution`; combined helper (once added): `bash assessment_solution/scripts/check_style.sh`

## Dependencies and requirements
- Runtime: standard library only (argparse, json, logging, datetime, pathlib, dataclasses, collections, re).
- Dev tooling: `black==24.10.0`, `ruff==0.6.7` (pinned in `assessment_solution/requirements.txt`).

## Task 1 (county pattern analysis)
- Streams JSONL line-by-line (no full-file loads), aggregates per-county instrument patterns, book/page hints, date ranges (earliest/latest/anomalies), and doc_type distributions, then writes `assessment_solution/outputs/county_patterns.json`.
- Logging-based progress every N records (configurable).

## Task 2 (LLM-assisted classification) – approach, prompts, validation, cost, trade-offs
- Approach: normalize doc_type strings (strip/uppercase/collapse spaces), take top ~200 frequent + ~50 long-tail examples, dedupe, and batch 20 items per LLM call. Allowed categories: `SALE_DEED`, `MORTGAGE`, `DEED_OF_TRUST`, `RELEASE`, `LIEN`, `PLAT`, `EASEMENT`, `LEASE`, `MISC`. Apply returned mapping locally to full JSONL (no per-record LLM calls).
- Prompt used (sample):
  - System: “You map noisy county recording doc_type strings to one of the allowed categories. Respond with JSON only.”
  - User: provides allowed categories list and JSON array of doc_type candidates; few-shot examples like `{"QUIT CLAIM DEED": "SALE_DEED", "DEED OF TRUST": "DEED_OF_TRUST", "SUBORDINATION OF LIEN": "LIEN"}`; request strict JSON mapping `{ "raw_doc_type": "STANDARD_CATEGORY" }` and “return ONLY valid JSON.”
- Validation methodology:
  - Parse JSON; reject and retry on errors.
  - Enforce outputs in allowed category set; rerun invalid entries.
  - Frequency-weighted spot checks (top 20) plus random rare samples; inspect mismatches and, if needed, patch mapping manually before applying.
  - Final mapping applied deterministically across the dataset; outputs stored in `assessment_solution/outputs/doc_type_mapping.json`.
- Cost breakdown (estimate):
  - ~250 doc_type strings, batched 20 per call ⇒ ~13 calls.
  - Per call ~300 input + ~200 output tokens ⇒ ~6.5k input + ~4.3k output total.
  - Using a cheap model (e.g., gpt-4o-mini pricing scale), estimated cost ≈ $0.05–$0.15 end-to-end.
- Trade-offs:
  - Cheap model reduces cost/latency but may miss nuanced types; mitigated with few-shot examples, normalization, and validation checks.
  - Batching improves throughput but risks context bleed; keep batches small (≤20) and include explicit allowed-category reminder each call.

## Bonus task plan (connections, PDFs, storytelling)
- Prompts (planned): concise JSON-only extraction prompts for PDF-derived snippets (parties, dates, amounts, addresses), followed by a summarization prompt to explain connections across 2–4 documents per group.
- Cost approach (planned): prefer local text extraction first; only LLM on targeted sections; batch related docs; reuse entities to minimize calls.
- Insights & scaling (planned): group by shared parties/dates/instrument patterns; cache parsed PDFs; keep connection discovery deterministic before narration.

## Assumptions
- Data lives under `data/nc_records_assessment.jsonl` and PDFs under `data/records/<county>/<instrument>.pdf`.
- Outputs are written under `assessment_solution/outputs` and can be regenerated locally.
- LLM access/credentials are available externally (not stored in repo).

## Relevant notes
- Code targets Python 3.9, PEP-8, type hints, defensive parsing, and logging (no bare prints except CLI-friendly summaries).
- JSON outputs should be stable (sorted keys where relevant).
