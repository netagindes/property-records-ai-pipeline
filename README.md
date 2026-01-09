# Assessment Solution

## Setup
- Python >= 3.9
- Install tooling: `pip install -r assessment_solution/requirements.txt`
- Add `.env` with `OPENAI_API_KEY=<your_key>` for Task 2.

## How to run scripts
- Task 1 (county patterns): `python -m assessment_solution.src.pattern_analyzer --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/county_patterns.json --log-interval 1000`
- Task 2 (doc_type classification): `python -m assessment_solution.src.llm_classifier --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/doc_type_mapping.json`
- Bonus (connections/PDF/story): `python -m assessment_solution.src.connection_finder`, `python -m assessment_solution.src.pdf_analyzer`, `python -m assessment_solution.src.story_generator` _(status: planned; see bonus plan)_
- Style checks: `ruff check assessment_solution` and `black --check assessment_solution`; combined helper (once added): `bash assessment_solution/scripts/check_style.sh`

## Dependencies and requirements
- Runtime: standard library plus `openai` and `python-dotenv` (for OpenAI access and `.env` loading).
- Dev tooling: `black==24.10.0`, `ruff==0.6.7` (pinned in `assessment_solution/requirements.txt`).

## Task 1 (county pattern analysis)
- Streams JSONL line-by-line (no full-file loads), aggregates per-county instrument patterns, book/page hints, date ranges (earliest/latest/anomalies), and doc_type distributions, then writes `assessment_solution/outputs/county_patterns.json`.
- Logging-based progress every N records (configurable).

## Task 2 (LLM-assisted classification) – approach, prompts, validation, cost, trade-offs
- Approach: stream once, normalize doc_type strings (strip/uppercase/collapse spaces, underscores→spaces), count frequencies, and classify unique values only (cuts LLM volume ~99%). Sample by descending frequency until ~95–98% record coverage, then add key abbreviations/rare semantic types (MTG, DOT, REL, LIEN, PLAT, EASEMENT). Non-sampled values default to `MISC`.
- Prompt (batch of up to 20 doc_types):  
  ```
  You are classifying real estate document types.
  
  Map each input document type to exactly ONE of the following categories:
  - SALE_DEED
  - MORTGAGE
  - DEED_OF_TRUST
  - RELEASE
  - LIEN
  - PLAT
  - EASEMENT
  - LEASE
  - MISC
  
  Rules:
  - Be conservative.
  - If unsure, choose MISC.
  - Return valid JSON only.
  - Do not include explanations.
  
  Input:
  [list of document type strings]
  ```
- Validation:
  - Parse JSON; enforce allowed category set; fallback to `MISC` on invalid/missing entries.
  - Log coverage achieved, LLM call count, and token-cost snapshot.
  - Manual spot-check recommended on top 20 frequent + a few rare values.
- Cost breakdown (estimate, OpenAI `gpt-4o-mini` pricing [link](https://openai.com/api/pricing)):
  - ~300 unique doc_types → ~15 calls at batch 20.
  - Token roughness: ~6k input + ~3.6k output → est. cost ≈ `$6k*0.15/1M + 3.6k*0.60/1M` ≈ $0.0031.
- Trade-offs:
  - Sampling to high coverage keeps cost minimal; rare tails default to `MISC`.
  - Small batches reduce prompt bleed; conservative instructions bias toward `MISC` when uncertain.

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
