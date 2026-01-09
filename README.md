# Assessment Solution

## Setup
- Python ≥ 3.9; recommended: `python -m venv .venv && source .venv/bin/activate`.
- Install tooling: `pip install -r assessment_solution/requirements.txt`.
- Task 2 requires an OpenAI key: add `.env` with `OPENAI_API_KEY=<your_key>` (or export in your shell).


## Instructions

```bash
cd PATH_TO_SAVE  # TODO: Change to your project path
git clone git@github.com:netagindes/property-records-ai-pipeline.git  # [1] Clone to the project's GitHub repository
cd property-records-ai-pipeline.git

# [2] Create data folder in the project root and add all content form: [Assessment Data - Google Drive](https://drive.google.com/drive/folders/1fL3eSt1SGYTk4lROcs-Hfab3u4lGjch4)

python -m assessment_solution.src.pattern_analyzer --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/county_patterns.json --log-interval 1000  # [3] Run task #1

python -m assessment_solution.src.llm_classifier --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/doc_type_mapping.json --model gpt-4o-mini --coverage-target 0.98 --batch-size 16  # [4] Run task #2
```


## How to run scripts
- Task 1 (county patterns): `python -m assessment_solution.src.pattern_analyzer --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/county_patterns.json --log-interval 1000` (wrapper: `bash assessment_solution/scripts/run_pattern_analyzer.sh`).
- Task 2 (doc_type classification): `python -m assessment_solution.src.llm_classifier --input data/nc_records_assessment.jsonl --output assessment_solution/outputs/doc_type_mapping.json --model gpt-4o-mini --coverage-target 0.98 --batch-size 16` (wrapper: `bash assessment_solution/scripts/run_llm_classifier.sh`; requires `OPENAI_API_KEY`).
- Apply saved mapping to records: `python assessment_solution/scripts/map_doc_types.py --input data/nc_records_assessment.jsonl --mapping assessment_solution/outputs/doc_type_mapping.json --output assessment_solution/outputs/doc_type_mapped.jsonl`.
- Style checks: `ruff check assessment_solution` and `black --check assessment_solution` (helper script `assessment_solution/scripts/check_style.sh` mirrors these).

## Dependencies and requirements
- Runtime: standard library plus `openai` and `python-dotenv` for LLM access and `.env` loading.
- Dev tooling: `black==24.10.0`, `ruff==0.6.7` (pinned in `assessment_solution/requirements.txt`).

## Outputs and current results
- `assessment_solution/outputs/county_patterns.json`: generated from 13,886 records across 13 counties; per-county stats include `record_count`, instrument/book/page patterns, `unique_doc_types`, `doc_type_distribution`, and `date_range` with anomalies flagged. Overall earliest date seen: 0050-09-22; latest: 2238-04-25; 15 anomaly timestamps called out.
- `assessment_solution/outputs/doc_type_mapping.json`: mapping for 309 unique doc_type strings. Category counts: `MISC` 246, `DEED_OF_TRUST` 10, `RELEASE` 12, `SALE_DEED` 15, `EASEMENT` 12, `MORTGAGE` 4, `LIEN` 6, `LEASE` 1, `PLAT` 3. Applying the mapping yields 100% coverage of observed doc_type values.
- `assessment_solution/outputs/document_connections.json`: placeholder `[]`; bonus scripts (`connection_finder.py`, `pdf_analyzer.py`, `story_generator.py`) are stubs pending PDF parsing + connection logic.

## Task 1 (county pattern analysis)
- Streams JSONL line-by-line (no full-file loads), aggregates per-county instrument patterns, book/page hints, date ranges (earliest/latest/anomalies), and doc_type distributions, then writes `assessment_solution/outputs/county_patterns.json`.
- Logging-based progress every N records (`--log-interval`).

## Task 2 (LLM-assisted classification) – approach, prompts, validation, cost, trade-offs
- Approach: stream once, normalize doc_type strings (strip/uppercase/collapse spaces, underscores→spaces), count frequencies, and classify unique values only (cuts LLM volume ~99%). Sampling targets ~0.98 coverage; rare/upsampled values default to `MISC`. Current mapping covers 309 unique doc_types (100% of observed values).
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
  - 309 unique doc_types → ~20 calls at batch 16–20.
  - Rough token volume similar to ~6k input + ~3.6k output → estimated <$0.01.
  * Might be free using HuggingFace
- Trade-offs:
  - Sampling to high coverage keeps cost minimal; rare tails default to `MISC`.
  - Small batches reduce prompt bleed; conservative instructions bias toward `MISC` when uncertain.

## Bonus task status (connections, PDFs, storytelling)
- Current state: scripts are placeholders; `document_connections.json` remains empty until PDF parsing and connection discovery are implemented.
- Planned prompts: JSON-only extraction for parties/dates/amounts/addresses from PDFs, then a concise narration prompt to explain connections across 2–4 related documents.
- Cost approach: favor local text extraction first; only LLM on targeted sections; batch related docs; reuse parsed entities to avoid repeat calls.
- Insights & scaling: group by shared parties/dates/instrument patterns; cache parsed PDFs; keep deterministic grouping before LLM narration; parallelize extraction per county when adding the feature.

## Assumptions
- Data lives under `data/nc_records_assessment.jsonl` and PDFs under `data/records/<county>/<instrument>.pdf`.
- Outputs are written under `assessment_solution/outputs` and can be regenerated locally.
- LLM access/credentials are available externally (not stored in repo).
- JSON outputs are stable (sorted keys where relevant); code targets Python 3.9, PEP-8, type hints, defensive parsing, and logging (no bare prints except CLI summaries).
