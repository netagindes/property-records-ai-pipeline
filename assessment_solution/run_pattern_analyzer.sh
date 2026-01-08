#!/usr/bin/env bash
set -euo pipefail

# Run the streaming county pattern analyzer.
# Usage:
#   ./assessment_solution/run_pattern_analyzer.sh [--input path] [--output path] [--log-interval N] [--help]
#
# Defaults mirror the Python module: input data/nc_records_assessment.jsonl,
# output assessment_solution/outputs/county_patterns.json, log every 1000 records.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

python -m assessment_solution.src.pattern_analyzer \
  --input "${REPO_ROOT}/data/nc_records_assessment.jsonl" \
  --output "${REPO_ROOT}/assessment_solution/outputs/county_patterns.json" \
  "$@"
