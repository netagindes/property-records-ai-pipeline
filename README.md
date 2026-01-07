# Assessment Solution

## Development standards
- Style is clean, well-documented, and PEP-8 compliant.
- Black enforces formatting (100 char lines, Python 3.9 targets).
- Ruff enforces linting (errors, imports, bugs) using `pyproject.toml` config.

## Setup
- Python >= 3.9
- Install tooling: `pip install -r assessment_solution/requirements.txt`

## Style validation
- Lint: `ruff check assessment_solution`
- Format check: `black --check assessment_solution`
- Combined helper: `bash assessment_solution/scripts/check_style.sh`
