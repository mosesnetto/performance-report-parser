# Contributing

Thanks for improving MERIDIAN Engine Intelligence. The engine parses scanned
marine-engine performance report PDFs into validated structured data — small
changes to the OCR / reconstruction / validation pipeline can have wide
impact, so changes are reviewed carefully.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest ruff
```

`tesseract-ocr` must be installed on the host (CI installs it automatically;
tests that need it are skipped locally when it is missing).

## Running checks

```bash
ruff check .          # scoped ruleset: E4,E9,F,I,E722
pytest                # full suite (uploaded corpus in tests/fixtures + pdfs_test/)
python -m meridian.cli --help
```

## Rules of thumb

- Keep the verified engine intact: prefer additive bug fixes over rewrites.
- Every engine behaviour change (OCR, table reconstruction, validation,
  aggregation) must ship with a regression test.
- Never commit PDFs except under `example/` (canonical samples) and
  `pdfs_test/` (regression corpus).
- Never commit `.env` or API keys. Secrets live in the environment or GitHub
  Secrets and are read by `llm_integration.py` at runtime.
- Match the project's ruff ruleset — statement-style rules are intentionally
  not enforced on legacy-derived code.

## Commit & PR

- Small, focused commits with descriptive messages.
- PRs must pass the CI jobs (lint, tests across Python 3.11–3.13, app-boot smoke).