## Description

Briefly describe the change and the problem it solves.

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature
- [ ] Engine behaviour change (OCR / validation / reconstruction)
- [ ] Docs / CI / tooling

## Checklist

- [ ] Code follows the project's ruff ruleset (bug-focused: `E4,E9,F,I,E722`)
- [ ] `ruff check .` passes
- [ ] `pytest` passes (230+ tests; OCR tests require tesseract — CI installs it)
- [ ] Modules import cleanly (`python -m meridian.cli --help`)
- [ ] If engine behaviour changed: regression tests added/updated
- [ ] No secrets or `.env` committed

## Verification

Describe how you verified the change (which PDFs, engine mode, screenshots).