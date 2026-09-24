#!/usr/bin/env bash
set -euo pipefail

# Parse a single quarterly performance report PDF with the v4 engine.
if [ -n "${VENV:-}" ]; then
  source "${VENV}/bin/activate"
else
  source "$(dirname "$0")/.venv/bin/activate" 2>/dev/null || true
fi

python -m meridian.cli --mode pdf \
  --pdf "example/2025__04__CE__01__ME_PERFORMANCE_REPORT_114.pdf" \
  --engine v4 \
  --output output