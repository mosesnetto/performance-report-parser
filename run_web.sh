#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${VENV:-}" ]; then
  source "${VENV}/bin/activate"
else
  source .venv/bin/activate 2>/dev/null || true
fi

exec python -m meridian.web