# MERIDIAN Engine Intelligence

**Scan → Read → Validate → Speak.**

MERIDIAN turns scanned marine-engine performance report PDFs into **audited,
machine-readable measurement data** — a validated 124-field extractor with an
analytics query engine, threshold *triggers*, AI-assisted insights and
**offline voice briefings**, all wrapped in a self-contained Flask dashboard.

It is built around the **Reading section** of the report rather than treating
the whole PDF as one OCR page: pages are detected automatically, 14 calibrated
table regions are cropped, OCR is run per region, and every extracted value is
matched against a registry of expected source fields with a fidelity checker
that flags missing, blank, misplaced or corrupted cells.

---

## Pipeline

```text
PDF (upload or CLI)
      ↓
Detect Reading pages automatically (rows of the "Reading" section)
      ↓
Render the two Reading pages
      ↓
Crop 14 calibrated table regions
      ↓
OCR each region with bounding boxes (tesseract)
      ↓
Match the registered 124 measurement fields
      ↓
Extract logical columns / values  (v3 + v4 zone-aware engines)
      ↓
Validate fidelity against the source grid
      ↓
Write report.json / summary.json / validation.json / audit.json + CSV/Excel
      ↓
Searchable HTML table · graphs · analytics · triggers · voice briefing
```

The OCR is the source of every value. The parser **never invents missing
values** — cells keep OCR confidence, raw text, bounding boxes, source-row
text and flags so every value stays auditable.

### The "Reading" pages

The report family uses two consecutive Reading pages. The detector searches
page headers for the `Reading` section, so page numbers don't need to be
hard-coded (e.g. 14-page reports → pages 13–14, 8-page reports → 7–8).

### 14 logical regions (Reading page A → B)

| # | Region | # | Region |
|---|--------|---|--------|
| 1 | General | 8 | Fuel Oil |
| 2 | Power / Speed | 9 | Cylinder Lubrication |
| 3 | Electronic Control | 10 | Cylinder Condition |
| 4 | Cylinder Pressure | 11 | Crankcase |
| 5 | Turbocharger | 12 | Liner Wall |
| 6 | Scavenge Air | 13 | TC / SAC Media |
| 7 | Exhaust Gas | 14 | Engine Media / Plant |

`T001 - TIV (tot-tot)` is intentionally excluded → **124 registered fields**.

---

## Quickstart

Requires Python ≥ 3.11 and `tesseract-ocr` on the host.

```bash
sudo apt install -y tesseract-ocr tesseract-ocr-eng

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Web dashboard

```bash
./run_web.sh          # or: python -m meridian.web
# open http://127.0.0.1:5000
```

Drag a PDF into the page. The parser auto-detects the Reading pages,
extracts and validates the fields, and renders:

- a **searchable table** (search by field, value, unit, column, page, region
  or raw OCR text) with sticky headers, cell-level confidence details and
  low-confidence highlighting
- **trend graphs** comparing parameters across reports
- a **validator / registry** to manage source fields, tags and discrepancies
- an **analytics / query** workspace over the union of validated reports
- a **triggers** engine that fires when a field crosses a threshold
- **voice briefings** — see below

### CLI

```bash
# Automatic PDF mode (recommended)
python -m meridian.cli --mode pdf \
  --pdf example/2025__04__CE__01__ME_PERFORMANCE_REPORT_114.pdf \
  --engine v4 \
  --output output

# Manual calibrated-crop mode (needs a crops/ directory)
python -m meridian.cli --mode crops --crops /path/to/crops --output output
```

### Output

```text
output/
├── report.json          # every registered field, even blank ones
├── summary.json
├── validation.json      # fidelity check vs source grid
├── audit.json           # provenance / confidence audit
├── field_inventory.{json,csv}
├── measurements.{json,csv}
├── pages/  generated_crops/  ocr/
```

---

## Voice briefings (offline)

MERIDIAN can *read the report out loud* using **[Piper](https://github.com/rhasspy/piper)**,
a local neural TTS that runs entirely on-device — no cloud, no audio leaves
the machine.

```bash
pip install piper-tts
# download a voice, e.g. en_US-lessac-medium:
mkdir -p voices && cd voices
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

export MERIDIAN_PIPER_BIN="$(command -v piper)"
export MERIDIAN_PIPER_MODEL="$PWD/voices/en_US-lessac-medium.onnx"
export MERIDIAN_PIPER_CONFIG="$PWD/voices/en_US-lessac-medium.onnx.json"
./run_web.sh
```

Web endpoints:

- `GET /api/report/<job_id>/briefing` — JSON briefing text
- `GET /api/report/<job_id>/briefing/audio` — WAV of the spoken briefing

Without piper/feedback the endpoints degrade gracefully (JSON `voice_available:
false`, audio → `501`).

---

## AI-assisted insights

`meridian/llm_integration.py` integrates free OpenRouter models to enrich the
analytics workspace. It reads `OPENROUTER_API_KEYS`, `AI_MODEL` and
`AI_FALLBACK_MODEL` **from the environment only** — never from files. If no
keys are configured, the workspace still works fully offline.

---

## Testing & quality

```bash
pip install pytest ruff
ruff check .                          # scoped ruleset (E4,E9,F,I,E722)
pytest                                # 240+ tests
python -m meridian.cli --help
```

- The **regression corpus** lives in `pdfs_test/` (6 canonical quarterly
  reports) and `tests/fixtures/`. OCR tests require tesseract and are skipped
  automatically when the binary is missing.
- **CI** (`~/.github/workflows/ci.yml`) lints, runs tests across Python
  3.11–3.13 and boots the Flask app to assert routes are served.
- The 230+ passing suite covers OCR→reconstruction→validation→aggregation
  including adversarial cases (corrupted OCR, missing columns, blank cells).

---

## Project layout

```text
meridian/
├── cli.py                 # CLI entry (pdf / crops modes, engine v3/v4)
├── web.py                 # Flask dashboard (routes, upload, jobs, PDF export)
├── extract.py             # logical column extraction
├── ocr.py  crop_pipeline.py  pdf_pipeline.py  layout.py  matcher.py
├── table_reconstruction.py# zone-aware row/region reconstruction (+ v4)
├── v4_adapter.py          # validator-facing FieldRecord aggregation
├── validate_fidelity.py   # source-grid fidelity validation
├── audit.py               # confidence / provenance audit
├── registry_manager.py    # field registry + tags
├── analytics.py           # union catalog + query layer
├── triggers.py            # threshold triggers engine
├── llm_integration.py     # free-model AI insights
├── voice.py               # offline Piper briefings
config/fields.yaml         # registered source fields
templates/                 # web dashboard
tests/                     # 240+ tests
pdfs_test/                 # canonical regression corpus
```

---

## Security

- `.env`, API keys and runtime artifacts (`uploads/`, `web_output/`,
  `validation_output/`, `output/`, `logs/`) are gitignored.
- The dashboard is designed for a trusted LAN; put an authenticated reverse
  proxy in front if exposed publicly.
- See `SECURITY.md` for reporting channels.

## License

MIT © 2026 Moses Netto — see [LICENSE](LICENSE).