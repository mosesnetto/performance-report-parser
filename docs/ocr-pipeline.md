# MERIDIAN OCR Pipeline — 124 Fields, 99.2% Accuracy

## Problem

Marine engine performance reports are scanned PDFs with dense tabular data. Need to extract 124 measurement fields with audit-grade accuracy for compliance, analytics, and voice briefings.

## Solution

A multi-stage pipeline: **PDF → Page Detection → Region Cropping → Per-Region OCR → Zone-Aware Reconstruction → Fidelity Validation → Structured Output**

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MERIDIAN OCR PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PDF Upload                                                                      │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. READING PAGE DETECTION                                               │   │
│  │    - Scans page headers for "Reading" section                           │   │
│  │    - Works regardless of page count (8, 14, 20+ pages)                  │   │
│  │    - Returns 2 consecutive page indices                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 2. PDF RENDERING (pdf2image / PyMuPDF)                                  │   │
│  │    - Renders detected pages at 300 DPI                                  │   │
│  │    - Outputs PNG for each Reading page                                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 3. CALIBRATED REGION CROPPING (14 regions)                              │   │
│  │    ┌──────────────┬──────────────┐  Page A (Reading page 1)             │   │
│  │    │ 1. General   │ 2. Power/    │  ┌──────────────┬──────────────┐    │   │
│  │    │              │    Speed     │  │ 8. Fuel Oil  │ 9. Cylinder  │    │   │
│  │    ├──────────────┼──────────────┤  │              │    Lubric.   │    │   │
│  │    │ 3. Electronic│ 4. Cylinder  │  ├──────────────┼──────────────┤    │   │
│  │    │    Control   │    Pressure  │  │ 10. Cylinder │ 11. Crank-   │    │   │
│  │    ├──────────────┼──────────────┤  │    Condition │    case      │    │   │
│  │    │ 5. Turbo-    │ 6. Scavenge  │  ├──────────────┼──────────────┤    │   │
│  │    │    charger   │    Air       │  │ 12. Liner    │ 13. TC/SAC   │    │   │
│  │    ├──────────────┼──────────────┤  │    Wall      │    Media     │    │   │
│  │    │ 7. Exhaust   │ (reserved)   │  ├──────────────┼──────────────┤    │   │
│  │    │    Gas       │              │  │ 14. Engine   │ (reserved)   │    │   │
│  │    └──────────────┴──────────────┘  │    Media/    │              │    │   │
│  │                                     │    Plant     │              │    │   │
│  │    Coordinates stored in            └──────────────┴──────────────┘    │   │
│  │    `config/fields.yaml` per region                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 4. PER-REGION OCR (Tesseract 5.x)                                       │   │
│  │    - Runs on each cropped region independently                          │   │
│  │    - Outputs: text + bounding boxes + confidence per word               │   │
│  │    - PSM 6 (single uniform block) for tables                            │   │
│  │    - Whitelist: alphanumeric + units + symbols (/, -, ., :, °, %)       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 5. ZONE-AWARE TABLE RECONSTRUCTION (Engine v3/v4)                       │   │
│  │    - v3: Row-based reconstruction using Y-coordinate clustering         │   │
│  │    - v4: Zone-aware (column + row) with header detection                │   │
│  │    - Handles: merged cells, multi-line values, rotated text             │   │
│  │    - Maps OCR tokens → logical columns (Tag, Description, Value, Unit)  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 6. FIELD MATCHING & REGISTRY (config/fields.yaml)                       │   │
│  │    - 124 registered fields with expected tags, units, pages, regions    │   │
│  │    - Fuzzy matching on tag + description (Levenshtein + token overlap)  │   │
│  │    - Confidence scoring: OCR conf × match conf × position conf          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 7. FIDELITY VALIDATION (validate_fidelity.py)                           │   │
│  │    - Compares extracted grid vs source OCR grid                         │   │
│  │    - Flags: MISSING, BLANK, MISPLACED, CORRUPTED, LOW_CONFIDENCE        │   │
│  │    - Every value keeps provenance: bbox, raw text, confidence, flags    │   │
│  │    - Output: validation.json + audit.json                               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                          │
│       ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 8. STRUCTURED OUTPUT                                                    │   │
│  │    - report.json: all 124 fields (even blanks) with full provenance     │   │
│  │    - summary.json: validated fields only                                │   │
│  │    - measurements.csv: flat table for analytics                         │   │
│  │    - field_inventory.csv: registry with tags, units, pages              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Reading Page Auto-Detection
```python
# meridian/layout.py
def find_reading_pages(pdf_path: Path) -> tuple[int, int]:
    """Returns (page_a_idx, page_b_idx) 0-indexed"""
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Reading" in text and "Section" in text:
            return i, i + 1
    raise ValueError("Reading section not found")
```
- No hardcoded page numbers
- Works across report formats (8–20+ pages)

### 2. Calibrated Regions (Not Full-Page OCR)
- 14 regions per report pair
- Coordinates calibrated once per report family
- Stored in `config/fields.yaml`:
```yaml
regions:
  - name: "General"
    page: 0  # Reading page A
    bbox: [x1, y1, x2, y2]  # pixels at 300 DPI
    expected_fields: ["T002", "T003", "T004", ...]
```

### 3. Zone-Aware Reconstruction (Engine v4)
```python
# meridian/table_reconstruction.py
class ZoneAwareReconstructor:
    def reconstruct(self, ocr_tokens: list[OCRToken]) -> Table:
        # 1. Cluster tokens into rows (Y-coordinate + tolerance)
        rows = self._cluster_rows(ocr_tokens)
        
        # 2. Detect header row (bold, top position, keywords)
        header_row = self._detect_header(rows)
        
        # 3. Map columns by header text + X-position
        col_map = self._map_columns(header_row)
        
        # 4. Assign each cell to (row_idx, col_name)
        return Table(cells=self._assign_cells(rows, col_map))
```
- Handles variable column counts per region
- Robust to OCR noise in headers

### 4. Fidelity Validation — Every Value Auditable
```json
// validation.json example
{
  "T015": {
    "status": "OK",
    "extracted": "78.5",
    "unit": "°C",
    "ocr_confidence": 0.94,
    "match_confidence": 0.98,
    "bbox": [124, 567, 189, 592],
    "source_text": "78.5 °C"
  },
  "T042": {
    "status": "LOW_CONFIDENCE",
    "extracted": "1.2",
    "unit": "bar",
    "ocr_confidence": 0.61,
    "match_confidence": 0.85,
    "flags": ["LOW_CONFIDENCE", "SUSPECT_DECIMAL"],
    "bbox": [445, 1123, 498, 1148],
    "source_text": "1.2 bar"
  }
}
```

## Accuracy Metrics

| Metric | Value | Method |
|--------|-------|--------|
| **Field-level accuracy** | 99.2% | 6 canonical reports × 124 fields = 744 values; manual verification |
| **OCR word confidence (median)** | 94% | Tesseract 5.3, 300 DPI, PSM 6 |
| **False positive rate** | <0.5% | Validation flags catch misplacements |
| **Missing field rate** | 1.8% | Mostly genuinely blank cells in source |

## Running the Pipeline

### Web Dashboard (Recommended)
```bash
./run_web.sh
# Open http://127.0.0.1:5000
# Drag PDF → auto-processes → shows table, graphs, validation, voice briefing
```

### CLI (Batch/Automation)
```bash
# Auto PDF mode (detects Reading pages automatically)
python -m meridian.cli --mode pdf \
  --pdf /path/to/report.pdf \
  --engine v4 \
  --output ./output

# Manual crops mode (if you have pre-cropped regions)
python -m meridian.cli --mode crops \
  --crops /path/to/crops_dir \
  --output ./output
```

### Output Files
```
output/
├── report.json          # Complete: all 124 fields + provenance
├── summary.json         # Validated fields only
├── validation.json      # Fidelity check results
├── audit.json           # Provenance audit trail
├── field_inventory.json # Registry with tags, units, pages
├── measurements.json    # Flat for analytics
├── measurements.csv     # Excel-ready
├── pages/               # Rendered Reading pages
├── generated_crops/     # 14 cropped regions
└── ocr/                 # Raw Tesseract output (HOCR/TSV)
```

## Voice Briefings (Offline)

```bash
# Install Piper TTS
pip install piper-tts
# Download voice
mkdir -p voices && cd voices
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# Set env vars
export MERIDIAN_PIPER_BIN="$(command -v piper)"
export MERIDIAN_PIPER_MODEL="$PWD/voices/en_US-lessac-medium.onnx"
export MERIDIAN_PIPER_CONFIG="$PWD/voices/en_US-lessac-medium.onnx.json"

./run_web.sh
```

Endpoints:
- `GET /api/report/<job_id>/briefing` — JSON text
- `GET /api/report/<job_id>/briefing/audio` — WAV audio

## Testing & Regression

```bash
# Run full test suite (240+ tests)
pytest

# Lint
ruff check .

# Regression corpus: 6 canonical quarterly reports in pdfs_test/
# Each has expected outputs in tests/fixtures/
```

## Files in This Repo

| File | Purpose |
|------|---------|
| `meridian/cli.py` | CLI entry (pdf/crops modes, v3/v4 engines) |
| `meridian/web.py` | Flask dashboard (upload, jobs, PDF export, voice) |
| `meridian/ocr.py` | Tesseract wrapper + bbox extraction |
| `meridian/crop_pipeline.py` | Region cropping from calibrated coords |
| `meridian/table_reconstruction.py` | Zone-aware row/col reconstruction (v3/v4) |
| `meridian/matcher.py` | Field registry matching |
| `meridian/validate_fidelity.py` | Source-grid fidelity validation |
| `meridian/audit.py` | Confidence/provenance audit |
| `meridian/voice.py` | Piper TTS briefings |
| `config/fields.yaml` | 124-field registry + region coordinates |
| `docs/ocr-pipeline.md` | This file |

## License

MIT — see [LICENSE](../LICENSE)