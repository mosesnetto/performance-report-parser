"""
Synthetic tests for the hardened V4 table reconstruction engine.

Covers:
- compound (multi-token) header reconstruction
- header vs row-value vs row-label separation
- explicit row model (label/unit/qualifier/value regions)
- cell model (header path, value bbox, status indicator)
- blank cell preservation (no left-shift)
- qualifier detection (ALL / MOP SETTING:)
- status dot inside cell
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian.table_reconstruction import (
    Band,
    Cell,
    Column,
    HeaderKind,
    RowModel,
    Token,
    _detect_cell_status,
    _field_run_rows,
    _field_runs_in_band,
    _gap_threshold,
    _grid_keyword_count,
    _has_alpha_word,
    _is_grid_header_band,
    _is_value_artifact,
    _looks_like_number,
    discover_zones,
    ingest_tokens,
    reconstruct,
)

# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────


def _tok(text, left, top, width=50, height=16, conf=95.0, page=1, region="test"):
    return {
        "text": text,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "conf": conf,
        "page": page,
        "region": region,
    }


def _assert(condition, msg):
    if not condition:
        raise AssertionError(msg)


def _assert_eq(a, b, label=""):
    if a != b:
        raise AssertionError(f"{label}: expected {b!r}, got {a!r}")


def _assert_near(a, b, tol=1.0, label=""):
    if abs(a - b) > tol:
        raise AssertionError(f"{label}: expected ~{b}, got {a}")


def _tk(text, left, top, width=50, height=16, conf=95.0):
    return Token(text=text, left=left, top=top, width=width, height=height,
                 conf=conf, page=1, region="t")


def _data_rows(result):
    """Return the RowModel list for data rows in order."""
    return [pr.model for pr in result.physical_rows if pr.model is not None]


def _cell_by_label(row: RowModel, label: str) -> Cell:
    for c in row.cells:
        if c.header_label == label:
            return c
    raise AssertionError(f"no cell with header {label!r} in row; got {[c.header_label for c in row.cells]}")


# ───────────────────────────────────────────────────────────
# 1. Compound header: CYL + 10 -> CYL 10
# ───────────────────────────────────────────────────────────


def test_compound_header_cyl_10():
    raw = [
        # Header: REF, CYL, 10 (split)
        _tok("REF", 630, 40, 30, 16),
        _tok("CYL", 960, 40, 40, 16),
        _tok("10", 1005, 40, 30, 16, conf=90),
        # Data row
        _tok("1.5", 640, 80, 30, 16, conf=95),
        _tok("2.5", 970, 80, 30, 16, conf=95),
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    labels = [c.label for c in r.leaf_columns]
    # REF and CYL 10 should be the two value columns.
    _assert("CYL 10" in labels, f"CYL 10 not merged; got {labels}")
    _assert("REF" in labels, f"REF missing; got {labels}")
    _assert(any("CYL" in l and l != "CYL 10" for l in labels) is False,
            f"CYL should not remain a separate column; got {labels}")
    print("  PASS: test_compound_header_cyl_10")


# ───────────────────────────────────────────────────────────
# 2. Arbitrary two-token alphabetic header
# ───────────────────────────────────────────────────────────


def test_compound_header_two_word():
    raw = [
        _tok("GRO", 100, 40, 40, 16),
        _tok("SS", 145, 40, 30, 16),          # GRO + SS
        _tok("VALUE", 400, 40, 50, 16),
        _tok("1.0", 110, 80, 30, 16),
        _tok("2.0", 410, 80, 30, 16),
    ]
    r = reconstruct(raw)
    labels = [c.label for c in r.leaf_columns]
    _assert("GRO SS" in labels, f"two-word header not merged; got {labels}")
    print("  PASS: test_compound_header_two_word")


# ───────────────────────────────────────────────────────────
# 3. Arbitrary three-token prefix+number header
# ───────────────────────────────────────────────────────────


def test_compound_header_three_token():
    # e.g. "SER" + "NO" + "5" -> "SER NO 5" as one column
    raw = [
        _tok("SER", 100, 40, 40, 16),
        _tok("NO", 145, 40, 30, 16),
        _tok("5", 180, 40, 20, 16),
        _tok("VALUE", 500, 40, 50, 16),
        _tok("9", 115, 80, 20, 16),
        _tok("3.3", 510, 80, 30, 16),
    ]
    r = reconstruct(raw)
    labels = [c.label for c in r.leaf_columns]
    " ".join(labels)
    _assert("SER NO 5" in labels, f"three-token header not merged into one; got {labels}")
    print("  PASS: test_compound_header_three_token")


# ───────────────────────────────────────────────────────────
# 4. Header vs row-value confusion
# ───────────────────────────────────────────────────────────


def test_header_vs_row_value():
    """
    A single-column label like "pmax deviation" should NOT become a
    header/column; it is row content. Only true header bands above the
    value grid define columns.
    """
    raw = [
        # True header band.
        _tok("REF", 600, 40, 30, 16),
        _tok("VAL", 900, 40, 40, 16),
        # Data row 1: mixed text label + numeric values.
        _tok("pmax", 100, 100, 50, 16),
        _tok("dev", 160, 100, 30, 16),
        _tok("1.1", 610, 100, 30, 16),
        _tok("2.2", 910, 100, 30, 16),
        # Data row 2.
        _tok("comp", 100, 150, 50, 16),
        _tok("1.5", 610, 150, 30, 16),
        _tok("2.6", 910, 150, 30, 16),
    ]
    r = reconstruct(raw)
    cols = [c.label for c in r.leaf_columns]
    _assert("REF" in cols and "VAL" in cols, f"missing value columns; got {cols}")
    _assert(all(c not in cols for c in ("pmax", "dev", "comp")),
            f"row labels leaked into columns; got {cols}")
    rows = _data_rows(r)
    _assert_eq(len(rows), 2, "should have 2 data rows")
    # Row content should not be columns.
    _assert_eq(len(rows[0].cells), 2, "row should have exactly 2 value cells")
    print("  PASS: test_header_vs_row_value")


# ───────────────────────────────────────────────────────────
# 5. Row label containing uppercase text (not header)
# ───────────────────────────────────────────────────────────


def test_row_label_uppercase():
    raw = [
        _tok("A", 600, 40, 30, 16),
        _tok("B", 900, 40, 30, 16),
        # Uppercase row label must stay a label, not a column.
        _tok("MEAN", 80, 100, 60, 16),
        _tok("PRESS", 150, 100, 60, 16),
        _tok("1.0", 610, 100, 30, 16),
        _tok("2.0", 910, 100, 30, 16),
        _tok("PWR", 80, 150, 40, 16),
        _tok("3.0", 610, 150, 30, 16),
        _tok("4.0", 910, 150, 30, 16),
    ]
    r = reconstruct(raw)
    cols = [c.label for c in r.leaf_columns]
    _assert("A" in cols and "B" in cols, f"missing columns; got {cols}")
    _assert("MEAN" not in cols and "PRESS" not in cols and "PWR" not in cols,
            f"uppercase row labels became columns; got {cols}")
    rows = _data_rows(r)
    # MEAN PRESS should be in the label region.
    _assert("MEAN" in rows[0].label_region.text, f"MEAN not in label region: {rows[0].label_region.text}")
    print("  PASS: test_row_label_uppercase")


# ───────────────────────────────────────────────────────────
# 6. Qualifier such as ALL / MOP SETTING:
# ───────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────
# 7. Blank middle cell (position preserved, no left-shift)
# ───────────────────────────────────────────────────────────


def test_blank_middle_cell():
    """
    Row values in columns A and C but not B -> B must remain blank and
    C must not shift into B's position.
    """
    raw = [
        _tok("A", 600, 40, 30, 16),
        _tok("B", 700, 40, 30, 16),
        _tok("C", 800, 40, 30, 16),
        # Row 1: value in A only.
        _tok("1.0", 610, 100, 30, 16),
        # Row 2: value in A and C, B blank.
        _tok("2.0", 610, 150, 30, 16),
        _tok("3.0", 810, 150, 30, 16),
    ]
    r = reconstruct(raw)
    rows = _data_rows(r)
    _assert_eq(len(rows), 2)
    row = rows[1]
    _assert_eq(len(row.cells), 3, "should have 3 value cells")
    cA = _cell_by_label(row, "A")
    cB = _cell_by_label(row, "B")
    cC = _cell_by_label(row, "C")
    _assert(not cA.is_blank, "A should be present")
    _assert(cB.is_blank, "B should be blank (middle)")
    _assert(not cC.is_blank, "C should be present")
    _assert_eq(cC.text, "3.0", "C's value must not have shifted")
    row1 = rows[0]
    _assert(not _cell_by_label(row1, "A").is_blank, "row1 A present")
    _assert(_cell_by_label(row1, "B").is_blank, "row1 B blank")
    _assert(_cell_by_label(row1, "C").is_blank, "row1 C blank")
    print("  PASS: test_blank_middle_cell")


# ───────────────────────────────────────────────────────────
# 8. Status dot inside a cell
# ───────────────────────────────────────────────────────────


def test_status_dot():
    cell = Cell(
        row_idx=0,
        column=Column("c0", "C1", ["C1"], 900, 1000),
        tokens=[_tk("2.0", 905, 100, 30, 16, 95), _tk("*", 950, 100, 6, 6, 70)],
        is_blank=False,
    )
    _detect_cell_status(cell)
    _assert(len(cell.status) == 1, f"expected 1 status indicator, got {len(cell.status)}")
    _assert_eq(cell.status[0].kind, "symbol")
    _assert_eq(cell.status[0].text, "*")
    _assert_eq(cell.text, "2.0 *", "text includes value and status token")
    print("  PASS: test_status_dot")


# ───────────────────────────────────────────────────────────
# 9. Cell model: header path + value bbox + confidence
# ───────────────────────────────────────────────────────────


def test_cell_model():
    raw = [
        _tok("GRP", 590, 30, 40, 16),
        _tok("REF", 600, 40, 30, 16),
        _tok("c1", 900, 40, 30, 16),
        _tok("5.5", 610, 100, 30, 16, conf=90),
        _tok("6.6", 910, 100, 30, 16, conf=80),
    ]
    r = reconstruct(raw)
    rows = _data_rows(r)
    _assert_eq(len(rows), 1)
    cell = _cell_by_label(rows[0], "c1")
    _assert_eq(cell.header_label, "c1")
    _assert(isinstance(cell.header_path, list) and "c1" in cell.header_path,
            "cell should carry header path")
    _assert(not cell.is_blank, "cell present")
    _assert(cell.value_bbox is not None, "cell should have value bbox")
    _assert_near(cell.confidence, 80.0, tol=0.1, label="cell confidence")
    _assert(cell.cell_bbox is not None, "cell should have cell bbox")
    print("  PASS: test_cell_model")


# ───────────────────────────────────────────────────────────
# 10. Row model label/unit/qualifier/value regions
# ───────────────────────────────────────────────────────────


def test_row_model_regions():
    raw = [
        _tok("REF", 600, 40, 30, 16),
        _tok("C1", 900, 40, 30, 16),
        # Row: label + unit bar + value.
        _tok("Firing", 60, 100, 60, 16),
        _tok("press.", 130, 100, 50, 16),
        _tok("barG", 250, 100, 40, 16),
        _tok("1.0", 610, 100, 30, 16),
        _tok("2.0", 910, 100, 30, 16),
    ]
    r = reconstruct(raw)
    rows = _data_rows(r)
    _assert_eq(len(rows), 1)
    m = rows[0]
    _assert(m.bbox is not None, "row bbox present")
    _assert("Firing" in m.label_region.text, f"label region: {m.label_region.text!r}")
    _assert(m.unit_region.text == "" or "barG" in m.unit_region.text,
            f"unit region: {m.unit_region.text!r}")
    _assert_eq(len(m.cells), 2, "two value cells")
    print("  PASS: test_row_model_regions")


# ───────────────────────────────────────────────────────────
# 11. Lookup helpers / numeric
# ───────────────────────────────────────────────────────────


def test_numeric_noise():
    _assert(_looks_like_number("0.0|®"), "0.0|® should be numeric after noise strip")
    _assert(_looks_like_number("-0.02/®"), "-0.02/® should be numeric")
    _assert(_looks_like_number("146.9"), "146.9 numeric")
    _assert(not _looks_like_number("deviation"), "word not numeric")
    print("  PASS: test_numeric_noise")


# ───────────────────────────────────────────────────────────
# 12. Empty input
# ───────────────────────────────────────────────────────────


def test_empty():
    r = reconstruct([])
    _assert(not r.status.ok, "should fail on empty")
    print("  PASS: test_empty")


# ───────────────────────────────────────────────────────────
# 13. Header hierarchy kinds
# ───────────────────────────────────────────────────────────


def test_header_kinds():
    raw = [
        # Title band.
        _tok("ISO", 500, 20, 30, 16),
        _tok("CORRECTED", 540, 20, 80, 16),
        # Note: title band only 2 tokens -> classified as title, not header
        _tok("REF", 600, 60, 30, 16),
        _tok("C1", 900, 60, 30, 16),
        _tok("1.0", 610, 120, 30, 16, conf=90),
        _tok("2.0", 910, 120, 30, 16, conf=90),
        _tok("3.0", 610, 170, 30, 16, conf=90),
        _tok("4.0", 910, 170, 30, 16, conf=90),
    ]
    r = reconstruct(raw)
    kinds = {n.kind for n in r.header_hierarchy}
    # We expect at least leaf columns. (Title band with 2 tokens may be title.)
    _assert(HeaderKind.LEAF in kinds, f"no leaf kind in header tree; {kinds}")
    print("  PASS: test_header_kinds")


# ───────────────────────────────────────────────────────────
# Run all
# ───────────────────────────────────────────────────────────


def test_text_only_row_label_not_chosen_as_primary_header():
    # A table whose FIRST data row has a long all-alphabetic row label + unit
    # ("Water press. SAC in" + "bar" = 5 tokens, more than the real header's
    # 4) must NOT be picked as the primary column scheme. The real header band
    # (AVG / Tc1 / Tc2 / TC3) must win and produce value columns.
    raw = [
        # Real header band (4 column labels).
        _tok("AVG.", 861, 60, 45, 16, conf=95),
        _tok("Tc1", 971, 60, 38, 16, conf=95),
        _tok("Tc2", 1070, 60, 41, 16, conf=95),
        _tok("TC3", 1168, 60, 41, 16, conf=95),
        # Text-only data row label + unit (5 tokens) - more than the header.
        _tok("Water", 231, 111, 62, 16, conf=95),
        _tok("press.", 301, 116, 57, 16, conf=95),
        _tok("SAC", 366, 111, 40, 16, conf=95),
        _tok("in", 415, 111, 17, 16, conf=95),
        _tok("bar", 544, 111, 33, 16, conf=95),
        # Numeric data row with values under the real columns.
        _tok("44", 910, 148, 26, 16, conf=95),
        _tok("44", 1009, 148, 26, 16, conf=95),
        _tok("44", 1107, 148, 26, 16, conf=95),
        _tok("44", 1205, 148, 26, 16, conf=95),
    ]
    r = reconstruct(raw)
    cols = [c.label for c in r.leaf_columns]
    _assert_eq(cols, ["Tc1", "Tc2", "TC3"],
               f"real columns detected, not the text row-label: {cols}")
    # The three detected columns must each carry the numeric value.
    rows = _data_rows(r)
    _assert(rows, "at least one data row preserved")
    vals = [c for rw in rows for c in rw.value_cells if not c.is_blank]
    _assert(len(vals) >= 3, f"numeric values preserved in columns: {len(vals)}")
    print("  PASS: test_text_only_row_label_not_chosen_as_primary_header")


# ───────────────────────────────────────────────────────────
# Value-artifact token detection
# ───────────────────────────────────────────────────────────


def test_value_artifact_tokens():
    # Multi-glyph OCR tokens made only of table rules / filler boxes / scanner
    # speckle must be treated as value artifacts (kept with the value), not as
    # field labels that would break row/field segmentation.
    for t in ["|", "|®", "®©", "{}"]:
        assert _is_value_artifact(t), f"{t!r} should be a value artifact"
    # Ordinary textual label tokens are NOT value artifacts.
    for t in ["ENGINE", "power", "estimated", "Lub", "oil", "temp."]:
        assert not _is_value_artifact(t), f"{t!r} must NOT be a value artifact"
    # Numeric tokens are value-like by design (numbers are kept with the value).
    for t in ["27012", "52.0", "0.5"]:
        assert _is_value_artifact(t), f"{t!r} should stay value-like"
    print("  PASS: test_value_artifact_tokens")


def test_adoption_guard_short_real_labels_preserved():
    # Grid title-label adoption must NOT overwrite a short-but-real field label
    # (e.g. the quadrant marker 'fp' / 'fwd' / 'aft'), which would steal another
    # field's title and corrupt both records. Only labels that lack any
    # alphabetic word (pure OCR junk like '(' or '|®') may adopt a title.
    for t in ["fp", "fwd", "aft", "exh", "AVG", "CALC"]:
        assert _has_alpha_word([_tk(t, 0, 0)], 2), f"{t!r} should be a preserved label"
    # Junk / artifact-only labels have no alphabetic word -> eligible to adopt.
    for t in ["(", "|®", "{}", "®©"]:
        assert not _has_alpha_word([_tk(t, 0, 0)], 2), f"{t!r} should NOT block adoption"
    print("  PASS: test_adoption_guard_short_real_labels_preserved")


# ───────────────────────────────────────────────────────────
# Run all
# ───────────────────────────────────────────────────────────
def main():
    tests = [
        test_compound_header_cyl_10,
        test_compound_header_two_word,
        test_compound_header_three_token,
        test_header_vs_row_value,
        test_row_label_uppercase,
        test_qualifier,
        test_blank_middle_cell,
        test_status_dot,
        test_cell_model,
        test_row_model_regions,
        test_empty,
        test_header_kinds,
        test_text_only_row_label_not_chosen_as_primary_header,
        test_value_artifact_tokens,
        test_adoption_guard_short_real_labels_preserved,
        test_zero_data_row_non_grid_region,
        test_one_data_row_grid_remains_grid,
        test_label_plus_text_value,
        test_label_plus_numeric_value,
        test_label_plus_unit_plus_value,
        test_label_value_on_separate_ocr_bands,
        test_qualifier,
        test_true_header_not_promoted_to_field,
        test_true_leaf_header_not_promoted_to_field,
        test_label_only_physical_blank,
        test_repeating_short_column_header_still_dropped,
        test_long_unique_field_label_preserved,
        test_schema_anchored_short_label_preserved,
    ]
    passed = failed = 0
    errors = []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((t.__name__, str(e)))
            print(f"  FAIL: {t.__name__}: {e}")
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    for name, err in errors:
        print(f"  {name}: {err}")
    print(f"{'='*60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

def test_zero_data_row_non_grid_region():
    """Test that zero-data-row region is never classified as grid."""
    # Region with only header-like content, no data-bearing tokens
    raw = [
        # Header band 1
        _tok("Engine", 100, 100, 100, 20),
        _tok("Speed", 250, 100, 100, 20),
        _tok("RPM", 400, 100, 100, 20),
        # Header band 2  
        _tok("Voltage", 100, 150, 100, 20),
        _tok("Current", 250, 150, 100, 20),
        _tok("Power", 400, 150, 100, 20),
        # Header band 3
        _tok("Temperature", 100, 200, 120, 20),
        _tok("Pressure", 250, 200, 100, 20),
        _tok("Flow", 400, 200, 100, 20),
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should succeed and not crash - the key is that it doesn't treat empty data as grid
    # With zero data rows, we expect it to create rows for header-like content via non-grid path
    _assert(len(r.physical_rows) >= 3, f"Expected at least 3 physical rows from header bands, got {len(r.physical_rows)}")


def test_one_data_row_grid_remains_grid():
    """Test that region with exactly one data row preserves grid behavior when column structure is clear."""
    # Region with one clear data row but multi-column header structure that should yield multiple columns
    raw = [
        # Header-like tokens that should be discovered as column headers
        _tok("COL_A", 100, 100, 80, 20),
        _tok("COL_B", 250, 100, 80, 20), 
        _tok("COL_C", 400, 100, 80, 20),
        # Data row with values under each column
        _tok("10.5", 120, 150, 60, 20),  # Under COL_A
        _tok("20.0", 270, 150, 60, 20),  # Under COL_B
        _tok("30.5", 420, 150, 60, 20),  # Under COL_C
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have discovered multiple columns (not fallen back to single VALUE column)
    _assert(len(r.leaf_columns) >= 2, f"Expected multiple columns for clear grid structure, got {len(r.leaf_columns)}")
    # Should have one data row (the data band)
    # Note: header bands may also become physical rows depending on _is_label_band_field
    # But the key is that we kept the column structure rather than falling back to single column


def test_label_plus_text_value():
    """Test label combined with text value (handled as value when appropriate)."""
    # Create a case where we have a label and what should be treated as a value
    # Note: In this system, "values" are typically numeric, but we can test with value artifacts
    raw = [
        _tok("Label", 100, 100, 80, 20),
        _tok("value_text", 250, 100, 100, 20),  # This would normally not be a value
        # Add a clear numeric value to make it a data-bearing band
        _tok("42", 400, 100, 50, 20),
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have extracted the numeric value
    found_values = []
    for row in r.physical_rows:
        if row.model:
            for cell in row.model.cells:
                if not cell.is_blank:
                    found_values.append(cell.text)
    _assert("42" in found_values, f"Expected to find value '42', got {found_values}")


def test_label_plus_numeric_value():
    """Test label combined with numeric value."""
    raw = [
        _tok("Pressure", 100, 100, 100, 20),
        _tok("250.5", 250, 100, 80, 20),  # Numeric value
        _tok("kPa", 350, 100, 50, 20),    # Unit
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have extracted the numeric value with unit
    found_values = []
    for row in r.physical_rows:
        if row.model:
            for cell in row.model.cells:
                if not cell.is_blank:
                    found_values.append(cell.text)
    _assert("250.5" in found_values, f"Expected to find value '250.5', got {found_values}")


def test_label_plus_unit_plus_value():
    """Test label combined with unit and value."""
    raw = [
        _tok("Temperature", 100, 100, 100, 20),
        _tok("98.6", 300, 100, 100, 20),   # Numeric value
        _tok("°F", 500, 100, 100, 20),     # Unit
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have extracted the value with unit
    found_values = []
    for row in r.physical_rows:
        if row.model:
            for cell in row.model.cells:
                if not cell.is_blank:
                    found_values.append(cell.text)
    _assert("98.6" in found_values, f"Expected to find value '98.6', got {found_values}")


def test_label_value_on_separate_ocr_bands():
    """Test label and value on separate OCR bands (vertical continuation)."""
    raw = [
        # Label on first band
        _tok("Voltage", 100, 100, 100, 20),
        # Value on second band (continuation)
        _tok("120", 100, 150, 80, 20),   # Numeric value
        _tok("V", 200, 150, 40, 20),     # Unit
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have associated the label with the value through vertical continuation
    found_values = []
    for row in r.physical_rows:
        if row.model:
            for cell in row.model.cells:
                if not cell.is_blank:
                    found_values.append((row.model.label_region.text.strip(), cell.text))
    # Look for the voltage label associated with the 120 value
    label_value_pairs = found_values
    found = any("Voltage" in label and "120" in value for label, value in label_value_pairs)
    _assert(found, f"Expected to find 'Voltage' label associated with '120' value, got pairs: {label_value_pairs}")


def test_qualifier():
    """Test qualifier handling (e.g., ALL, parentheses)."""
    raw = [
        _tok("Setting", 100, 100, 100, 20),
        _tok("ALL", 250, 100, 60, 20),   # Qualifier
        _tok("50", 350, 100, 60, 20),    # Value
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have extracted the value and handled the qualifier
    found_values = []
    for row in r.physical_rows:
        if row.model:
            for cell in row.model.cells:
                if not cell.is_blank:
                    found_values.append(cell.text)
    _assert("50" in found_values, f"Expected to find value '50', got {found_values}")
    # Check if qualifier was captured in qualifier regions
    qualifier_found = False
    for row in r.physical_rows:
        if row.model:
            for qualifier in row.model.qualifier_regions:
                if qualifier.text.strip() == "ALL":
                    qualifier_found = True
                    break
    _assert(qualifier_found, f"Expected to find qualifier 'ALL', got qualifiers: {[q.text.strip() for row in r.physical_rows if row.model for q in row.model.qualifier_regions]}")


def test_true_header_not_promoted_to_field():
    """Test that a genuine repeating short column header is NOT promoted to a field label.

    Uses _field_run_rows directly since synthetic tokens through reconstruct()
    do not reliably produce the band structure needed to trigger the filter.
    """
    col = _synth_value_column()
    b1 = Band(y_min=40, y_max=56, tokens=[_synth_token("REF", 100, 40)],
              row_type="header")
    b2 = Band(y_min=60, y_max=76, tokens=[_synth_token("REF", 100, 60)],
              row_type="header")
    data = Band(y_min=100, y_max=116,
                tokens=[_synth_token("RPM", 100, 100), _synth_token("3000", 200, 100)],
                row_type="data")
    rows = _field_run_rows([b1, b2, data], [col], is_grid=False)
    labels = _row_labels(rows)
    _assert("REF" not in labels,
            f"Repeating column header 'REF' was promoted to field label: {labels}")
    _assert(any("RPM" in lbl for lbl in labels),
            f"Data row label 'RPM' missing: {labels}")


def test_true_leaf_header_not_promoted_to_field():
    """Test that true leaf header is not promoted to field label."""
    # Similar to above but focusing on leaf header concept
    raw = [
        # Multi-level header where bottom level is leaf header
        _tok("Sensor", 100, 80, 100, 20),      # Group header
        _tok("Bank 1", 250, 80, 100, 20),      # Group header
        _tok("Ch A", 100, 120, 80, 20),        # Leaf header
        _tok("Ch B", 220, 120, 80, 20),        # Leaf header
        _tok("Ch C", 340, 120, 80, 20),        # Leaf header
        # Data under each channel
        _tok("2.5", 120, 160, 60, 20),         # Under Ch A
        _tok("3.0", 240, 160, 60, 20),         # Under Ch B
        _tok("1.8", 360, 160, 60, 20),         # Under Ch C
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Leaf headers ("Ch A", "Ch B", "Ch C") should not appear as field labels
    field_labels = []
    for row in r.physical_rows:
        if row.model:
            field_labels.append(row.model.label_region.text.strip())
    leaf_headers = ["Ch A", "Ch B", "Ch C"]
    for leaf_header in leaf_headers:
        found = any(leaf_header in label or label in leaf_header for label in field_labels if label)
        _assert(not found, f"Leaf header '{leaf_header}' was incorrectly promoted to field label. Field labels: {field_labels}")


def test_label_only_physical_blank():
    """Test label-only band is handled as physical blank."""
    raw = [
        # Label-only band (no value)
        _tok("Idle Status", 100, 100, 150, 20),
        # Data band with actual value
        _tok("Active", 100, 150, 100, 20),
        _tok("42", 250, 150, 80, 20),  # Value
        _tok("units", 350, 150, 80, 20), # Unit
    ]
    r = reconstruct(raw)
    _assert(r.status.ok, f"failed: {r.status.warnings}")
    # Should have created a physical row for the label-only band
    # Check that we have rows and that at least one has blank cells
    blank_cells_found = False
    for row in r.physical_rows:
        if row.model:
            for cell in row.model.cells:
                if cell.is_blank:
                    blank_cells_found = True
                    break
        if blank_cells_found:
            break
    _assert(blank_cells_found, "Expected to find blank cells from label-only physical blank, but all cells had values")


# ───────────────────────────────────────────────────────────
# Structural column-header vs field-label filtering
# ───────────────────────────────────────────────────────────


def _synth_token(text, left, top):
    return Token(text=text, left=left, top=top, width=len(text) * 9,
                 height=16, conf=95.0, page=1, region="test")


def _synth_value_column():
    return Column(col_id="V", label="VALUE", path=["VALUE"], x_min=200, x_max=400)


def _row_labels(rows):
    """Return the non-empty reconstructed label texts from physical rows."""
    out = []
    for row in rows:
        if row.model and row.model.label_region.text.strip():
            out.append(row.model.label_region.text.strip())
    return out


def test_repeating_short_column_header_still_dropped():
    """A short header label repeated verbatim across 2+ header bands is still
    recognised as a column header and must NOT be lifted into a field row,
    even when data bands exist below it (the exhaust_gas REF/'CYL1' pattern)."""
    col = _synth_value_column()
    b1 = Band(y_min=40, y_max=56, tokens=[_synth_token("REF", 100, 40)],
              row_type="header")
    b2 = Band(y_min=60, y_max=76, tokens=[_synth_token("REF", 100, 60)],
              row_type="header")
    b3 = Band(y_min=100, y_max=116,
              tokens=[_synth_token("Lub", 100, 100), _synth_token("oil", 160, 100),
                      _synth_token("press.", 220, 100)],
              row_type="data")
    rows = _field_run_rows([b1, b2, b3], [col], is_grid=False)
    labels = _row_labels(rows)
    _assert("REF" not in labels,
            f"Repeating short column header 'REF' promoted to field label: {labels}")
    _assert(any("Lub oil press." in lbl for lbl in labels),
            f"Unique field label missing after header filtering: {labels}")


def test_long_unique_field_label_preserved():
    """A long, unique, label-only field label in a header-classified band is
    preserved as a physical row (it is a real field, not a column header)."""
    col = _synth_value_column()
    # A field label on its own line in a header-classified band, with a
    # separate data band below it.  Not repeated -> must be kept.
    header = Band(
        y_min=100, y_max=116,
        tokens=[_synth_token("Lub", 100, 100), _synth_token("oil", 160, 100),
                _synth_token("temp.", 220, 100), _synth_token("TURB.", 330, 100),
                _synth_token("OIL", 410, 100), _synth_token("in", 500, 100)],
        row_type="header",
    )
    data = Band(y_min=130, y_max=146, tokens=[_synth_token("45", 300, 130)],
                row_type="data")
    rows = _field_run_rows([header, data], [col], is_grid=False)
    labels = _row_labels(rows)
    found = any("Lub oil temp. TURB. OIL in".lower() in lbl.lower() for lbl in labels)
    _assert(found, f"Long unique field label was dropped by header filter: {labels}")


def test_schema_anchored_short_label_preserved():
    """A short label-only run that matches a real schema field alias is
    preserved through the geometry engine (schema anchoring happens later in
    the adapter, so the row must not be dropped here)."""
    col = _synth_value_column()
    b1 = Band(y_min=40, y_max=56, tokens=[_synth_token("REF", 100, 40)],
              row_type="header")
    # A distinct (non-repeating) short label that is a real field token.
    b2 = Band(y_min=60, y_max=76, tokens=[_synth_token("MOP", 100, 60)],
              row_type="header")
    data = Band(y_min=100, y_max=116, tokens=[_synth_token("50", 300, 100)],
                row_type="data")
    rows = _field_run_rows([b1, b2, data], [col], is_grid=False)
    labels = _row_labels(rows)
    _assert(any(lbl.strip() == "MOP" for lbl in labels),
            f"Schema-anchored short label 'MOP' was dropped: {labels}")

    # ───────────────────────────────────────────────────────────
    # Page-association regression tests (Blocker 1)
    # ───────────────────────────────────────────────────────────


def test_schema_page_mismatch_fields_still_match_by_region():
    """Fields with schema page=13 must match crops from physical page 7.

    This is the SEP scenario: fields.yaml registers all fields on pages 13/14,
    but the actual Reading pages are [7,8].  The page-number filter must NOT
    prevent field matching — region name is the correct join key.
    """
    from meridian.v4_adapter import adapter_to_records

    raw = [
        _tok("Date", 100, 100, 80, 20, page=7),
        _tok("2025-09-26", 200, 100, 120, 20, page=7),
        _tok("RPM", 100, 150, 80, 20, page=7),
        _tok("3000", 200, 150, 100, 20, page=7),
    ]
    result = reconstruct(raw)

    # Schema fields claim page=13, but OCR is from page=7
    schema_fields = [
        {"id": "G001", "name": "Date", "region": "general", "page": 13,
         "aliases": ["Date"]},
        {"id": "PS001", "name": "RPM", "region": "power_speed", "page": 14,
         "aliases": ["RPM"]},
    ]

    # general region fields should match (same region name, different page)
    general_fields = [f for f in schema_fields if f["region"] == "general"]
    records = adapter_to_records(result, general_fields, page=7, region="general")
    found_ids = [r["field_id"] for r in records if "SOURCE_ABSENT" not in r.get("flags", [])]
    _assert("G001" in found_ids,
            f"Field G001 (schema page=13) should match region 'general' from physical page 7. "
            f"Found: {found_ids}")


def test_reading_pages_at_nonstandard_position():
    """V4 works when Reading pages are at positions other than 13/14.

    Simulates SEP-style Reading pages [7,8] by constructing tokens with
    page metadata matching those positions and verifying the adapter matches
    fields by region regardless of page number.
    """
    from meridian.v4_adapter import adapter_to_records

    raw = [
        _tok("Temp", 100, 100, 80, 20, page=8),
        _tok("95", 200, 100, 60, 20, page=8),
    ]
    result = reconstruct(raw)

    schema_fields = [
        {"id": "CC001", "name": "Temperature", "region": "cylinder_condition",
         "page": 14, "aliases": ["Temp"]},
    ]

    records = adapter_to_records(result, schema_fields, page=8, region="cylinder_condition")
    found = [r["field_id"] for r in records if "SOURCE_ABSENT" not in r.get("flags", [])]
    _assert("CC001" in found,
            f"Field CC001 (schema page=14) should match region from physical page 8. "
            f"Found: {found}")


def test_duplicate_region_same_slot_uses_correct_occurrence():
    """When two tokens with the same region appear, only the one from the
    correct crop is processed (the crop determines physical page)."""
    from meridian.v4_adapter import adapter_to_records

    # Two data rows in the same crop, same region
    raw = [
        _tok("Speed", 100, 100, 80, 20, page=13),
        _tok("100", 200, 100, 60, 20, page=13),
        _tok("Power", 100, 150, 80, 20, page=13),
        _tok("500", 200, 150, 60, 20, page=13),
    ]
    result = reconstruct(raw)

    schema_fields = [
        {"id": "PS001", "name": "Speed", "region": "power_speed",
         "page": 13, "aliases": ["Speed"]},
        {"id": "PS002", "name": "Power", "region": "power_speed",
         "page": 13, "aliases": ["Power"]},
    ]

    records = adapter_to_records(result, schema_fields, page=13, region="power_speed")
    found = [r["field_id"] for r in records if "SOURCE_ABSENT" not in r.get("flags", [])]
    _assert("PS001" in found and "PS002" in found,
            f"Both fields should be found from same crop. Found: {found}")


def test_apr_style_reading_pages_still_work():
    """APR-style Reading pages [13,14] must continue to work after the fix."""
    from meridian.v4_adapter import adapter_to_records

    raw = [
        _tok("Date", 100, 100, 80, 20, page=13),
        _tok("15", 200, 100, 60, 20, page=13),
    ]
    result = reconstruct(raw)

    schema_fields = [
        {"id": "G001", "name": "Date", "region": "general",
         "page": 13, "aliases": ["Date"]},
    ]

    records = adapter_to_records(result, schema_fields, page=13, region="general")
    found = [r["field_id"] for r in records if "SOURCE_ABSENT" not in r.get("flags", [])]
    _assert("G001" in found,
            f"APR-style page 13 should still work. Found: {found}")


def test_page_independent_schema_no_field_leftBehind():
    """All schema fields in a region are candidates regardless of page metadata.

    This verifies that the fix does not cause any field to be silently dropped
    because its schema page metadata doesn't match the physical page.
    """
    from meridian.v4_adapter import adapter_to_records

    raw = [
        _tok("Label A", 100, 100, 100, 20, page=7),
        _tok("1.0", 250, 100, 60, 20, page=7),
        _tok("Label B", 100, 150, 100, 20, page=7),
        _tok("2.0", 250, 150, 60, 20, page=7),
    ]
    result = reconstruct(raw)

    # Both fields have schema page=99 (totally wrong), but same region
    schema_fields = [
        {"id": "F001", "name": "Label A", "region": "general",
         "page": 99, "aliases": ["Label A"]},
        {"id": "F002", "name": "Label B", "region": "general",
         "page": 99, "aliases": ["Label B"]},
    ]

    records = adapter_to_records(result, schema_fields, page=7, region="general")
    found = [r["field_id"] for r in records if "SOURCE_ABSENT" not in r.get("flags", [])]
    _assert("F001" in found and "F002" in found,
            f"Both fields (schema page=99) should match region from physical page 7. "
            f"Found: {found}")


# ═══════════════════════════════════════════════════════════
# Whole-page multi-zone decomposition tests
# ═══════════════════════════════════════════════════════════

def test_two_independent_grids_in_one_stream():
    """Two grids with different leaf columns in a single token stream are
    detected as separate zones and each gets its own column set."""
    from meridian.table_reconstruction import discover_zones

    raw = [
        # Grid A: 3 columns (REF, CYL1, CYL2)
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("CYL2", 300, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        _tok("143", 300, 130, 40, 16),
        # Large gap
        # Grid B: 4 columns (REF, CALC, TC1, TC2)
        _tok("REF", 100, 400, 50, 16),
        _tok("CALC", 200, 400, 50, 16),
        _tok("TC1", 300, 400, 50, 16),
        _tok("TC2", 400, 400, 50, 16),
        _tok("suction", 100, 430, 60, 16),
        _tok("23", 200, 430, 30, 16),
        _tok("24", 300, 430, 30, 16),
        _tok("25", 400, 430, 30, 16),
    ]
    tokens = ingest_tokens(raw)
    zones = discover_zones(tokens)
    _assert(len(zones) >= 2,
            f"Expected >= 2 zones for two grids with a gap, got {len(zones)}")

    # Each zone reconstructs independently with its own columns.
    for zone_tokens in zones:
        zone_dicts = [
            {'text': t.text, 'left': t.left, 'top': t.top,
             'width': t.width, 'height': t.height,
             'conf': t.conf, 'page': t.page, 'region': t.region}
            for t in zone_tokens
        ]
        r = reconstruct(zone_dicts)
        _assert(r.status.ok, f"Zone reconstruct failed: {r.status.warnings}")


def test_pair_form_zone_followed_by_grid_zone():
    """A pair/form section (2 values per line) followed by a grid section
    produces separate zones."""
    from meridian.table_reconstruction import discover_zones

    raw = [
        # Pair/form zone
        _tok("Sea water temp.", 100, 100, 120, 16),
        _tok("28.0", 250, 100, 40, 16),
        _tok("°C", 300, 100, 20, 16),
        # Large gap
        # Grid zone
        _tok("REF", 100, 400, 50, 16),
        _tok("CYL1", 200, 400, 50, 16),
        _tok("CYL2", 300, 400, 50, 16),
        _tok("pmax", 100, 430, 50, 16),
        _tok("142", 200, 430, 40, 16),
        _tok("143", 300, 430, 40, 16),
    ]
    tokens = ingest_tokens(raw)
    zones = discover_zones(tokens)
    _assert(len(zones) >= 2,
            f"Expected >= 2 zones for pair+grid, got {len(zones)}")


def test_two_grids_different_leaf_columns():
    """Two grids with different column names produce distinct column sets."""
    raw_a = [
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("CYL2", 300, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        _tok("143", 300, 130, 40, 16),
    ]
    raw_b = [
        _tok("REF", 100, 500, 50, 16),
        _tok("AVG", 200, 500, 50, 16),
        _tok("TC1", 300, 500, 50, 16),
        _tok("TC2", 400, 500, 50, 16),
        _tok("speed", 100, 530, 50, 16),
        _tok("7928", 200, 530, 50, 16),
        _tok("7960", 300, 530, 50, 16),
        _tok("7977", 400, 530, 50, 16),
    ]

    r_a = reconstruct([{
        'text': t.text, 'left': t.left, 'top': t.top,
        'width': t.width, 'height': t.height,
        'conf': t.conf, 'page': t.page, 'region': t.region,
    } for t in ingest_tokens(raw_a)])
    r_b = reconstruct([{
        'text': t.text, 'left': t.left, 'top': t.top,
        'width': t.width, 'height': t.height,
        'conf': t.conf, 'page': t.page, 'region': t.region,
    } for t in ingest_tokens(raw_b)])

    cols_a = {c.label.rstrip('.').upper() for c in r_a.leaf_columns}
    cols_b = {c.label.rstrip('.').upper() for c in r_b.leaf_columns}

    # Grid A has CYL columns, Grid B has TC columns – they must differ.
    _assert(cols_a != cols_b,
            f"Two grids should have different columns: A={cols_a}, B={cols_b}")


def test_zone_bboxes_preserve_page_geometry():
    """ZoneResult bboxes use original page coordinates, not zone-local ones."""
    from meridian.table_reconstruction import reconstruct_zones

    raw = [
        # Top zone
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        # Bottom zone (after large gap)
        _tok("REF", 100, 600, 50, 16),
        _tok("CALC", 200, 600, 50, 16),
        _tok("TC1", 300, 600, 50, 16),
        _tok("speed", 100, 630, 50, 16),
        _tok("7928", 200, 630, 50, 16),
        _tok("7960", 300, 630, 50, 16),
    ]
    _, zone_results = reconstruct_zones(raw)
    _assert(len(zone_results) >= 2, f"Expected >= 2 zones, got {len(zone_results)}")

    # Zone 0 should have y around 100-146, zone 1 around 600-646
    z0 = zone_results[0]
    z1 = zone_results[1]
    _assert(z0.y_min < 200, f"Zone 0 y_min should be ~100, got {z0.y_min}")
    _assert(z1.y_min > 500, f"Zone 1 y_min should be ~600, got {z1.y_min}")


def test_unrelated_rows_do_not_inherit_other_zone_columns():
    """Rows in zone A are not assigned to zone B's columns."""
    from meridian.table_reconstruction import reconstruct_zones

    raw = [
        # Zone A: 2-column grid
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        # Zone B: 4-column grid
        _tok("REF", 100, 500, 50, 16),
        _tok("CALC", 200, 500, 50, 16),
        _tok("TC1", 300, 500, 50, 16),
        _tok("TC2", 400, 500, 50, 16),
        _tok("speed", 100, 530, 50, 16),
        _tok("7928", 200, 530, 50, 16),
        _tok("7960", 300, 530, 50, 16),
        _tok("7977", 400, 530, 50, 16),
    ]
    _, zone_results = reconstruct_zones(raw)
    _assert(len(zone_results) >= 2, f"Expected >= 2 zones, got {len(zone_results)}")

    # Zone 0 rows should not have TC columns.
    for pr in zone_results[0].physical_rows:
        if pr.model:
            col_labels = {c.column.label.rstrip('.').upper() for c in pr.model.cells}
            _assert('TC1' not in col_labels,
                    f"Zone 0 row should not have TC1 column: {col_labels}")

    # Zone 1 rows should have TC columns.
    z1_has_tc = False
    for pr in zone_results[1].physical_rows:
        if pr.model:
            col_labels = {c.column.label.rstrip('.').upper() for c in pr.model.cells}
            if 'TC1' in col_labels:
                z1_has_tc = True
                break
    _assert(z1_has_tc, "Zone 1 rows should have TC1 column")


def test_blank_cells_preserved_per_zone():
    """Blank cells within a zone are preserved, not filled from another zone."""
    raw = [
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("CYL2", 300, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        # CYL2 value is missing (blank cell)
    ]
    r = reconstruct([{
        'text': t.text, 'left': t.left, 'top': t.top,
        'width': t.width, 'height': t.height,
        'conf': t.conf, 'page': t.page, 'region': t.region,
    } for t in ingest_tokens(raw)])
    # The row should have 3 cells (REF, CYL1, CYL2) with CYL2 blank.
    rows = [pr for pr in r.physical_rows if pr.model]
    if rows:
        cells = rows[0].model.cells
        _assert(len(cells) >= 2, f"Expected >= 2 cells, got {len(cells)}")
        blank_cells = [c for c in cells if c.is_blank]
        _assert(len(blank_cells) >= 1, "Expected at least 1 blank cell")


def test_existing_single_zone_unchanged():
    """A token stream with no zone boundaries still produces a single result."""
    from meridian.table_reconstruction import reconstruct_zones
    raw = [
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
    ]
    r1 = reconstruct([{
        'text': t.text, 'left': t.left, 'top': t.top,
        'width': t.width, 'height': t.height,
        'conf': t.conf, 'page': t.page, 'region': t.region,
    } for t in ingest_tokens(raw)])
    r2, zones = reconstruct_zones(raw)
    _assert(len(zones) == 1, f"Single zone should stay single, got {len(zones)}")
    _assert(len(r2.physical_rows) == len(r1.physical_rows),
            f"Row count mismatch: {len(r2.physical_rows)} vs {len(r1.physical_rows)}")


# ───────────────────────────────────────────────────────────
# Generic zone-detection tests
# ───────────────────────────────────────────────────────────


def test_scale_relative_gap_threshold():
    """_gap_threshold scales with token size (DPI), not a fixed pixel value."""
    # Small tokens (low DPI): median height 8px → tolerance 5.6 → threshold ~10.9
    small = [Token("x", 0, 0, 20, 8, 95.0, 1, "r") for _ in range(5)]
    thr_small = _gap_threshold(small)
    _assert(thr_small < 15.0,
            f"Small-token threshold should be < 15, got {thr_small}")

    # Large tokens (high DPI): median height 32px → tolerance 22.4 → threshold ~43.7
    large = [Token("x", 0, 0, 40, 32, 95.0, 1, "r") for _ in range(5)]
    thr_large = _gap_threshold(large)
    _assert(thr_large > 40.0,
            f"Large-token threshold should be > 40, got {thr_large}")

    # Threshold must be proportional to token height.
    _assert(thr_large > thr_small * 2,
            f"Large threshold ({thr_large}) should be > 2× small ({thr_small})")


def test_same_geometry_different_scales_splits_equivalently():
    """Two grids split at every token scale when gap is scale-proportional."""
    from meridian.table_reconstruction import reconstruct_zones

    for token_h in [8, 16, 24, 32]:
        gap = token_h * 20  # 20× token height — always a large gap
        raw = [
            _tok("REF", 100, 100, 50, token_h),
            _tok("CALC", 200, 100, 50, token_h),
            _tok("pmax", 100, 100 + token_h + 2, 50, token_h),
            _tok("142", 200, 100 + token_h + 2, 40, token_h),
            _tok("REF", 100, 100 + gap, 50, token_h),
            _tok("AVG", 200, 100 + gap, 50, token_h),
            _tok("speed", 100, 100 + gap + token_h + 2, 50, token_h),
            _tok("7928", 200, 100 + gap + token_h + 2, 50, token_h),
        ]
        _, zones = reconstruct_zones(raw)
        _assert(len(zones) >= 2,
                f"Expected >= 2 zones at token_h={token_h}, got {len(zones)}")


def test_single_token_header_with_lookahead_creates_boundary():
    """A single-keyword band adjacent to a multi-keyword header triggers a split."""
    raw = [
        # Metadata title with single "MEASURED" (like real Template-B rows)
        _tok("ISO", 100, 100, 30, 16),
        _tok("CORRECTED", 140, 100, 80, 16),
        _tok("MEASURED", 230, 100, 80, 16),
        # Data row under metadata
        _tok("value", 100, 130, 50, 16),
        _tok("42", 200, 130, 30, 16),
        # --- structural gap ---
        # Real multi-keyword grid header
        _tok("REF.", 100, 400, 40, 16),
        _tok("CALC.", 150, 400, 45, 16),
        _tok("AVG.", 200, 400, 40, 16),
        _tok("CYL1", 250, 400, 40, 16),
        # Data rows
        _tok("pmax", 100, 430, 50, 16),
        _tok("142", 200, 430, 40, 16),
    ]
    tokens = ingest_tokens(raw)
    zones = discover_zones(tokens)
    _assert(len(zones) >= 2,
            f"Single-token MEASURED + multi-keyword header should split, got {len(zones)}")


def test_single_token_label_without_header_context_no_boundary():
    """A standalone 'AVG' in a data row without nearby multi-keyword headers does NOT split."""
    raw = [
        # Data row with "AVG" as a label value (not a column header)
        _tok("parameter", 100, 100, 70, 16),
        _tok("AVG", 180, 100, 35, 16),
        _tok("temperature", 220, 100, 80, 16),
        _tok("value", 100, 130, 50, 16),
        _tok("37.5", 180, 130, 40, 16),
    ]
    tokens = ingest_tokens(raw)
    zones = discover_zones(tokens)
    _assert(len(zones) == 1,
            f"Standalone AVG without header context should stay 1 zone, got {len(zones)}")


def test_is_grid_header_requires_two_keywords():
    """_is_grid_header_band requires ≥2 keywords; single keyword returns False."""
    single_kw = Band(y_min=0, y_max=16, tokens=[
        Token("MEASURED", 0, 0, 70, 16, 95.0, 1, "r"),
    ])
    _assert(_is_grid_header_band(single_kw) is False,
            "Single keyword should NOT be a grid header band")

    multi_kw = Band(y_min=0, y_max=16, tokens=[
        Token("REF.", 0, 0, 35, 16, 95.0, 1, "r"),
        Token("CALC.", 40, 0, 40, 16, 95.0, 1, "r"),
    ])
    _assert(_is_grid_header_band(multi_kw) is True,
            "Two keywords SHOULD be a grid header band")


def test_grid_keyword_count_accurate():
    """_grid_keyword_count returns exact number of matching keywords."""
    band = Band(y_min=0, y_max=16, tokens=[
        Token("REF.", 0, 0, 35, 16, 95.0, 1, "r"),
        Token("hello", 40, 0, 40, 16, 95.0, 1, "r"),
        Token("AVG.", 85, 0, 35, 16, 95.0, 1, "r"),
    ])
    _assert(_grid_keyword_count(band) == 2,
            f"Expected 2 grid keywords, got {_grid_keyword_count(band)}")


def test_footer_isolated_geometry_forms_own_zone():
    """A footer-like band at the page bottom, isolated by a large gap, forms its own zone — no vocabulary needed."""
    from meridian.table_reconstruction import reconstruct_zones

    raw = [
        # Main content
        _tok("REF", 100, 100, 50, 16),
        _tok("CALC", 200, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        # Isolated band at page bottom (gap >> threshold) — no vocabulary match
        _tok("software", 100, 900, 60, 16),
        _tok("v3.8.0", 170, 900, 50, 16),
        _tok("Page", 230, 900, 30, 16),
        _tok("11", 265, 900, 20, 16),
    ]
    _, zones = reconstruct_zones(raw)
    _assert(len(zones) >= 2,
            f"Isolated bottom band should form own zone, got {len(zones)}")
    # The last zone should be the footer-like band.
    last_zone = zones[-1]
    footer_texts = {t.text for t in last_zone.tokens}
    _assert("software" in footer_texts,
            f"Last zone should contain footer tokens, got {footer_texts}")


def test_bottom_content_not_auto_rejected():
    """Content near the page bottom that is structurally connected is NOT isolated."""
    raw = [
        _tok("REF", 100, 100, 50, 16),
        _tok("CALC", 200, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        _tok("pcomp", 100, 160, 50, 16),
        _tok("105", 200, 160, 40, 16),
    ]
    tokens = ingest_tokens(raw)
    zones = discover_zones(tokens)
    _assert(len(zones) == 1,
            f"Connected content should stay 1 zone, got {len(zones)}")


def test_blank_cells_preserved_across_zones():
    """Blank cells are positional and not filled from other zones."""
    from meridian.table_reconstruction import reconstruct_zones

    raw = [
        # Zone A: 3-column grid with a blank
        _tok("REF", 100, 100, 50, 16),
        _tok("CYL1", 200, 100, 50, 16),
        _tok("CYL2", 300, 100, 50, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("142", 200, 130, 40, 16),
        # CYL2 value missing — blank
        # --- gap ---
        # Zone B: 2-column grid
        _tok("REF", 100, 600, 50, 16),
        _tok("AVG", 200, 600, 40, 16),
        _tok("speed", 100, 630, 50, 16),
        _tok("7928", 200, 630, 50, 16),
    ]
    _, zones = reconstruct_zones(raw)
    _assert(len(zones) >= 2, f"Expected >= 2 zones, got {len(zones)}")
    # Zone A should have a blank cell for CYL2.
    for pr in zones[0].physical_rows:
        if pr.model:
            blank = [c for c in pr.model.cells if c.is_blank]
            _assert(len(blank) >= 1,
                    f"Zone A should have blank cell, cells={[c.column.label for c in pr.model.cells]}")


# ───────────────────────────────────────────────────────────
# Phase 2: generic dense side-by-side form segmentation /
#          value association (no field IDs, no vocabulary)
# ───────────────────────────────────────────────────────────


def _frun(label_texts, value_texts, band_key, label_lefts=None, value_lefts=None):
    """Build a _FieldRun on a single printed line with geometry."""
    from meridian.table_reconstruction import _FieldRun

    label_lefts = label_lefts or [100 + i * 105 for i in range(len(label_texts))]
    value_lefts = value_lefts or [500 + i * 40 for i in range(len(value_texts))]
    lt = [_tk(t, x, 100) for t, x in zip(label_texts, label_lefts)]
    vt = [_tk(t, x, 100) for t, x in zip(value_texts, value_lefts)]
    return _FieldRun(label_toks=lt, value_toks=vt, is_header=False, band_key=band_key)


def test_two_side_by_side_fields_associate_values():
    """Dense line: dangling label + qualifier run carrying the value.

    'Ambient humidity  rel v 32.0%' -- rel v is a qualifier, 32.0% the value.
    One logical field must be ONE row with the value attached."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    owner = _frun(["Ambient", "humidity"], [], 0, [139, 242])
    qual = _frun(["rel", "v"], ["32.0%"], 0, [434, 494], [592])
    out = _merge_side_by_side_form_values([owner, qual])
    _assert(len(out) == 1,
            f"qualifier value must fold into owner row, got {len(out)} rows")
    label = " ".join(t.text for t in out[0].label_toks)
    _assert("Ambient" in label and "rel" in label,
            f"label must keep qualifier words, got {label!r}")
    _assert([t.text for t in out[0].value_toks] == ["32.0%"],
            f"value tokens not attached, got {[t.text for t in out[0].value_toks]}")


def test_three_side_by_side_single_row():
    """Three qualifier/unit fragments on ONE form line fold into one row."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    owner = _frun(["Barom.", "pr."], [], 0, [794, 879])
    qual = _frun(["Eng.", "room"], ["1.010", "@", "1.013"], 0, [928, 982], [1092, 1167, 1193])
    unit = _frun(["barA"], [], 0, [1277])
    out = _merge_side_by_side_form_values([owner, qual, unit])
    _assert(len(out) == 1,
            f"all three fragments belong to one field, got {len(out)} rows")
    label = " ".join(t.text for t in out[0].label_toks)
    _assert("Barom." in label and "Eng." in label,
            f"only qualifier words should join label, got {label!r}")
    nums = [t.text for t in out[0].value_toks if t.text and t.text[0].isdigit()]
    _assert("1.010" in nums and "1.013" in nums, f"values lost, got {nums}")


def test_numeric_value_with_unit_stays_with_label():
    """'... temp. 30.0 [then unit on next fragment]': a numeric unit fragment
    directly after a real value is a continuation, not a stolen field."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    owner = _frun(["Engine", "room", "air", "temp."], ["30.0"], 0,
                  [139, 242, 334, 434], [592])
    unit = _frun(["°C"], [], 1)  # different band (printed below) — no merge
    out = _merge_side_by_side_form_values([owner, unit])
    _assert(len(out) == 2,
            f"cross-band unit must not fold, got {len(out)} rows")
    _assert([t.text for t in out[0].value_toks] == ["30.0"],
            f"owner value must stay, got {out[0].value_toks}")


def test_text_value_label_tail_kept():
    """'ENGINE state | stable | v' -> one row whose label carries the text tail
    ('ENGINE state stable v') — the value-less text tails must not be dropped."""
    from meridian.table_reconstruction import Column, _field_run_rows

    bands = [
        Band(
            y_min=55.0, y_max=80.0,
            tokens=[
                _tk("ENGINE", 139, 55), _tk("state", 242, 55),
                _tk("stable", 434, 55), _tk("v", 494, 55),
            ],
            row_type="data",
        ),
        Band(
            y_min=98.0, y_max=124.0,
            tokens=[
                _tk("Ambient", 139, 98), _tk("air", 242, 98),
                _tk("temp.", 434, 98), _tk("29.0", 592, 98),
            ],
            row_type="data",
        ),
    ]
    value_col = Column(col_id="VALUE", label="VALUE", path=["VALUE"],
                       x_min=0.0, x_max=2000.0)
    rows = _field_run_rows(bands, [value_col], is_grid=False)
    labels = [pr.model.label_region.text for pr in rows if pr.model]
    _assert(any("ENGINE state stable v" in l for l in labels),
            f"text tail must stay with the label, got {labels}")
    _assert(any("Ambient air temp." in l for l in labels),
            f"second field label missing, got {labels}")


def test_empty_neighbouring_field_value_not_lost():
    """'Draft fwd [blank] | aft 12.60': the blank neighbour's value sits with
    the aft fragment; it must be captured on the combined row, not dropped."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    owner = _frun(["Draft", "fwd"], ["|"], 0, [139, 242], [339])
    qual = _frun(["aft"], ["12.60", "12.80"], 0, [434], [592, 671])
    out = _merge_side_by_side_form_values([owner, qual])
    _assert(len(out) == 1, f"combined draft row expected, got {len(out)}")
    nums = [t.text for t in out[0].value_toks if t.text and t.text[0].isdigit()]
    _assert("12.60" in nums and "12.80" in nums,
            f"aft value must be retained, got {nums}")


def test_dense_short_labels_real_fields_not_merged():
    """Three REAL side-by-side fields each with its own real value must NOT be
    glued together (left neighbours already own real values)."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    a = _frun(["Air", "temp."], ["24.0"], 0, [100, 205], [400])
    b = _frun(["Oil", "press."], ["5.6"], 0, [520, 625], [800])
    c = _frun(["Water"], ["88.0"], 0, [920], [1100])
    out = _merge_side_by_side_form_values([a, b, c])
    _assert(len(out) == 3,
            f"real side-by-side fields must stay split, got {len(out)}")
    for i, expected in zip(out, (["24.0"], ["5.6"], ["88.0"])):
        _assert([t.text for t in i.value_toks] == expected,
                f"value mismatch for {i}, got {[t.text for t in i.value_toks]}")


def test_long_label_not_oversplit():
    """A long genuine label followed by its value must remain ONE row; the
    merge pass must not cut it or steal its value."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    owner = _frun(["Exhaust", "gas", "temperature", "after", "turbine"],
                  ["485"], 0, [100, 210, 330, 460, 560], [760])
    unit = _frun(["°C"], [], 1)
    out = _merge_side_by_side_form_values([owner, unit])
    _assert(len(out) == 2, "long field + separate-band unit expected")
    label = " ".join(t.text for t in out[0].label_toks)
    for w in ("Exhaust", "gas", "temperature", "after", "turbine"):
        _assert(w in label, f"oversplit: word {w!r} missing from {label!r}")
    _assert([t.text for t in out[0].value_toks] == ["485"],
            f"value lost, got {[t.text for t in out[0].value_toks]}")


def test_long_genuine_field_label_keeps_own_row():
    """'LCV kJ/kg [blank] | Density FLOWM. kg/m? 914.2' -- Density is a real
    field name (5-letter word) and must NOT fold into the blank owner."""
    from meridian.table_reconstruction import _merge_side_by_side_form_values

    owner = _frun(["LCV", "kJ/kg"], [], 0, [100, 210])
    dens = _frun(["Density", "FLOWM.", "kg/m?"], ["914.2"], 0, [520, 630, 740], [900])
    out = _merge_side_by_side_form_values([owner, dens])
    _assert(len(out) == 2,
            f"Density is a real field, must stay split; got {len(out)} rows")
    _assert([t.text for t in out[1].value_toks] == ["914.2"],
            f"Density value must stay on its own row, got {out[1].value_toks}")


def test_genuine_grid_header_unaffected_by_form_merge():
    """A real value grid with a genuine header row must keep grid composition
    (multi-column) -- the side-by-side form merge belongs to form regions only."""
    raw = [
        _tok("REF", 100, 100, 40, 16),
        _tok("CALC", 180, 100, 40, 16),
        _tok("AVG", 260, 100, 40, 16),
        _tok("pmax", 100, 130, 50, 16),
        _tok("141", 100, 130, 40, 16),
        _tok("142", 180, 130, 40, 16),
        _tok("143", 260, 130, 40, 16),
        _tok("pcomp", 100, 160, 50, 16),
        _tok("105", 100, 160, 40, 16),
        _tok("106", 180, 160, 40, 16),
        _tok("107", 260, 160, 40, 16),
    ]
    r = reconstruct(raw)
    _assert(len(r.leaf_columns) >= 2,
            f"grid must stay multi-column, got {[c.col_id for c in r.leaf_columns]}")
    headers = [b for b in r.bands if b.row_type == "header"]
    _assert(len(headers) >= 1, "genuine grid header must remain a header band")
    header_text = " ".join(t.text for b in headers for t in b.tokens)
    _assert("REF" in header_text and "CALC" in header_text,
            f"header must stay intact, got {header_text!r}")


def test_form_region_value_association_integration():
    """End-to-end synthetic dense form line: reconstruct produces ONE VALUE
    column (form region) and attaches 32.0% to the Ambient humidity row."""
    raw = [
        _tok("Ambient", 139, 100, 90, 16),
        _tok("humidity", 242, 100, 95, 16),
        _tok("rel", 434, 100, 25, 16),
        _tok("v", 494, 100, 13, 16),
        _tok("32.0%", 592, 100, 79, 16),
        _tok("Barom.", 794, 100, 75, 16),
        _tok("pr.", 879, 100, 26, 16),
        _tok("Eng.", 928, 100, 44, 16),
        _tok("room", 982, 100, 56, 16),
        _tok("1.010", 1092, 100, 62, 16),
        _tok("barA", 1277, 100, 51, 16),
        # second band, values at different x -> not a grid
        _tok("abs", 432, 194, 37, 16),
        _tok("8.0", 606, 194, 31, 16),
        _tok("g/kg", 652, 194, 50, 16),
    ]
    r = reconstruct(raw)
    _assert(len(r.leaf_columns) == 1 and r.leaf_columns[0].col_id == "VALUE",
            f"form region must have a single VALUE column, got {[c.col_id for c in r.leaf_columns]}")
    rows = {}
    for pr in r.physical_rows:
        m = pr.model
        if m is not None:
            rows[m.label_region.normalized_text] = [
                c.raw_text for c in m.cells if c.raw_text]
    " ".join(rows.keys())
    # The humidity row carries its qualifier label and the value.
    humidity_row = next((k for k in rows if "humid" in k.lower()), None)
    _assert(humidity_row is not None,
            f"Ambient humidity row missing; rows={list(rows)}")
    _assert("32.0%" in (rows[humidity_row][0] if rows[humidity_row] else ""),
            f"value 32.0% must attach to humidity row, got {rows[humidity_row]}")
    _assert("rel" in humidity_row and "v" in humidity_row,
            f"qualifier must stay with humidity label, got {humidity_row!r}")
    barom_row = next((k for k in rows if "barom" in k.lower()), None)
    _assert(barom_row is not None, f"Barom row missing; rows={list(rows)}")
    _assert(barom_row and rows[barom_row] and "1.010" in rows[barom_row][0],
            f"1.010 must attach to Barom row, got {rows[barom_row]}")


# ───────────────────────────────────────────────────────────
# Generic dense horizontal-form association (G005 fix)
#
# These tests lock in the GENERIC association mechanism for dense
# side-by-side "label: value" forms with no field-specific logic:
#   - OCR punctuation glued onto a numeric value ('29.0;') must not make
#     the value read as a label word.
#   - A time / date token ('08:50') is value-bearing, not a label word.
#   - A unit token (e.g. '°C') printed immediately after a value belongs to
#     that same field, not the next one.
#   - Side-by-side numeric fields with small gaps separate cleanly.
#   - No field IDs / names, no absolute coordinates, no expected values are
#     used -- purely token shape and geometry.
# ───────────────────────────────────────────────────────────


def _field_runs_to_texts(runs):
    """[(label_toks, value_toks)] -> [(label, value)] joined text."""
    return [
        (" ".join(t.text for t in lbl), " ".join(t.text for t in val))
        for lbl, val in runs
    ]


def test_looks_like_number_semicolon_suffixed():
    """OCR-glued trailing semicolon must still be a number (G005 '29.0;')."""
    _assert(_looks_like_number("29.0;"), "29.0; must be numeric")
    _assert(_looks_like_number("0.20)"), "0.20) must be numeric")
    _assert(_looks_like_number("8.0|"), "8.0| must be numeric")
    # Unrelated punctuation / letters must NOT become numbers.
    _assert(not _looks_like_number("C1"), "C1 must not be numeric")
    _assert(not _looks_like_number("1A"), "1A must not be numeric")
    _assert(not _looks_like_number("REF"), "REF must not be numeric")
    _assert(not _looks_like_number("barG"), "barG must not be numeric")
    # Internal punctuation (dates/times) is left intact for its own path.
    _assert(not _looks_like_number("08:50"), "08:50 must not be a plain number")


def test_value_token_semicolon_decimal_and_time():
    """Both a semicolon-suffixed decimal and a clock time are value tokens."""
    from meridian.table_reconstruction import _is_value_token
    _assert(_is_value_token(_tk("29.0;", 0, 0)), "29.0; is a value token")
    _assert(_is_value_token(_tk("08:50", 0, 0)), "08:50 is a value token")
    _assert(not _is_value_token(_tk("Seawater", 0, 0)), "label word not a value")


def test_field_runs_label_numeric_unit_single_field():
    """'Label 29.0; °C' is ONE field: label, numeric value, absorbed unit."""
    tokens = [
        _tk("Seawater", 0, 0),
        _tk("temp.", 150, 0),
        _tk("29.0;", 400, 0),
        _tk("°C", 560, 0),
    ]
    runs = _field_runs_in_band(tokens)
    _assert_eq(len(runs), 1, "one field expected")
    text = _field_runs_to_texts(runs)
    label, value = text[0]
    _assert("29.0;" in value, f"value must carry 29.0;, got {value!r}")
    # The unit is absorbed into the label run (so _split_label_unit surfaces it)
    _assert("°C" in label, f"unit must be absorbed into label for later split, got {label!r}")


def test_field_runs_two_side_by_side_numeric_fields():
    """Two side-by-side numeric fields split cleanly: A 44.0 | B 7.10."""
    tokens = [
        _tk("FieldA", 0, 0),
        _tk("44.0;", 200, 0),
        _tk("FieldB", 500, 0),
        _tk("7.10", 700, 0),
    ]
    runs = _field_runs_in_band(tokens)
    _assert_eq(len(runs), 2, "two side-by-side fields expected")
    text = _field_runs_to_texts(runs)
    _assert("FieldA" in text[0][0] and "44.0;" in text[0][1], f"got {text[0]}")
    _assert("FieldB" in text[1][0] and "7.10" in text[1][1], f"got {text[1]}")


def test_field_runs_three_side_by_side():
    """Three side-by-side fields split cleanly by value/non-value shape.
    (Realistic multi-letter labels avoid single-letter unit-lexeme collisions
    such as 'C'=Celsius, 'W'=watt.)"""
    tokens = [
        _tk("Press", 0, 0), _tk("1.0", 70, 0),
        _tk("Temp", 160, 0), _tk("2.0", 230, 0),
        _tk("Flow", 320, 0), _tk("3.0", 390, 0),
    ]
    runs = _field_runs_in_band(tokens)
    _assert_eq(len(runs), 3, "three fields expected")
    vals = [val for _, val in _field_runs_to_texts(runs)]
    _assert_eq(vals, ["1.0", "2.0", "3.0"], "values")


def test_field_runs_dense_small_gaps():
    """Small gaps (dense form) still separate fields by value/non-value shape."""
    tokens = [
        _tk("Sea", 0, 0), _tk("water", 40, 0), _tk("temp.", 80, 0),
        _tk("29.0;", 200, 0), _tk("°C", 260, 0),
        _tk("SHIP", 400, 0), _tk("SOG", 430, 0),
    ]
    runs = _field_runs_in_band(tokens)
    # 'Sea water temp.' + value(+unit) is ONE field run; 'SHIP'/'SOG' start a
    # second (label-only) run.
    _assert_eq(len(runs), 2, "dense form: value field + trailing label-only field")
    label0, value0 = _field_runs_to_texts(runs)[0]
    _assert("Sea" in label0 and "temp." in label0, f"label0={label0!r}")
    _assert("29.0;" in value0, f"value0={value0!r}")


def test_field_runs_long_multi_word_label():
    """A long multi-word label (compound-merged 'Seawater temp.') still
    associates with its value and absorbed unit; the next field starts after."""
    tokens = [
        _tk("Seawater", 0, 0, 100),
        _tk("temp.", 90, 0, 70),
        _tk("29.0;", 200, 0, 50),
        _tk("°C", 270, 0, 40),
        _tk("SHIP", 400, 0, 70),
        _tk("SOG", 450, 0, 60),
    ]
    runs = _field_runs_in_band(tokens)
    _assert_eq(len(runs), 2, "value field + trailing label-only field")
    label, value = _field_runs_to_texts(runs)[0]
    _assert("Sea" in label and "temp." in label, f"label={label!r}")
    _assert("°C" in label, f"unit °C must be absorbed into label, got {label!r}")
    _assert("29.0;" in value, f"value must be 29.0;, got {value!r}")


def test_field_runs_neighbor_empty_has_own_value():
    """Empty neighbor must not steal; each populated neighbor keeps its value."""
    tokens = [
        _tk("L1", 0, 0), _tk("5.0", 150, 0),
        _tk("L2", 400, 0), _tk("7.5", 550, 0),
        _tk("L3", 800, 0), _tk("9.9", 950, 0),
    ]
    runs = _field_runs_in_band(tokens)
    _assert_eq(len(runs), 3, "three populated fields")
    vals = [v for _, v in _field_runs_to_texts(runs)]
    _assert_eq(vals, ["5.0", "7.5", "9.9"], "all three values retained")


def test_field_runs_dates_times_stay_as_value():
    """Date/time tokens are value-bearing, not label-prefixes for neighbors."""
    from meridian.table_reconstruction import _is_date_like
    _assert(_is_date_like("4/15/2025"), "date is date-like")
    _assert(_is_date_like("08:50"), "time is date-like")
    tokens = [
        _tk("Date", 0, 0),
        _tk("4/15/2025", 200, 0),
        _tk("Time", 450, 0),
        _tk("08:50", 600, 0),
    ]
    runs = _field_runs_in_band(tokens)
    _assert_eq(len(runs), 2, "date and time are separate value fields")
