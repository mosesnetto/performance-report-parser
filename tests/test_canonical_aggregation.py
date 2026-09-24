"""Generic tests for zone-aware canonical aggregation.

Covers the integration between reconstruct_zones → zone-level
ReconstructionResult → build_field_records_wholepage → final canonical
output.  Tests are synthetic and do NOT use PDF-specific values.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian.table_reconstruction import (
    Cell,
    Column,
    PhysicalRow,
    Region,
    RegionType,
    RowModel,
    Token,
    ZoneResult,
)
from meridian.v4_adapter import (
    EXTRACTION_FAILED,
    SOURCE_BLANK,
    aggregate_field_records_across_pages,
    build_field_records_canonical,
)

# ─── helpers ──────────────────────────────────────────────

def _tok(text, left, top, conf=95.0, width=40, height=16, page=1, region="reading"):
    return Token(text=text, left=left, top=top, width=width, height=height,
                 conf=conf, page=page, region=region)


def _col(col_id, label, x_min, x_max, path=None):
    return Column(col_id=col_id, label=label, path=path or [label],
                  x_min=x_min, x_max=x_max)


def _cell(col, text, blank=False, left=600, conf=95.0, status=None):
    tokens = [] if blank else [_tok(text, left, 100, conf=conf)]
    return Cell(row_idx=0, column=col, tokens=tokens, is_blank=blank,
                status=status or [])


def _row(idx, label_tokens, cells, unit_tokens=None, quals=None, y=100):
    lab = Region(RegionType.LABEL, label_tokens)
    unit = Region(RegionType.UNIT, unit_tokens or [])
    qs = [Region(RegionType.QUALIFIER, [t]) for t in (quals or [])]
    return RowModel(
        row_idx=idx,
        bbox=(50.0, float(y), 600.0, float(y + 20)),
        label_region=lab,
        unit_region=unit,
        qualifier_regions=qs,
        cells=cells,
        status_indicators=[s for c in cells for s in c.status],
        status_metadata={},
    )


def _zone_result(zone_idx, rows, columns, page=1, y_min=100, y_max=500, tokens=None):
    """Create a minimal ZoneResult."""
    physical = [PhysicalRow(row_idx=i, y_min=100 + i * 20, y_max=120 + i * 20,
                            row_type="data", model=r)
                for i, r in enumerate(rows)]
    return ZoneResult(
        zone_idx=zone_idx,
        page=page,
        y_min=y_min,
        y_max=y_max,
        physical_rows=physical,
        header_bands=[],
        leaf_columns=columns,
        header_hierarchy=[],
        tokens=tokens or [],
    )


# ─── Schema definitions (generic, no PDF values) ─────────

SCHEMA_AB = [
    {"id": "XTEST1", "name": "Field Alpha", "aliases": ["Field Alpha"], "unit": "bar"},
    {"id": "XTEST2", "name": "Field Beta", "aliases": ["Field Beta"], "unit": "°C"},
]

SCHEMA_ABC = [
    {"id": "XTEST1", "name": "Field Alpha", "aliases": ["Field Alpha"], "unit": "bar"},
    {"id": "XTEST2", "name": "Field Beta", "aliases": ["Field Beta"], "unit": "°C"},
    {"id": "XTEST3", "name": "Field Gamma", "aliases": ["Field Gamma"], "unit": "kW"},
]

SCHEMA_TEXT = [
    {"id": "XTEST4", "name": "Running mode", "aliases": ["Running mode"], "unit": "",
     "allow_text_values": True},
    {"id": "XTEST5", "name": "Numeric field", "aliases": ["Numeric field"], "unit": "bar"},
]


# ─── Test 1: two pages → one canonical field set ─────────

def test_two_pages_one_canonical_set():
    """Two pages each producing F001 and F002 must merge into exactly
    one F001 and one F002 (no duplicates)."""
    cols_p1 = [_col("c0", "COL1", 600, 640)]
    row_p1 = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols_p1[0], "10.0")],
                  unit_tokens=[_tok("bar", 400, 100)])
    zone_p1 = _zone_result(0, [row_p1], cols_p1, page=1)

    cols_p2 = [_col("c0", "COL1", 600, 640)]
    row_p2 = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols_p2[0], "20.0")],
                  unit_tokens=[_tok("bar", 400, 100)])
    zone_p2 = _zone_result(0, [row_p2], cols_p2, page=2)

    frs_p1 = build_field_records_canonical([zone_p1], SCHEMA_AB, page=1)
    frs_p2 = build_field_records_canonical([zone_p2], SCHEMA_AB, page=2)

    merged = aggregate_field_records_across_pages([frs_p1, frs_p2], SCHEMA_AB)

    fids = [fr.field_id for fr in merged]
    assert len(fids) == 2, f"expected 2 fields, got {len(fids)}"
    assert set(fids) == {"XTEST1", "XTEST2"}, f"unexpected field IDs: {fids}"
    print("  PASS: test_two_pages_one_canonical_set")


# ─── Test 2: one field appearing on one page only ────────

def test_field_on_one_page_only():
    """F003 only exists on page 2. Page 1 emits SOURCE_ABSENT; page 2
    emits extracted.  Canonical must show extracted, not absent."""
    cols = [_col("c0", "COL1", 600, 640)]
    row = _row(0, [_tok("Field Gamma", 100, 100)], [_cell(cols[0], "99.0")],
               unit_tokens=[_tok("kW", 400, 100)])
    zone_p2 = _zone_result(0, [row], cols, page=2)

    frs_p1 = build_field_records_canonical([], SCHEMA_ABC, page=1)
    frs_p2 = build_field_records_canonical([zone_p2], SCHEMA_ABC, page=2)

    merged = aggregate_field_records_across_pages([frs_p1, frs_p2], SCHEMA_ABC)
    f003 = [fr for fr in merged if fr.field_id == "XTEST3"]
    assert len(f003) == 1
    assert f003[0].found, "XTEST3 should be found on page 2"
    vals = [c.value for c in f003[0].cells if c.value is not None]
    assert 99.0 in vals, f"XTEST3 should have value 99.0, got {vals}"
    print("  PASS: test_field_on_one_page_only")


# ─── Test 3: same schema region on multiple zones ────────

def test_field_in_multiple_zones():
    """F001 appears in zone A with value 10.0 and zone B with value 20.0.
    The aggregation must pick one (the one with more non-blank cells)."""
    cols_a = [_col("c0", "COL1", 600, 640), _col("c1", "COL2", 650, 690)]
    row_a = _row(0, [_tok("Field Alpha", 100, 100)],
                 [_cell(cols_a[0], "10.0"), _cell(cols_a[1], "11.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_a = _zone_result(0, [row_a], cols_a, page=1)

    cols_b = [_col("c0", "XCOL", 600, 640)]
    row_b = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols_b[0], "20.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_b = _zone_result(1, [row_b], cols_b, page=1)

    frs = build_field_records_canonical([zone_a, zone_b], SCHEMA_AB, page=1)
    f001 = [fr for fr in frs if fr.field_id == "XTEST1"]
    assert len(f001) == 1
    # zone_a has 2 cells, zone_b has 1 cell → zone_a wins
    assert len(f001[0].cells) == 2, f"expected 2 cells from zone_a, got {len(f001[0].cells)}"
    cols = {c.column for c in f001[0].cells}
    assert "COL1" in cols and "COL2" in cols, f"expected zone_a columns, got {cols}"
    print("  PASS: test_field_in_multiple_zones")


# ─── Test 4: extracted beats blank ───────────────────────

def test_extracted_beats_blank():
    """Zone A has actual data for F001, zone B has only blanks.
    Canonical must pick zone A's extracted record."""
    cols_a = [_col("c0", "COL1", 600, 640)]
    row_a = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols_a[0], "10.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_a = _zone_result(0, [row_a], cols_a, page=1)

    cols_b = [_col("c0", "COL1", 600, 640)]
    row_b = _row(0, [_tok("Field Alpha", 100, 100)],
                 [_cell(cols_b[0], None, blank=True)],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_b = _zone_result(1, [row_b], cols_b, page=1)

    frs = build_field_records_canonical([zone_a, zone_b], SCHEMA_AB, page=1)
    f001 = [fr for fr in frs if fr.field_id == "XTEST1"][0]
    assert f001.found, "should be found"
    vals = [c.value for c in f001.cells if c.value is not None]
    assert 10.0 in vals, f"should have extracted value 10.0, got {vals}"
    assert not any(SOURCE_BLANK in c.flags for c in f001.cells
                   if c.value is not None), "non-blank cell should not have SOURCE_BLANK"
    print("  PASS: test_extracted_beats_blank")


# ─── Test 5: unrelated zone columns are never inherited ──

def test_unrelated_zone_columns_not_inherited():
    """Zone A has columns [X1, X2] for field F001.
    Zone B has columns [Y1, Y2] for field F002.
    F001 must only have columns from zone A, never Y1/Y2."""
    cols_a = [_col("c0", "X1", 600, 640), _col("c1", "X2", 650, 690)]
    row_a = _row(0, [_tok("Field Alpha", 100, 100)],
                 [_cell(cols_a[0], "10.0"), _cell(cols_a[1], "11.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_a = _zone_result(0, [row_a], cols_a, page=1)

    cols_b = [_col("c0", "Y1", 600, 640), _col("c1", "Y2", 650, 690)]
    row_b = _row(0, [_tok("Field Beta", 100, 100)],
                 [_cell(cols_b[0], "30.0"), _cell(cols_b[1], "31.0")],
                 unit_tokens=[_tok("°C", 400, 100)])
    zone_b = _zone_result(1, [row_b], cols_b, page=1)

    frs = build_field_records_canonical([zone_a, zone_b], SCHEMA_AB, page=1)
    f001 = [fr for fr in frs if fr.field_id == "XTEST1"]
    f002 = [fr for fr in frs if fr.field_id == "XTEST2"]

    cols_001 = {c.column for c in f001[0].cells}
    cols_002 = {c.column for c in f002[0].cells}

    assert cols_001 == {"X1", "X2"}, f"XTEST1 should have X columns, got {cols_001}"
    assert cols_002 == {"Y1", "Y2"}, f"XTEST2 should have Y columns, got {cols_002}"
    assert "Y1" not in cols_001, "F001 must not inherit Y1 from zone B"
    assert "X1" not in cols_002, "F002 must not inherit X1 from zone A"
    print("  PASS: test_unrelated_zone_columns_not_inherited")


# ─── Test 6: independent grids keep independent columns ──

def test_independent_grids_keep_columns():
    """Zone A is a 3-column grid, zone B is a 2-column grid.
    Each zone's fields retain their zone-specific columns."""
    cols_a = [_col("c0", "REF", 600, 630), _col("c1", "CALC", 640, 670),
              _col("c2", "AVG", 680, 710)]
    row_a = _row(0, [_tok("Field Alpha", 100, 100)],
                 [_cell(cols_a[0], "1.0"), _cell(cols_a[1], "2.0"),
                  _cell(cols_a[2], "3.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_a = _zone_result(0, [row_a], cols_a, page=1)

    cols_b = [_col("c0", "CYL1", 600, 640), _col("c1", "CYL2", 650, 690)]
    row_b = _row(0, [_tok("Field Beta", 100, 100)],
                 [_cell(cols_b[0], "4.0"), _cell(cols_b[1], "5.0")],
                 unit_tokens=[_tok("°C", 400, 100)])
    zone_b = _zone_result(1, [row_b], cols_b, page=1)

    frs = build_field_records_canonical([zone_a, zone_b], SCHEMA_AB, page=1)
    f001 = [fr for fr in frs if fr.field_id == "XTEST1"]
    f002 = [fr for fr in frs if fr.field_id == "XTEST2"]

    assert {c.column for c in f001[0].cells} == {"REF", "CALC", "AVG"}
    assert {c.column for c in f002[0].cells} == {"CYL1", "CYL2"}
    print("  PASS: test_independent_grids_keep_columns")


# ─── Test 7: blank cells remain positional ───────────────

def test_blank_cells_positional():
    """A blank CYL2 in a 3-column grid must not shift CYL3 left."""
    cols = [_col("c0", "CYL1", 600, 630), _col("c1", "CYL2", 640, 670),
            _col("c2", "CYL3", 680, 710)]
    row = _row(0, [_tok("Field Alpha", 100, 100)],
               [_cell(cols[0], "1.0"), _cell(cols[1], None, blank=True),
                _cell(cols[2], "3.0")],
               unit_tokens=[_tok("bar", 400, 100)])
    zone = _zone_result(0, [row], cols, page=1)

    frs = build_field_records_canonical([zone], SCHEMA_AB, page=1)
    f001 = [fr for fr in frs if fr.field_id == "XTEST1"][0]

    by_col = {c.column: c for c in f001.cells}
    assert by_col["CYL1"].value == 1.0
    assert by_col["CYL2"].value is None
    assert SOURCE_BLANK in by_col["CYL2"].flags
    assert by_col["CYL3"].value == 3.0
    print("  PASS: test_blank_cells_positional")


# ─── Test 8: text-valued fields not forced through numeric parsing ──

def test_text_value_not_forced_numeric():
    """A text-value field (allow_text_values=true) with non-numeric
    content must retain the raw text and not get EXTRACTION_FAILED."""
    cols = [_col("c0", "VALUE", 600, 900)]
    row = _row(0, [_tok("Running mode", 100, 100)],
               [_cell(cols[0], "Economy")],
               unit_tokens=[])
    zone = _zone_result(0, [row], cols, page=1)

    frs = build_field_records_canonical([zone], SCHEMA_TEXT, page=1)
    t001 = [fr for fr in frs if fr.field_id == "XTEST4"]
    assert len(t001) == 1
    t001 = t001[0]
    assert t001.found
    raw_cells = [c for c in t001.cells if c.raw_text.strip()]
    assert any("Economy" in c.raw_text for c in raw_cells), \
        f"raw text 'Economy' must be preserved, got {[c.raw_text for c in t001.cells]}"
    assert not any(EXTRACTION_FAILED in c.flags for c in t001.cells), \
        "text field must not get EXTRACTION_FAILED"
    print("  PASS: test_text_value_not_forced_numeric")


# ─── Test 9: final aggregation produces exactly one record per field ──

def test_exactly_one_record_per_schema_field():
    """Regardless of how many pages/zones produce candidates, the final
    aggregation must produce exactly one FieldRecord per schema field."""
    schema = SCHEMA_ABC

    # Page 1: zone A matches F001, F002
    cols_a = [_col("c0", "COL1", 600, 640)]
    row_a = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols_a[0], "1.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_a1 = _zone_result(0, [row_a], cols_a, page=1)

    # Page 1: zone B matches F001 again (different zone)
    cols_b = [_col("c0", "COL2", 600, 640)]
    row_b = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols_b[0], "2.0")],
                 unit_tokens=[_tok("bar", 400, 100)])
    zone_b1 = _zone_result(1, [row_b], cols_b, page=1)

    # Page 2: zone A matches F003
    cols_c = [_col("c0", "COL3", 600, 640)]
    row_c = _row(0, [_tok("Field Gamma", 100, 100)], [_cell(cols_c[0], "99.0")],
                 unit_tokens=[_tok("kW", 400, 100)])
    zone_c2 = _zone_result(0, [row_c], cols_c, page=2)

    frs_p1 = build_field_records_canonical([zone_a1, zone_b1], schema, page=1)
    frs_p2 = build_field_records_canonical([zone_c2], schema, page=2)

    merged = aggregate_field_records_across_pages([frs_p1, frs_p2], schema)
    fids = [fr.field_id for fr in merged]
    assert len(fids) == 3, f"expected 3 fields, got {len(fids)}"
    assert len(set(fids)) == 3, f"duplicate field IDs: {[f for f in fids if fids.count(f) > 1]}"
    assert set(fids) == {"XTEST1", "XTEST2", "XTEST3"}
    print("  PASS: test_exactly_one_record_per_schema_field")


# ─── Test 10: zero-candidate field must not blow up ───────
# Regression: aggregate_field_records_across_pages used to NameError on
# FieldRecord/VC when a schema field had no candidate on any page.

def test_zero_candidate_schema_field_emits_absent():
    """A schema field with no page candidate must produce a single
    SOURCE_ABSENT record, not raise NameError."""
    cols = [_col("c0", "COL1", 600, 640)]
    row1 = _row(0, [_tok("Field Alpha", 100, 100)], [_cell(cols[0], "10.0")],
                unit_tokens=[_tok("bar", 400, 100)])
    row2 = _row(0, [_tok("Field Beta", 100, 100)], [_cell(cols[0], "22.0")],
                unit_tokens=[_tok("°C", 400, 100)])

    frs_p1 = build_field_records_canonical([_zone_result(0, [row1], cols, page=1)],
                                           SCHEMA_AB, page=1)
    frs_p2 = build_field_records_canonical([_zone_result(0, [row2], cols, page=2)],
                                           SCHEMA_AB, page=2)

    extra_field = {"id": "XTEST9", "name": "Ghost Field",
                   "aliases": ["Ghost Field"], "unit": "m"}
    merged = aggregate_field_records_across_pages(
        [frs_p1, frs_p2], SCHEMA_AB + [extra_field])

    by_id = {fr.field_id: fr for fr in merged}
    assert len(merged) == len(SCHEMA_AB) + 1, f"got {len(merged)} records"
    assert by_id["XTEST1"].value == 10.0
    assert by_id["XTEST2"].value == 22.0
    ghost = by_id["XTEST9"]
    assert ghost.found is False
    assert ghost.cells and "SOURCE_ABSENT" in ghost.cells[0].flags
    print("  PASS: test_zero_candidate_schema_field_emits_absent")


# ─── runner ───────────────────────────────────────────────

def main():
    tests = [
        test_two_pages_one_canonical_set,
        test_field_on_one_page_only,
        test_field_in_multiple_zones,
        test_extracted_beats_blank,
        test_unrelated_zone_columns_not_inherited,
        test_independent_grids_keep_columns,
        test_blank_cells_positional,
        test_text_value_not_forced_numeric,
        test_exactly_one_record_per_schema_field,
        test_zero_candidate_schema_field_emits_absent,
    ]
    ok = fail = 0
    for f in tests:
        try:
            f()
            ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {f.__name__}: {e}")
    print(f"\nResults: {ok} passed, {fail} failed / {len(tests)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
