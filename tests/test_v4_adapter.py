"""Adapter unit tests: V4 ReconstructionResult -> semantic records."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from meridian.table_reconstruction import (
    Cell,
    Column,
    PhysicalRow,
    ReconstructionResult,
    ReconstructionStatus,
    Region,
    RegionType,
    RowModel,
    StatusIndicator,
    Token,
)
from meridian.v4_adapter import (
    OCR_AMBIGUOUS,
    SOURCE_ABSENT,
    SOURCE_BLANK,
    _is_datetime_shape,
    adapter_to_records,
    build_field_records,
    make_record,
    match_field,
    parse_float,
    score_label,
)


def _tok(text, left, top, conf=95.0, width=40, height=16):
    return Token(text=text, left=left, top=top, width=width, height=height,
                 conf=conf, page=13, region="cylinder_pressure")


def _col(col_id, label, x_min, x_max, path=None):
    return Column(col_id=col_id, label=label, path=path or [label], x_min=x_min, x_max=x_max)


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


def _cell(col, text, blank=False, left=600, conf=95.0, status=None):
    tokens = [] if blank else [_tok(text, left, 100, conf=conf)]
    return Cell(row_idx=0, column=col, tokens=tokens, is_blank=blank,
                status=status or [])


def _result(rows, columns, page=13, region="cylinder_pressure"):
    physical = [PhysicalRow(row_idx=i, y_min=100, y_max=120, row_type="data",
                            model=r) for i, r in enumerate(rows)]
    return ReconstructionResult(
        tokens=[], bands=[], physical_rows=physical, header_bands=[],
        header_hierarchy=[], leaf_columns=columns,
        status=ReconstructionStatus(), debug={},
    )


# CYLINDER-like schema (region cylinder_pressure, page 13)
SCHEMA = [
    {"id": "C001", "page": 13, "region": "cylinder_pressure",
     "name": "Firing press. pmax", "aliases": ["Firing press. pmax", "Firing press pmax"], "unit": "barG"},
    {"id": "C002", "page": 13, "region": "cylinder_pressure",
     "name": "pmax deviation", "aliases": ["pmax deviation"], "unit": "bar"},
    {"id": "C011", "page": 13, "region": "cylinder_pressure",
     "name": "Power indicated", "aliases": ["Power indicated"], "unit": "kW"},
    {"id": "G001", "page": 13, "region": "general",
     "name": "Date and time of recording", "aliases": ["Date and time of recording"], "unit": "", "allow_text_values": True},
]


def _assert(c, m):
    if not c:
        raise AssertionError(m)


def _assert_eq(a, b, label=""):
    if a != b:
        raise AssertionError(f"{label}: expected {b!r}, got {a!r}")


# 1. single-value standalone field
def test_single_value_field():
    cols = [_col("c0", "REF.", 600, 640), _col("c1", "CYL1", 650, 690)]
    cells = [_cell(cols[0], "146.9", left=605), _cell(cols[1], "146.8", left=660)]
    row = _row(0, [_tok("Firing", 100, 100), _tok("press.", 160, 100), _tok("pmax", 220, 100)],
               cells, unit_tokens=[_tok("barG", 420, 100)])
    r = _result([row], cols)
    recs = adapter_to_records(r, SCHEMA, 13, "cylinder_pressure")
    fr = [x for x in recs if x["field_id"] == "C001"]
    _assert_eq(len(fr), 2, "two columns for C001")
    ref = [x for x in fr if x["column"] == "REF"][0]
    _assert_eq(ref["value"], 146.9, "REF value")
    _assert_eq(ref["unit"], "barG", "unit from schema")
    _assert_eq(ref["field_name"], "Firing press. pmax")
    print("  PASS: test_single_value_field")


# 2. multi-column field (REF / CALC / AVG / CYL1..)
def test_multi_column_field():
    cols = [_col("c0", "REF.", 600, 630), _col("c1", "CYL1", 640, 670)]
    cells = [_cell(cols[0], "146.9"), _cell(cols[1], "146.8")]
    row = _row(0, [_tok("Firing press. pmax", 100, 100)], cells,
               unit_tokens=[_tok("barG", 400, 100)])
    recs = adapter_to_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    c001 = [x for x in recs if x["field_id"] == "C001"]
    _assert_eq({x["column"] for x in c001}, {"REF", "CYL1"}, "columns preserved")
    byc = {x["column"]: x["value"] for x in c001}
    _assert_eq(byc["REF"], 146.9)
    _assert_eq(byc["CYL1"], 146.8)
    print("  PASS: test_multi_column_field")


# 3. hierarchical header path provenance
def test_header_path_provenance():
    cols = [_col("c0", "CYL1", 600, 640, path=["CYLINDER PRESSURE", "CYL1"])]
    cells = [_cell(cols[0], "146.8")]
    row = _row(0, [_tok("Firing press. pmax", 100, 100)], cells,
               unit_tokens=[_tok("barG", 400, 100)])
    recs = adapter_to_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    c001 = [x for x in recs if x["field_id"] == "C001"][0]
    _assert("provenance" in c001, "provenance key present")
    _assert_eq(c001["provenance"]["header_path"], ["CYLINDER PRESSURE", "CYL1"],
               "parent header path preserved")
    print("  PASS: test_header_path_provenance")


# 4. blank cell preserved in place
def test_blank_cell():
    cols = [_col("c0", "CYL1", 600, 630), _col("c1", "CYL2", 640, 670)]
    cells = [_cell(cols[0], "146.9"), _cell(cols[1], None, blank=True)]
    row = _row(0, [_tok("Firing press. pmax", 100, 100)], cells,
               unit_tokens=[_tok("barG", 400, 100)])
    recs = adapter_to_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    c001 = [x for x in recs if x["field_id"] == "C001"]
    _assert_eq({x["column"] for x in c001}, {"CYL1", "CYL2"}, "both columns present")
    blank = [x for x in c001 if x["column"] == "CYL2"][0]
    _assert_eq(blank["flags"], [SOURCE_BLANK], "blank flag, no fabricated value")
    _assert_eq(blank["value"], None, "no value fabricated")
    print("  PASS: test_blank_cell")


# 5. status cell: status stays metadata on same cell
def test_status_cell():
    cols = [_col("c0", "CYL2", 600, 640)]
    status = [StatusIndicator(kind="symbol", text="@", bbox=(610, 100, 616, 106))]
    cells = [_cell(cols[0], "@ 0.0/0", conf=95.0, status=status)]
    row = _row(0, [_tok("pmax deviation", 100, 100)], cells,
               unit_tokens=[_tok("bar", 400, 100)])
    recs = adapter_to_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    c002 = [x for x in recs if x["field_id"] == "C002"]
    _assert_eq(len(c002), 1, "no extra value/column for status")
    r0 = c002[0]
    _assert_eq(r0["value"], 0.0, "value parsed, status excluded")
    _assert("status_indicator" in r0["flags"], "status flag on same cell")
    _assert_eq(r0["provenance"]["status"], ["@"], "status preserved as metadata")
    print("  PASS: test_status_cell")


# 6. OCR word merger
def test_ocr_word_merger():
    _assert(score_label("Firing press pmax", "Firing press. pmax") > 0.9,
            "merged punctuation matches")
    _assert(score_label("press.", "press") == 1.0, "trailing period normalized")
    print("  PASS: test_ocr_word_merger")


# 7. ambiguous field label
def test_ambiguous_field_label():
    amb_schema = [
        {"id": "A1", "page": 13, "region": "cylinder_pressure", "name": "Temp dev X",
         "aliases": ["Temp dev X"], "unit": "°C"},
        {"id": "A2", "page": 13, "region": "cylinder_pressure", "name": "Temp dev Y",
         "aliases": ["Temp dev Y"], "unit": "°C"},
    ]
    # A row "Temp dev" matches both equally -> ambiguous, not guessed.
    field, flag = match_field("Temp dev", "", [], amb_schema)
    _assert_eq(flag, OCR_AMBIGUOUS, "ambiguous reported")
    print("  PASS: test_ambiguous_field_label")


# 8. wrong-column prevention: cell stays in its V4 column
def test_wrong_column_prevention():
    cols = [_col("c0", "REF.", 600, 630), _col("c1", "CYL1", 640, 670),
            _col("c2", "CYL2", 680, 710)]
    cells = [_cell(cols[0], "146.9"), _cell(cols[1], "146.8"), _cell(cols[2], "146.7")]
    row = _row(0, [_tok("Firing press. pmax", 100, 100)], cells,
               unit_tokens=[_tok("barG", 400, 100)])
    recs = adapter_to_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    c001 = [x for x in recs if x["field_id"] == "C001"]
    byc = {x["column"]: x["value"] for x in c001}
    _assert_eq(byc["CYL1"], 146.8, "CYL1 value not shifted")
    _assert_eq(byc["CYL2"], 146.7, "CYL2 value not shifted")
    _assert_eq(byc["REF"], 146.9)
    print("  PASS: test_wrong_column_prevention")


# 9. text value
def test_text_value():
    cols = [_col("c0", "VALUE", 600, 900)]
    cells = [_cell(cols[0], "4/15/2025 21:40")]
    row = _row(0, [_tok("Date and time of recording", 100, 100)], cells)
    r = _result([row], cols, page=13, region="general")
    recs = adapter_to_records(r, SCHEMA, 13, "general")
    g001 = [x for x in recs if x["field_id"] == "G001"]
    _assert_eq(len(g001), 1)
    _assert_eq(g001[0]["value"], None, "text value -> raw kept, value None")
    _assert_eq(g001[0]["raw_text"], "4/15/2025 21:40")
    _assert_eq(g001[0]["column"], "VALUE")
    print("  PASS: test_text_value")


# 10. datetime as a standalone (text) value
def test_datetime():
    _assert(parse_float("21:40") is None, "time not a scalar float")
    # A text field (datetime) must keep raw text and never a fabricated float.
    cols = [_col("c0", "VALUE", 600, 900)]
    cells = [_cell(cols[0], "4/15/2025 21:40")]
    row = _row(0, [_tok("Date and time of recording", 100, 100)], cells)
    recs = adapter_to_records(_result([row], cols, region="general"), SCHEMA, 13, "general")
    g001 = [x for x in recs if x["field_id"] == "G001"][0]
    _assert_eq(g001["value"], None, "datetime -> no fabricated float")
    _assert_eq(g001["raw_text"], "4/15/2025 21:40")
    print("  PASS: test_datetime")


# 11. negative / zero / decimal values
def test_number_variants():
    _assert_eq(parse_float("-0.02"), -0.02, "negative")
    _assert_eq(parse_float("0.0"), 0.0, "zero")
    _assert_eq(parse_float("146.8"), 146.8, "decimal")
    _assert_eq(parse_float("146.8%"), 146.8, "percent tail for numeric")
    print("  PASS: test_number_variants")


# 12. source-absent field
def test_source_absent():
    # No rows match C002 at all -> SOURCE_ABSENT, no fabricated value.
    cols = [_col("c0", "CYL1", 600, 640)]
    cells = [_cell(cols[0], "146.8")]
    row = _row(0, [_tok("Firing press. pmax", 100, 100)], cells,
               unit_tokens=[_tok("barG", 400, 100)])
    recs = adapter_to_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    c002 = [x for x in recs if x["field_id"] == "C002"]
    _assert_eq(len(c002), 1, "absent field reported once")
    _assert_eq(c002[0]["flags"], [SOURCE_ABSENT])
    _assert_eq(c002[0]["value"], None)
    print("  PASS: test_source_absent")


# 13. duplicate labels in different sections
def test_duplicate_labels_diff_sections():
    # Same label text, different region -> region filters prevent cross-match.
    s2 = [
        {"id": "X1", "page": 13, "region": "exhaust_gas", "name": "temp. deviation",
         "aliases": ["temp. deviation"], "unit": "°C"},
        {"id": "Y1", "page": 13, "region": "cylinder_pressure", "name": "temp. deviation",
         "aliases": ["temp. deviation"], "unit": "°C"},
    ]
    # Only cylinder_pressure fields are candidates.
    cand = [f for f in s2 if f["region"] == "cylinder_pressure"]
    _assert_eq(len(cand), 1, "region scoping")
    field, flag = match_field("temp. deviation", "", [], cand)
    _assert_eq(field["id"], "Y1", "correct section field chosen")
    print("  PASS: test_duplicate_labels_diff_sections")


def test_build_field_records_for_validator():
    cols = [_col("c0", "REF.", 600, 630), _col("c1", "CYL1", 640, 670)]
    cells = [_cell(cols[0], "146.9"), _cell(cols[1], "146.8")]
    row = _row(0, [_tok("Firing press. pmax", 100, 100)], cells,
               unit_tokens=[_tok("barG", 400, 100)])
    from meridian.validate import validate
    frs = build_field_records(_result([row], cols), SCHEMA, 13, "cylinder_pressure")
    _assert(frs, "field records produced")
    issues = validate(frs)
    _assert(isinstance(issues, list), "validator runs without error")
    print("  PASS: test_build_field_records_for_validator")


def test_packed_cell_multi_value_no_truncation():
    # C013 pmax-pcomp / pscav: one AVG cell holds TWO independent numbers
    # (30.0 and 46.8). Both must be emitted, in the SAME column, with the
    # second flagged SECONDARY_VALUE - never silently truncated.
    cols = [_col("c0", "REF.", 600, 700), _col("c1", "AVG.", 700, 900)]
    s = [{"id": "C013", "page": 13, "region": "cylinder_pressure",
          "name": "pmax-pcomp / pscav",
          "aliases": ["pmax-pcomp / pscav", "pmax-pcomp"], "unit": "bar"}]
    avgtok1 = _tok("30.0|", 705, 100, conf=91.0)
    avgtok2 = _tok("46.8", 760, 100, conf=96.0)
    reftok = _tok("46.1", 605, 100)
    cells = [
        Cell(row_idx=0, column=cols[0], tokens=[reftok], is_blank=False, status=[]),
        Cell(row_idx=0, column=cols[1], tokens=[avgtok1, avgtok2], is_blank=False, status=[]),
    ]
    row = _row(0, [_tok("pmax-pcomp", 100, 100)], cells)
    flat = adapter_to_records(_result([row], cols), s, 13, "cylinder_pressure")
    avg = [r for r in flat if r["column"] == "AVG"]
    _assert_eq(len(avg), 2, "both AVG values emitted")
    vals = sorted(x["value"] for x in avg)
    _assert_eq(vals, [30.0, 46.8], "both values present, none lost")
    sec = [x for x in avg if x["value"] == 46.8][0]
    _assert("SECONDARY_VALUE" in sec["flags"], "second value flagged, not hidden")
    _assert_eq(sec["column"], "AVG", "stays in same column")
    ref = [r for r in flat if r["column"] == "REF"][0]
    _assert_eq(ref["value"], 46.1, "unrelated columns unaffected")
    print("  PASS: test_packed_cell_multi_value_no_truncation")


# ── Geometric schema column mapping tests ────────────────────────────

def test_geometric_exact_match():
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        v4_adapter.get_explicit_columns = lambda f: ["MCR", "ENGINE"]
        v4 = [_col("c0", "MCR", 100, 200), _col("c1", "ENGINE", 300, 400)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result == {"MCR": "MCR", "ENGINE": "ENGINE"}, result
        print("  PASS: test_geometric_exact_match")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_geometric_extra_v4_column():
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        v4_adapter.get_explicit_columns = lambda f: ["A", "B", "C", "D"]
        v4 = [_col("c0", "A", 100, 150), _col("c1", "B", 200, 250),
              _col("c2", "EXTRA", 300, 350), _col("c3", "C", 400, 450),
              _col("c4", "D", 500, 550)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result == {"A": "A", "B": "B", "C": "C", "D": "D"}, result
        print("  PASS: test_geometric_extra_v4_column")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_geometric_ocr_imperfect_by_position():
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        v4_adapter.get_explicit_columns = lambda f: [
            "REF", "CALC", "AVG", "TC1", "TC2", "TC3"]
        v4 = [_col("c0", "REF", 100, 150), _col("c1", "CALC", 200, 250),
              _col("c2", "AVG", 300, 350), _col("c3", "T1", 400, 450),
              _col("c4", "TC2", 500, 550), _col("c5", "TC3", 600, 650)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result.get("TC1") == "T1", f"TC1 should map to T1, got {result.get('TC1')}"
        assert result.get("REF") == "REF"
        assert result.get("AVG") == "AVG"
        print("  PASS: test_geometric_ocr_imperfect_by_position")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_geometric_two_unmatched_between_anchors():
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        v4_adapter.get_explicit_columns = lambda f: ["A", "B", "C", "D", "E"]
        v4 = [_col("c0", "A", 100, 150), _col("c1", "X", 200, 250),
              _col("c2", "Y", 300, 350), _col("c3", "D", 500, 550),
              _col("c4", "E", 600, 650)]
        result = v4_adapter._build_schema_column_map({}, v4)
        # B and C sit between anchors (A at idx 0, D/E at idx 3/4).
        # Geometric interpolation fills them — no ordinal fallback needed.
        assert result.get("A") == "A"
        assert result.get("B") == "X", f"B should geometrically map to X, got {result.get('B')}"
        assert result.get("C") == "Y", f"C should geometrically map to Y, got {result.get('C')}"
        assert result.get("D") == "D"
        assert result.get("E") == "E"
        print("  PASS: test_geometric_two_unmatched_between_anchors")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_geometric_brg_ocr_corrupted():
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        schema = ["AVG", "BRG1", "BRG2", "BRG3", "BRG4", "BRG5", "BRG6",
                  "BRG7", "BRG8", "BRG9", "BRG10", "BRG11", "BRG12", "BRG13"]
        v4_adapter.get_explicit_columns = lambda f: schema
        v4 = [_col("c0", "AVG", 862, 863), _col("c1", "ENG", 922, 923),
              _col("c2", "BRG1", 988, 989), _col("c3", "BRG2", 1066, 1067),
              _col("c4", "BRG3", 1145, 1146), _col("c5", "BRG4", 1224, 1225),
              _col("c6", "BRGS", 1303, 1304), _col("c7", "BRG6", 1382, 1383),
              _col("c8", "BRG7", 1460, 1461), _col("c9", "BRG8", 1539, 1540),
              _col("c10", "BRGY", 1616, 1617), _col("c11", "BRG10", 1696, 1697),
              _col("c12", "BRG11", 1778, 1779), _col("c13", "BRG12", 1859, 1860),
              _col("c14", "BRG13", 1937, 1938)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result.get("BRG5") == "BRGS", f"BRG5->BRGS, got {result.get('BRG5')}"
        assert result.get("BRG9") == "BRGY", f"BRG9->BRGY, got {result.get('BRG9')}"
        assert "ENG" not in result.values(), "ENG should not be mapped"
        print("  PASS: test_geometric_brg_ocr_corrupted")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_geometric_missing_schema_column():
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        v4_adapter.get_explicit_columns = lambda f: ["A", "B", "C", "D"]
        v4 = [_col("c0", "A", 100, 150), _col("c1", "C", 300, 350),
              _col("c2", "D", 400, 450)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result.get("A") == "A"
        assert "B" not in result, f"B should be unmapped, got {result.get('B')}"
        assert result.get("C") == "C"
        assert result.get("D") == "D"
        print("  PASS: test_geometric_missing_schema_column")
    finally:
        v4_adapter.get_explicit_columns = orig


# ── Geometry-driven column-mapping: resolved vs unresolved ──────────

def test_clean_schema_match_maps_correctly():
    """A V4 leaf set whose labels exactly match the schema produces a complete,
    correct mapping — no regression from removing ordinal fallback."""
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        schema = ["CYL1", "CYL2", "CYL3", "CYL4", "CYL5"]
        v4_adapter.get_explicit_columns = lambda f: schema
        v4 = [_col("c0", "CYL1", 100, 150), _col("c1", "CYL2", 200, 250),
              _col("c2", "CYL3", 300, 350), _col("c3", "CYL4", 400, 450),
              _col("c4", "CYL5", 500, 550)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result is not None, "complete match should return a mapping"
        for sc in schema:
            assert result.get(sc) == sc, f"{sc} should self-map, got {result.get(sc)}"
        print("  PASS: test_clean_schema_match_maps_correctly")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_mismatched_leaves_return_none_unresolved():
    """When V4 labels cannot be matched to schema columns by anchor or
    geometry (no anchors at all), the mapping returns None — signalling
    the caller to leave fields UNRESOLVED rather than guessing by ordinal."""
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        schema = ["CYL1", "CYL2", "CYL3", "CYL4", "CYL5"]
        v4_adapter.get_explicit_columns = lambda f: schema
        # All V4 labels are gibberish — no anchor matches.
        v4 = [_col("c0", "ZZZ", 100, 150), _col("c1", "QQQ", 200, 250),
              _col("c2", "WWW", 300, 350), _col("c3", "EEE", 400, 450),
              _col("c4", "RRR", 500, 550)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result is None, f"no anchors → None (unresolved), got {result}"
        print("  PASS: test_mismatched_leaves_return_none_unresolved")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_apply_schema_unresolved_columns_get_absent_flag():
    """When _build_schema_column_map returns None (no anchors), the
    _apply_schema_to_records path must flag unmapped columns — never
    silently guess by ordinal position."""
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        v4_adapter.get_explicit_columns = lambda f: ["A", "B", "C"]
        v4_mismatch = [_col("c0", "X", 100, 150), _col("c1", "Y", 200, 250),
                       _col("c2", "Z", 300, 350)]
        field = {"id": "F1", "name": "test field", "unit": "bar"}
        recs = [{"field_id": "F1", "field_name": "test field", "page": 1,
                 "region": "r", "column": "X", "raw_text": "10", "value": 10.0,
                 "unit": "bar", "confidence": 95.0, "bbox": [],
                 "source_row": "r1", "flags": []},
                {"field_id": "F1", "field_name": "test field", "page": 1,
                 "region": "r", "column": "Y", "raw_text": "20", "value": 20.0,
                 "unit": "bar", "confidence": 95.0, "bbox": [],
                 "source_row": "r1", "flags": []},
                {"field_id": "F1", "field_name": "test field", "page": 1,
                 "region": "r", "column": "Z", "raw_text": "30", "value": 30.0,
                 "unit": "bar", "confidence": 95.0, "bbox": [],
                 "source_row": "r1", "flags": []}]
        remapped = v4_adapter._apply_schema_to_records(field, recs, v4_mismatch)
        # No anchor match (X≠A, Y≠B, Z≠C), mapping returns None.
        # _apply_schema_to_records falls back to positional + column_unresolved.
        assert len(remapped) == 3, f"3 records, got {len(remapped)}"
        cols = [r["column"] for r in remapped]
        assert cols == ["A", "B", "C"], f"positional remap, got {cols}"
        for r in remapped:
            assert "column_unresolved" in r["flags"], \
                f"{r['column']} must be flagged unresolved, flags={r['flags']}"
        print("  PASS: test_apply_schema_unresolved_columns_get_absent_flag")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_partial_anchor_completes_by_geometry_not_ordinal():
    """When some columns anchor by label and the remaining columns sit
    between anchors, geometric interpolation fills them — not ordinal."""
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        schema = ["A", "B", "C", "D", "E"]
        v4_adapter.get_explicit_columns = lambda f: schema
        # A, C, E anchor by label; B, D are between and must be found
        # by geometric interpolation (not ordinal position).
        v4 = [_col("c0", "A", 100, 150), _col("c1", "X", 200, 250),
              _col("c2", "C", 300, 350), _col("c3", "Y", 400, 450),
              _col("c4", "E", 500, 550)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result is not None, "partial anchors + geometry should complete"
        assert result["A"] == "A"
        assert result["C"] == "C"
        assert result["E"] == "E"
        assert result["B"] == "X", f"B should geometrically match X, got {result.get('B')}"
        assert result["D"] == "Y", f"D should geometrically match Y, got {result.get('D')}"
        print("  PASS: test_partial_anchor_completes_by_geometry_not_ordinal")
    finally:
        v4_adapter.get_explicit_columns = orig


def test_no_anchors_returns_none_not_ordinal():
    """Zero label matches → mapping is None (all unresolved), never ordinal."""
    from meridian import v4_adapter
    orig = v4_adapter.get_explicit_columns
    try:
        schema = ["P001", "P002", "P003"]
        v4_adapter.get_explicit_columns = lambda f: schema
        v4 = [_col("c0", "GARBAGE1", 100, 150), _col("c1", "GARBAGE2", 200, 250),
              _col("c2", "GARBAGE3", 300, 350)]
        result = v4_adapter._build_schema_column_map({}, v4)
        assert result is None, f"no anchors → None, got {result}"
        print("  PASS: test_no_anchors_returns_none_not_ordinal")
    finally:
        v4_adapter.get_explicit_columns = orig


def main():
    t = [
        test_single_value_field, test_multi_column_field, test_header_path_provenance,
        test_blank_cell, test_status_cell, test_ocr_word_merger, test_ambiguous_field_label,
        test_wrong_column_prevention, test_text_value, test_datetime, test_number_variants,
        test_source_absent, test_duplicate_labels_diff_sections,
        test_build_field_records_for_validator,
        test_packed_cell_multi_value_no_truncation,
        test_geometric_exact_match,
        test_geometric_extra_v4_column,
        test_geometric_ocr_imperfect_by_position,
        test_geometric_two_unmatched_between_anchors,
        test_geometric_brg_ocr_corrupted,
        test_geometric_missing_schema_column,
    ]
    ok = fail = 0
    for f in t:
        try:
            f(); ok += 1
        except Exception as e:
            fail += 1
            print(f"  FAIL {f.__name__}: {e}")
    print(f"\nResults: {ok} passed, {fail} failed / {len(t)}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


# ─────────────────────────────────────────────────────────────────
# Phase 3: generic datetime-shape recognition tests
# ─────────────────────────────────────────────────────────────────

def test_datetime_slash_date():
    """Slash date '10/25/2025' must not be coerced to a leading number."""
    assert _is_datetime_shape("10/25/2025")
    assert parse_float("10/25/2025") is None

def test_datetime_iso_date():
    """ISO date '2025-09-26' must not be coerced to a leading number."""
    assert _is_datetime_shape("2025-09-26")
    assert parse_float("2025-09-26") is None

def test_datetime_slash_with_time():
    """Date + time '10/25/2025 15:30' must return None."""
    assert _is_datetime_shape("10/25/2025 15:30")
    assert parse_float("10/25/2025 15:30") is None

def test_datetime_slash_with_suffix():
    """Date + bracket artifact '10/25/2025 [15]' must return None."""
    assert _is_datetime_shape("10/25/2025 [15]")
    assert parse_float("10/25/2025 [15]") is None

def test_datetime_integer_stays_numeric():
    """Plain integer remains numeric."""
    assert not _is_datetime_shape("123")
    assert parse_float("123") == 123.0

def test_datetime_decimal_stays_numeric():
    """Plain decimal remains numeric."""
    assert not _is_datetime_shape("123.45")
    assert parse_float("123.45") == 123.45

def test_datetime_percentage_stays_numeric():
    """Percentage '32.0%' remains numeric."""
    assert not _is_datetime_shape("32.0%")
    assert parse_float("32.0%") == 32.0

def test_datetime_malformed_no_crash():
    """Malformed date-like text must not crash."""
    assert not _is_datetime_shape("")
    assert parse_float("") is None
    assert not _is_datetime_shape("10/25")
    assert parse_float("10/25") == 10.0
    assert not _is_datetime_shape("2025-09")
    assert parse_float("2025-09") == 2025.0

def test_datetime_text_value_stays_text():
    """Plain text value returns None from parse_float."""
    assert not _is_datetime_shape("ENGINE running")
    assert parse_float("ENGINE running") is None

def test_datetime_no_field_id_logic():
    """All detection is shape-based; same input returns same result regardless
    of any field context (no field-ID-specific logic)."""
    v1 = parse_float("10/25/2025")
    v2 = parse_float("10/25/2025")
    assert v1 is None and v2 is None
    assert not _is_datetime_shape("69556")
    assert parse_float("69556") == 69556.0


# ─────────────────────────────────────────────────────────────────
# Phase 3 completion: field-level evidence propagation
# ─────────────────────────────────────────────────────────────────

def _make_result_with_rows(rows):
    """Build a minimal ReconstructionResult wrapping pre-built PhysicalRows."""
    return ReconstructionResult(
        tokens=[], bands=[], physical_rows=rows, header_bands=[],
        header_hierarchy=[], leaf_columns=[
            Column(col_id="VALUE", label="VALUE", path=["VALUE"],
                   x_min=0.0, x_max=2000.0),
        ],
        status=ReconstructionStatus(ok=True, warnings=[]),
    )

def _pr(label_text, value_text, label_left=100, value_left=400, y=50,
        value_conf=95.0, value_flags=None):
    """Create a PhysicalRow with a single VALUE cell."""
    from meridian.table_reconstruction import Cell as TCell
    lt = Token(text=label_text, left=label_left, top=y, width=60, height=16,
               conf=95.0, page=1, region="reading")
    vt = Token(text=value_text, left=value_left, top=y, width=60, height=16,
               conf=value_conf, page=1, region="reading")
    lr = Region(RegionType.LABEL, tokens=[lt])
    value_col = Column(col_id="VALUE", label="VALUE", path=["VALUE"],
                       x_min=350.0, x_max=600.0)
    cell = TCell(row_idx=0, column=value_col, tokens=[vt],
                 is_blank=not value_text.strip())
    model = RowModel(row_idx=0, bbox=(label_left, y, value_left + 60, y + 16),
                     label_region=lr, cells=[cell])
    pr = PhysicalRow(row_idx=0, y_min=float(y), y_max=float(y + 16),
                     row_type="data", model=model)
    return pr

def _schema(fid, name, unit=""):
    return {"id": fid, "name": name, "unit": unit, "region": "general",
            "aliases": [name]}

def test_propagation_date_cell():
    """Date-like VALUE cell propagates raw_text to field record."""
    from meridian.v4_adapter import build_field_records
    pr = _pr("Date and time", "10/25/2025 [15]")
    result = _make_result_with_rows([pr])
    frs = build_field_records(result, [_schema("F1", "Date and time")], 1, "general")
    assert len(frs) == 1
    fr = frs[0]
    assert fr.found
    assert fr.raw_text == "10/25/2025 [15]"
    assert fr.value is None

def test_propagation_iso_date():
    """ISO date VALUE cell propagates raw_text."""
    from meridian.v4_adapter import build_field_records
    pr = _pr("Report date", "2025-09-26")
    result = _make_result_with_rows([pr])
    frs = build_field_records(result, [_schema("F1", "Report date")], 1, "general")
    fr = frs[0]
    assert fr.found
    assert fr.raw_text == "2025-09-26"
    assert fr.value is None

def test_propagation_text_value():
    """Ordinary text VALUE propagates raw_text."""
    from meridian.v4_adapter import build_field_records
    pr = _pr("ENGINE state", "running")
    result = _make_result_with_rows([pr])
    frs = build_field_records(result, [_schema("F1", "ENGINE state")], 1, "general")
    fr = frs[0]
    assert fr.found
    assert fr.raw_text == "running"
    assert fr.value is None

def test_propagation_numeric_value():
    """Numeric VALUE remains numeric at field level."""
    from meridian.v4_adapter import build_field_records
    pr = _pr("Ambient air temp.", "29.0")
    result = _make_result_with_rows([pr])
    frs = build_field_records(result, [_schema("F1", "Ambient air temp.", "°C")], 1, "general")
    fr = frs[0]
    assert fr.found
    assert fr.raw_text == "29.0"
    assert fr.value == 29.0

def test_propagation_blank_value():
    """Blank VALUE cell yields raw_text=None and value=None."""
    from meridian.v4_adapter import build_field_records
    pr = _pr("Sea water temp.", "")
    result = _make_result_with_rows([pr])
    frs = build_field_records(result, [_schema("F1", "Sea water temp.", "°C")], 1, "general")
    fr = frs[0]
    assert fr.found
    assert fr.raw_text is None
    assert fr.value is None

def test_propagation_multi_column_no_collapse():
    """Field with multiple VALUE columns does not collapse into a scalar."""
    from meridian.table_reconstruction import Cell as TCell
    from meridian.v4_adapter import build_field_records
    lt = Token(text="CYL temps", left=100, top=50, width=60, height=16,
               conf=95.0, page=1, region="reading")
    v1t = Token(text="45.0", left=400, top=50, width=40, height=16,
                conf=95.0, page=1, region="reading")
    v2t = Token(text="46.0", left=500, top=50, width=40, height=16,
                conf=95.0, page=1, region="reading")
    lr = Region(RegionType.LABEL, tokens=[lt])
    c1 = Column(col_id="CYL1", label="CYL1", path=["CYL1"], x_min=380, x_max=460)
    c2 = Column(col_id="CYL2", label="CYL2", path=["CYL2"], x_min=470, x_max=560)
    cell1 = TCell(row_idx=0, column=c1, tokens=[v1t], is_blank=False)
    cell2 = TCell(row_idx=0, column=c2, tokens=[v2t], is_blank=False)
    model = RowModel(row_idx=0, bbox=(100, 50, 560, 66), label_region=lr,
                     cells=[cell1, cell2])
    pr = PhysicalRow(row_idx=0, y_min=50.0, y_max=66.0, row_type="data",
                     model=model)
    result = ReconstructionResult(
        tokens=[], bands=[], physical_rows=[pr], header_bands=[],
        header_hierarchy=[], leaf_columns=[c1, c2],
        status=ReconstructionStatus(ok=True, warnings=[]),
    )
    schema = _schema("F1", "CYL temps", "°C")
    schema["columns"] = ["CYL1", "CYL2"]
    frs = build_field_records(result, [schema], 1, "general")
    fr = frs[0]
    assert fr.found
    # Multi-column field: raw_text/value should be None (not collapsed)
    assert fr.raw_text is None
    assert fr.value is None
    assert len(fr.cells) >= 2

def test_propagation_no_field_id_logic():
    """Propagation is purely structural; no field-ID-specific branching."""
    from meridian.v4_adapter import build_field_records
    pr1 = _pr("Field A", "10/25/2025")
    pr2 = _pr("Field B", "10/25/2025")
    result = _make_result_with_rows([pr1, pr2])
    frs = build_field_records(result, [
        _schema("X99", "Field A"),
        _schema("Y42", "Field B"),
    ], 1, "general")
    for fr in frs:
        assert fr.raw_text == "10/25/2025"
        assert fr.value is None


# ───────────────────────────────────────────────────────────
# Generic compound-word label match (G005 'Seawater' == 'Sea water')
#
# OCR merges two printed words into a single token with no separator
# (e.g. 'Seawater' for alias words 'Sea water' / 'Sea water temp.'). The
# matcher must recognise a row token equal to the concatenation of
# consecutive alias tokens, generically (pure string concat, no vocabulary).
# ───────────────────────────────────────────────────────────

def test_compound_word_score_label():
    _assert_eq(score_label("Seawater temp.", "Sea water temp"), 1.0,
               "merged compound merges alias tokens")
    _assert_eq(score_label("Seawater", "Sea water"), 1.0,
               "two-word alias matched by one merged token")


def test_compound_word_match_field():
    schema = [
        {"id": "G005", "page": 13, "region": "general",
         "name": "Sea water temp.", "aliases": ["Sea water temp"], "unit": "°C"},
    ]
    field, flag = match_field("Seawater temp.", "", [], schema)
    _assert(None in (flag or [None]) and not (flag or ""),
            f"no flag for clean compound match, got {flag!r}")
    _assert(field is not None and field["id"] == "G005",
            "compound 'Seawater temp.' must match G005")


def test_compound_word_non_match_no_collision():
    _assert(score_label("Seawater", "Sea") < 1.0,
            "over-long merged token must not fully match a shorter alias")


# ───────────────────────────────────────────────────────────
# Generic low-OCR-confidence flagging (E003 example)
#
# A numeric value cell's confidence is the MEAN of its token confidences on
# the OCR 0-100 scale. A high-confidence sibling token (e.g. the '%' unit)
# must NOT mask a low-confidence value token (e.g. an OCR-glued '673|®' at
# conf 29): the cell must be flagged low_ocr_confidence so the reading is not
# reported as a confident, definitive number.
# ───────────────────────────────────────────────────────────

def test_low_confidence_value_not_masked_by_unit():
    """A low-conf value token beside a high-conf unit token is still flagged."""
    from meridian.table_reconstruction import Cell, Column
    from meridian.v4_adapter import LOW_CONFIDENCE
    field = {"id": "E003", "page": 13, "region": "electronic_control",
             "name": "Fuel index ECU", "aliases": ["Fuel index ECU"], "unit": "%"}
    col = Column(col_id="c0", label="VALUE", path=["VALUE"], x_min=0.0, x_max=900.0)
    toks = [_tok("%", 486, 100, conf=92.0), _tok("673|®", 592, 100, conf=29.0)]
    cell = Cell(row_idx=0, column=col, tokens=toks, is_blank=False, status=[])
    rec = make_record(field, "VALUE", cell, 13, "electronic_control",
                      "Fuel index ECU", [])
    # The high-confidence '%' unit masks the cell MEAN (60.5 >= 60) but the
    # value token itself '673|®' is conf 29 < 60 -- it must still be flagged.
    _assert(cell.confidence >= 60.0, "unit inflates the cell mean above 60")
    _assert(LOW_CONFIDENCE in rec["flags"],
            f"low-conf value must carry {LOW_CONFIDENCE}, got {rec['flags']}")
    # The value is still read (no value manufactured), just flagged uncertain.
    _assert_eq(rec["value"], 673.0, "value still parsed")


def test_high_confidence_cell_not_flagged():
    """A genuinely high-confidence numeric cell gets no low_ocr_confidence."""
    from meridian.table_reconstruction import Cell, Column
    from meridian.v4_adapter import LOW_CONFIDENCE
    field = {"id": "E003", "page": 13, "region": "electronic_control",
             "name": "Fuel index ECU", "aliases": ["Fuel index ECU"], "unit": "%"}
    col = Column(col_id="c0", label="VALUE", path=["VALUE"], x_min=0.0, x_max=900.0)
    toks = [_tok("%", 486, 100, conf=96.0), _tok("65.5", 592, 100, conf=98.0)]
    cell = Cell(row_idx=0, column=col, tokens=toks, is_blank=False, status=[])
    rec = make_record(field, "VALUE", cell, 13, "electronic_control",
                      "Fuel index ECU", [])
    _assert(cell.confidence >= 60.0, "mean cell confidence must be high")
    _assert(LOW_CONFIDENCE not in rec["flags"],
            f"high-conf cell must NOT be flagged, got {rec['flags']}")


def test_low_confidence_whole_page_path():
    """The whole-page (Template B) adapter path also flags low confidence."""
    from meridian.table_reconstruction import Cell, Column
    from meridian.v4_adapter import LOW_CONFIDENCE
    field = {"id": "E003", "page": 13, "region": "electronic_control",
             "name": "Fuel index ECU", "aliases": ["Fuel index ECU"], "unit": "%"}
    col = Column(col_id="c0", label="VALUE", path=["VALUE"], x_min=0.0, x_max=900.0)
    toks = [_tok("%", 486, 100, conf=92.0), _tok("59.2", 592, 100, conf=26.0)]
    cell = Cell(row_idx=0, column=col, tokens=toks, is_blank=False, status=[])
    rec = make_record(field, "VALUE", cell, 13, "electronic_control",
                      "Fuel index ECU", [])
    _assert(LOW_CONFIDENCE in rec["flags"], "whole-page make_record flags low conf")
