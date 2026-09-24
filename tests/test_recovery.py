"""Tests for the generic geometry fallback recovery pass (Phase 6).

These test the production ``app.recovery.recover_dense_empty_cells`` directly
with synthetic geometry, plus the adapter/flag integration. Every test uses
purely geometric / field-agnostic data -- no filename, month, or field-ID branches.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meridian.recovery import recover_dense_empty_cells
from meridian.table_reconstruction import (
    Cell,
    Column,
    PhysicalRow,
    ReconstructionResult,
    ReconstructionStatus,
    Region,
    RegionType,
    RowModel,
    Token,
)
from meridian.v4_adapter import (
    RECOVERED_BY_GEOMETRY_FALLBACK,
    build_field_records,
)


def _tok(text, left, top, conf=95.0, width=40, height=16):
    return Token(text=text, left=left, top=top, width=width, height=height,
                 conf=conf, page=13, region="cylinder_pressure")


def _col(col_id, label, x_min, x_max, path=None):
    return Column(col_id=col_id, label=label, path=path or [label],
                  x_min=float(x_min), x_max=float(x_max))


def _row(idx, label_tokens, cells, y=100, unit_tokens=None):
    lab = Region(RegionType.LABEL, label_tokens)
    unit = Region(RegionType.UNIT, unit_tokens or [])
    return RowModel(
        row_idx=idx,
        bbox=(50.0, float(y), 900.0, float(y + 20)),
        label_region=lab, unit_region=unit, qualifier_regions=[],
        cells=cells, status_indicators=[],
    )


def _blank_cell(col):
    return Cell(row_idx=0, column=col, tokens=[], is_blank=True)


def _populated_cell(col, text, left):
    return Cell(row_idx=0, column=col, tokens=[_tok(text, left, 100)],
                is_blank=False)


def _result(rows, columns, all_tokens):
    physical = [PhysicalRow(row_idx=i, y_min=100, y_max=120, row_type="data",
                            model=r) for i, r in enumerate(rows)]
    return ReconstructionResult(
        tokens=list(all_tokens), bands=[], physical_rows=physical,
        header_bands=[], header_hierarchy=[], leaf_columns=columns,
        status=ReconstructionStatus(), debug={},
    )


def _assert(c, m):
    if not c:
        raise AssertionError(m)


# 1. Dense CYL grid: a blank CYL column recoverable from an unplaced value token.
def test_dense_cyl_grid_recovery():
    cols = [_col("cyl1", "CYL1", 380, 460),
            _col("cyl2", "CYL2", 470, 550),
            _col("cyl3", "CYL3", 560, 640)]
    cells = [_populated_cell(cols[0], "48.0", 390),
             _populated_cell(cols[1], "49.0", 480),
             _blank_cell(cols[2])]
    row = _row(0, [_tok("CYL", 100, 100), _tok("press", 160, 100)], cells)
    # value token for CYL3 exists in source but was not assigned to the cell
    stray = _tok("50.0", 575, 100)
    result = _result([row], cols, [stray])
    filled = recover_dense_empty_cells(result)
    _assert(filled == 1, "one cell should be recovered")
    c3 = cells[2]
    _assert(not c3.is_blank, "CYL3 should no longer be blank")
    _assert(c3.raw_text == "50.0", "CYL3 raw text should be 50.0")
    _assert(c3.recovered, "CYL3 should be flagged recovered")


# 2. Dense BRG grid: same geometry recovery for a bearing-style column set.
def test_dense_brg_grid_recovery():
    cols = [_col("b1", "BRG1", 380, 460),
            _col("b2", "BRG2", 470, 550),
            _col("b3", "BRG3", 560, 640)]
    cells = [_blank_cell(cols[0]),
             _populated_cell(cols[1], "61.2", 480),
             _populated_cell(cols[2], "61.4", 570)]
    row = _row(0, [_tok("BRG", 100, 100), _tok("temp", 160, 100)], cells)
    stray = _tok("60.9", 390, 100)
    result = _result([row], cols, [stray])
    filled = recover_dense_empty_cells(result)
    _assert(filled == 1, "one bearing cell should be recovered")
    _assert(not cells[0].is_blank and cells[0].raw_text == "60.9",
            "BRG1 should recover 60.9")


# 3. Multi-column numeric row: a numeric field spanning several columns with one blank.
def test_multi_column_numeric_row_recovery():
    cols = [_col("ref", "REF.", 560, 640),
            _col("avg", "AVG.", 810, 900),
            _col("mop", "MOP", 1200, 1290)]
    cells = [_populated_cell(cols[0], "10.0", 570),
             _blank_cell(cols[1]),
             _populated_cell(cols[2], "11.0", 1210)]
    row = _row(0, [_tok("Firing", 100, 100), _tok("press", 160, 100),
                   _tok("pmax", 220, 100)], cells)
    stray = _tok("10.5", 830, 100)
    result = _result([row], cols, [stray])
    filled = recover_dense_empty_cells(result)
    _assert(filled == 1, "AVG cell should recover")
    _assert(cells[1].raw_text == "10.5", "AVG should be 10.5")


# 4. Blank cell with a nearby (within tolerance) value token.
def test_nearby_value_within_tolerance():
    col = _col("c0", "VALUE", 800, 860)  # width 60
    cells = [_blank_cell(col)]
    row = _row(0, [_tok("Field", 100, 100)], cells)
    # token just outside the column boundary but within 35% tolerance
    stray = _tok("7.7", 830, 100)  # inside
    result = _result([row], [col], [stray])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 1, "nearby value within tolerance should recover")


# 5. Multiple candidates near the same blank cell -> stays empty (no guessing).
def test_multiple_candidates_stays_empty():
    col = _col("c0", "VALUE", 800, 860)
    cells = [_blank_cell(col)]
    row = _row(0, [_tok("Field", 100, 100)], cells)
    t1 = _tok("7.7", 810, 100)
    t2 = _tok("8.8", 840, 100)
    result = _result([row], [col], [t1, t2])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 0, "multiple candidates must stay empty")
    _assert(cells[0].is_blank, "cell must remain blank when ambiguous")


# 6. Populated cell protection: never overwrite an existing value, and a token
#    already claimed by a populated sibling is never reassigned.
def test_populated_cell_protection():
    cols = [_col("a", "A", 380, 460), _col("b", "B", 470, 550)]
    cells = [_populated_cell(cols[0], "42.0", 390), _blank_cell(cols[1])]
    row = _row(0, [_tok("F", 100, 100)], cells)
    # '45.0' is physically near column A; it must NOT overwrite A's value, and
    # since it's already claimed by A's populated cell, B must not steal it.
    claimed = _tok("45.0", 410, 100)
    cells[0].tokens = [claimed]
    result = _result([row], cols, [claimed])
    fill = recover_dense_empty_cells(result)
    _assert(cells[0].raw_text == "45.0", "A populated value unchanged")
    _assert(fill == 0 or cells[1].is_blank,
            "claimed token must not be reassigned to B")


# 6b. A populated cell must never be overwritten even when a different value
#     token sits in its window (only blank cells are touched).
def test_populated_cell_never_overwritten():
    col = _col("a", "A", 380, 460)
    cells = [_populated_cell(col, "42.0", 390)]
    row = _row(0, [_tok("F", 100, 100)], cells)
    stray = _tok("99.0", 400, 100)  # near populated cell window
    result = _result([row], [col], [stray])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 0, "populated cell must not be modified")
    _assert(cells[0].raw_text == "42.0", "value unchanged")


# 7. Numeric punctuation (comma thousands / decimal) is a value token.
def test_numeric_punctuation_recovery():
    col = _col("c0", "VALUE", 800, 880)
    cells = [_blank_cell(col)]
    row = _row(0, [_tok("F", 100, 100)], cells)
    stray = _tok("1,234.5", 815, 100)
    result = _result([row], [col], [stray])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 1, "comma/decimal numeric should recover")
    _assert(cells[0].raw_text == "1,234.5", "raw text preserved")


# 8. Unit-after-number: a token like '12.5 C' is still a value candidate.
def test_unit_after_number_recovery():
    col = _col("c0", "VALUE", 800, 900)
    cells = [_blank_cell(col)]
    row = _row(0, [_tok("Lub", 100, 100), _tok("oil", 140, 100)], cells)
    stray = _tok("12.5", 820, 100)
    result = _result([row], [col], [stray])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 1, "numeric value should recover")


# 9. Text value (non-numeric) token is NOT a numeric value candidate -> stays blank.
def test_text_value_not_recovered():
    col = _col("c0", "VALUE", 800, 900)
    cells = [_blank_cell(col)]
    row = _row(0, [_tok("Mode", 100, 100), _tok("Running", 160, 100)], cells)
    stray = _tok("STANDBY", 820, 100)
    result = _result([row], [col], [stray])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 0, "text token must not be recovered as a numeric value")
    _assert(cells[0].is_blank, "cell stays blank for text")


# 10. Ambiguous candidate (two distinct values) -> stays empty.
def test_ambiguous_candidate_stays_empty():
    col = _col("c0", "VALUE", 800, 880)
    cells = [_blank_cell(col)]
    row = _row(0, [_tok("F", 100, 100)], cells)
    a = _tok("3.1", 810, 100)
    b = _tok("3.2", 840, 100)
    result = _result([row], [col], [a, b])
    fill = recover_dense_empty_cells(result)
    _assert(fill == 0 and cells[0].is_blank,
            "ambiguous two-value cell stays empty")


# Integration: recovered cells are emitted with RECOVERED_BY_GEOMETRY_FALLBACK.
SCHEMA = [{
    "id": "C001", "page": 13, "region": "cylinder_pressure",
    "name": "pmax", "aliases": ["pmax"], "unit": "bar",
}]


def test_recovery_flag_in_field_records():
    cols = [_col("cyl1", "CYL1", 380, 460), _col("cyl2", "CYL2", 470, 550)]
    cells = [_populated_cell(cols[0], "48.0", 390), _blank_cell(cols[1])]
    row = _row(0, [_tok("pmax", 100, 100)], cells)
    stray = _tok("49.0", 480, 100)
    result = _result([row], cols, [stray])
    frs = build_field_records(result, SCHEMA, 13, "cylinder_pressure")
    _assert(len(frs) == 1, "one field record")
    c2 = [c for c in frs[0].cells if c.column == "CYL2"][0]
    _assert(c2.value == 49.0, "recovered value emitted")
    _assert(RECOVERED_BY_GEOMETRY_FALLBACK in c2.flags,
            "recovered cell carries the fallback flag")


def test_all_tests_pass():
    print("  PASS: recovery synthetic tests")
