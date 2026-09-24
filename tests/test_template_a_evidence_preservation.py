"""Regression contracts for Template-A evidence preservation.

The extractor may classify an ambiguous token, but the crop stage must never
rewrite a value based on a field id, a numeric range, or a desired value.
"""

import pytest

from meridian.crop_pipeline import _postprocess_record
from meridian.models import Cell, FieldRecord


def _record(field_id, values):
    cells = [
        Cell(
            field_id=field_id, field_name="test", page=14, region="test",
            column=f"VAL{i}", raw_text=raw, value=value, unit="",
            confidence=0.9, bbox=[100 + i * 20, 20, 10, 10], source_row="test",
        )
        for i, (raw, value) in enumerate(values, 1)
    ]
    return FieldRecord(field_id, "test", 14, "test", "", True, "test", 0.9, cells)


@pytest.mark.parametrize(
    "field_id,values",
    [
        ("F006", [("-165", -165.0), ("50", 50.0)]),
        ("F007", [("0.48", 0.48), ("6", 6.0)]),
        ("F009", [("43730", 43730.0), ("0", 0.0), ("43730", 43730.0)]),
        ("F010", [("15", 15.0), ("911.5", 911.5)]),
        ("F012", [("125", 125.0)]),
        ("L002", [("73", 73.0)]),
        ("L003", [("0.60", 0.60), ("0.70", 0.70)]),
        ("L006", [("15", 15.0), ("940", 940.0)]),
        ("L007", [("40", 40.0), ("924", 924.0)]),
        ("EM012", [("74", 74.0)]),
        ("P001", [(str(i), float(i)) for i in range(8)]),
        ("P003", [(str(i), float(i)) for i in range(4)]),
    ],
)
def test_crop_postprocessing_preserves_all_ocr_evidence(field_id, values):
    record = _record(field_id, values)
    before = [(c.column, c.raw_text, c.value, list(c.flags)) for c in record.cells]

    returned = _postprocess_record(record)

    after = [(c.column, c.raw_text, c.value, list(c.flags)) for c in returned.cells]
    assert after == before


def test_zero_and_negative_ocr_values_are_preserved():
    record = _record("F006", [("-0.4", -0.4), ("0", 0.0)])
    assert [c.value for c in _postprocess_record(record).cells] == [-0.4, 0.0]


def test_missing_decimal_stays_ambiguous_evidence_not_inferred():
    record = _record("EM012", [("74", 74.0)])
    cell = _postprocess_record(record).cells[0]
    assert cell.raw_text == "74"
    assert cell.value == 74.0
    assert "decimal_repaired" not in cell.flags
