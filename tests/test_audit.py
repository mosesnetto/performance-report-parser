"""Unit tests for the self-auditing engine (app/audit.py).

These test the audit CLASSIFICATION logic in isolation with synthetic
records, so they are fast and independent of OCR runs.
"""

from meridian.audit import (
    COLUMN_MISMATCH,
    MISSING_VALUE,
    OCR_AMBIGUOUS,
    OCR_MALFORMED,
    OK,
    POSSIBLE_POSITIONAL_SHIFT,
    SCHEMA_MISMATCH,
    SOURCE_BLANK,
    TEXT_VALUE,
    audit_report,
)


def _cell(column, raw_text, value, confidence=0.9, flags=None):
    return {
        "column": column,
        "raw_text": raw_text,
        "value": value,
        "confidence": confidence,
        "bbox": [0, 0, 10, 10],
        "flags": flags or [],
    }


def _field(fid, name, cells, found=True, row_text="", flags=None):
    return {
        "field_id": fid,
        "field_name": name,
        "page": 13,
        "region": "general",
        "unit": "",
        "found": found,
        "row_text": row_text,
        "flags": flags or [],
        "cells": cells,
    }


def test_ok_field():
    fields = [
        _field(
            "P002",
            "ENGINE speed",
            [
                _cell("MCR", "84.0", 84.0),
                _cell("MCR_%", "80.1", 80.1),
                _cell("ENGINE", "67.3", 67.3),
            ],
        )
    ]
    audit = audit_report({"fields": fields})
    fa = audit["fields"][0]
    assert OK in fa["statuses"]
    assert not any(
        s in fa["statuses"]
        for s in (SOURCE_BLANK, OCR_AMBIGUOUS, MISSING_VALUE, COLUMN_MISMATCH)
    )


def test_field_count_invariant_reported():
    fields = [_field(f"F{i:03d}", f"field {i}", []) for i in range(124)]
    audit = audit_report(
        {"fields": fields},
        summary={"expected_fields": 124},
    )
    assert audit["integrity"]["expected_count_invariant_holds"] is True
    assert audit["integrity"]["records_in_report"] == 124


def test_blank_source_vs_lost_value():
    # No cells, row has digits -> MISSING_VALUE (lost), not blank.
    lost = audit_report(
        {"fields": [_field("F001", "x", [], row_text="label 123 456")]}
    )
    assert MISSING_VALUE in lost["fields"][0]["statuses"]

    # No cells, row has no digits -> SOURCE_BLANK.
    blank = audit_report(
        {"fields": [_field("F002", "y", [], row_text="some indicator label")]}
    )
    assert SOURCE_BLANK in blank["fields"][0]["statuses"]


def test_debris_cell_is_ambiguous():
    cells = [
        _cell("CYL1", "21 /®", 21.0, confidence=0.37),
    ]
    fa = audit_report({"fields": [_field("C001", "z", cells)]})["fields"][0]
    assert OCR_AMBIGUOUS in fa["statuses"]
    # Raw token preserved.
    assert fa["cells"][0]["raw_text"] == "21 /®"
    assert any("debris" in fd for fd in fa["findings"])


def test_unresolved_column_is_mismatch():
    fa = audit_report(
        {
            "fields": [
                _field(
                    "X004",
                    "w",
                    [_cell("VAL1", "0.95", 0.95, flags=["column_unresolved"])],
                )
            ]
        }
    )["fields"][0]
    assert COLUMN_MISMATCH in fa["statuses"]


def test_schema_mismatch_detected():
    # Field with explicit REF/CALC/AVG schema but a cell on an invalid column.
    from meridian.extract import get_explicit_columns

    get_explicit_columns({"id": "C001", "name": "x"})
    fa = audit_report(
        {
            "fields": [
                _field(
                    "C001",
                    "x",
                    [_cell("BOGUS", "5.0", 5.0)],
                )
            ]
        }
    )["fields"][0]
    assert SCHEMA_MISMATCH in fa["statuses"]


def test_sparse_row_flags_positional_shift():
    # Expected 12 CYL columns but only 6 values captured.
    cells = [_cell(f"CYL{i}", f"{i}", float(i)) for i in range(1, 7)]
    fa = audit_report({"fields": [_field("X002", "cyl pressure", cells)]})["fields"][0]
    assert POSSIBLE_POSITIONAL_SHIFT in fa["findings"]


def test_text_value_detected():
    fa = audit_report(
        {
            "fields": [
                _field(
                    "G002",
                    "running mode",
                    [_cell("VALUE", "stable", "stable")],
                )
            ]
        }
    )["fields"][0]
    assert TEXT_VALUE in fa["statuses"]


def test_ocr_malformed_for_unparseable():
    fa = audit_report(
        {
            "fields": [
                _field("G014", "voy", [_cell("VALUE", "CEW", None, confidence=0.9)])
            ]
        }
    )["fields"][0]
    assert OCR_MALFORMED in fa["statuses"]


def test_cell_integrity_detects_missing_keys():
    # A cell missing required structural keys must be flagged.
    broken = {
        "field_id": "P001",
        "field_name": "engine power",
        "page": 13,
        "region": "general",
        "unit": "",
        "found": True,
        "row_text": "engine power 100",
        "flags": [],
        "cells": [
            {
                "column": "XPERT",
                "raw_text": "100",
                "value": 100.0,
                # missing confidence, bbox, source_row, field_id, field_name
            }
        ],
    }
    audit = audit_report({"fields": [broken]})
    assert audit["integrity"]["cell_integrity_ok"] is False
    assert len(audit["integrity"]["cell_integrity_violations"]) == 1

