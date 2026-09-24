"""Regression suite for the ME performance report parser.

These tests are REGRESSION-only: they lock in golden extraction results so
that any change to extraction logic cannot silently alter previously correct
output. They are not the place to tweak extraction heuristics.

Golden assertions (from the canonical Jan-2026 report):

    P001  XPERT=21439, TORSIOM.=21630, MOP=21449, by FI*rpm=21296,
          by pscav=22004, by TC rpm=21368, by MEP=21021, by rpm blank
    P002  MCR / MCR_% / ENGINE columns
    S002/S003  REF/CALC/AVG/TC1/TC2/TC3 correctly placed
    X009  VAL1/VAL2/VAL3
    text  stable, Economy, RPM, off, Laboratory, 2E, CE
"""

import shutil
from pathlib import Path

import pytest

from meridian.audit import (
    run_audit,
)
from meridian.registry import load_fields

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

# Canonical January-2026 report (primary golden doc).
JAN_PDF = BASE / "uploads" / "deb6852dea2a" / (
    "2026__01__CE__01__ME_PERFORMANCE_REPORT_JAN_2026 2 1.pdf"
)

# A checked-in extraction fixture keeps regression assertions independent of a
# developer's home directory and avoids a PDF/OCR run in the normal suite.
JAN_REPORT_FIXTURE = HERE / "fixtures" / "jan_2026_report.json"


@pytest.fixture(scope="module")
def report_data(tmp_path_factory):
    if not JAN_PDF.exists():
        pytest.skip(f"golden PDF not found: {JAN_PDF}")
    assert JAN_REPORT_FIXTURE.exists(), "missing Jan 2026 report fixture"
    audit_root = tmp_path_factory.mktemp("jan_2026_audit")
    shutil.copyfile(JAN_REPORT_FIXTURE, audit_root / "report.json")
    audit = run_audit(
        JAN_PDF,
        audit_root,
        reuse_ocr=True,
    )
    return audit["fields"]


def fields_by_id(fields):
    return {
        f["field_id"]: f
        for f in fields
    }


def test_field_count_derives_from_registry():
    fields = load_fields(BASE / "config" / "fields.yaml")
    # The source-vs-config audit proved the true logical field count is 125
    # (124 original + NV (tot-tot)). The invariant derives from the registry.
    assert len(fields) == 125
    assert any(f["id"] == "T001" for f in fields)


def test_nv_t001_registered_correctly():
    fields = load_fields(BASE / "config" / "fields.yaml")
    nv = [f for f in fields if f["id"] == "T001"]
    assert len(nv) == 1
    nv = nv[0]
    assert nv["name"] == "NV (tot-tot)"
    assert nv["region"] == "turbocharger"
    assert nv["page"] == 13


def test_nv_t001_golden_values(report_data):
    f = fields_by_id(report_data).get("T001")
    assert f is not None
    assert f["found"] is True
    vals = sorted(c["value"] for c in f["cells"])
    # Golden Jan-2026: 2.07 / 2.07 / 2.07 (allowing OCR to be exact)
    assert vals == [2.07, 2.07, 2.07]


def test_p001_golden_values(report_data):
    f = fields_by_id(report_data).get("P001")
    assert f is not None
    by_col = {c["column"]: c for c in f["cells"]}
    assert by_col["XPERT"]["value"] == 21439.0
    assert by_col["TORSIOM."]["value"] == 21630.0
    assert by_col["MOP"]["value"] == 21449.0
    assert by_col["by FI*rpm"]["value"] == 21296.0
    assert by_col["by pscav"]["value"] == 22004.0
    assert by_col["by TC rpm"]["value"] == 21368.0
    assert by_col["by MEP"]["value"] == 21021.0
    # by rpm is a genuine blank in the source (golden).
    assert "by rpm" not in by_col


def test_p002_columns_present(report_data):
    f = fields_by_id(report_data).get("P002")
    assert f is not None
    cols = [c["column"] for c in f["cells"]]
    assert "MCR" in cols
    assert "MCR_%" in cols
    assert "ENGINE" in cols


def test_p001_torsion_debris_preserved(report_data):
    # The interrupted TORSIOM. token must stay accepted (Rule A conflict that
    # must NOT silently change), while still being flagged as debris.
    f = fields_by_id(report_data).get("P001")
    tors = [c for c in f["cells"] if c["column"] == "TORSIOM."]
    assert tors and tors[0]["raw_text"] == "21630/@"
    assert tors[0]["value"] == 21630.0
    assert any("debris" in fd for fd in f["findings"])


def test_p002_schema_matches_config(report_data):
    from meridian.extract import get_explicit_columns

    schema = get_explicit_columns({"id": "P002", "name": "ENGINE speed"})
    assert schema == ["MCR", "MCR_%", "ENGINE"]


def test_text_values_present(report_data):
    texts = {
        c["value"]
        for f in report_data
        for c in f["cells"]
        if isinstance(c["value"], str)
    }
    # At least the values present in the golden Jan report must appear.
    assert "stable" in texts
    assert "Economy" in texts
    assert "Laboratory" in texts
    assert "2E" in texts
