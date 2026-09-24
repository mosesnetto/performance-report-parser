"""UI/API tests for the FINAL APPLICATION web layer.

These tests use a temporary OUTPUT_DIR (monkeypatched) and the Flask test
client, so they are self-contained and do not depend on staged reports.

They cover: report loading, field filtering structure, field detail, semantic
columns, graph data, multi-report comparison identity, and the audit endpoint.
"""

import json

import pytest

import meridian.web as web


def _cell(column, raw_text, value, confidence=0.9, flags=None):
    return {
        "column": column,
        "raw_text": raw_text,
        "value": value,
        "confidence": confidence,
        "flags": flags or [],
        "bbox": [677, 3, 66, 18],
        "source_row": "row",
    }


def _field(fid, name, cells, region="power_speed", unit="", row_text="row"):
    return {
        "field_id": fid,
        "field_name": name,
        "page": 13,
        "region": region,
        "unit": unit,
        "found": True,
        "row_confidence": 0.9,
        "row_text": row_text,
        "flags": [],
        "cells": cells,
    }


def _build_report():
    fields = [
        _field(
            "P001",
            "ENGINE power estimated",
            [
                _cell("XPERT", "21439", 21439.0, confidence=0.53, flags=["low_ocr_confidence"]),
                _cell("TORSIOM.", "21630/@", 21630.0, confidence=0.17, flags=["low_ocr_confidence"]),
                _cell("MOP", "21449", 21449.0, confidence=0.9),
            ],
            unit="kW",
        ),
        _field(
            "P002",
            "ENGINE speed",
            [
                _cell("MCR", "84.0", 84.0),
                _cell("MCR_%", "80.1", 80.1),
                _cell("ENGINE", "67.3", 67.3),
            ],
            unit="rpm",
        ),
        _field(
            "T001",
            "NV (tot-tot)",
            [
                _cell("VAL1", "2.07", 2.07),
                _cell("VAL2", "2.07", 2.07),
                _cell("VAL3", "2.07", 2.07),
            ],
            region="turbocharger",
        ),
        _field("S002", "Suction press.", [_cell("AVG", "20", 20.0), _cell("TC1", "20", 20.0)], region="scavenge_air", unit="mmWG"),
    ]
    return {"metadata": {"reading_pages": [13, 14]}, "fields": fields, "validation": {}}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    job_id = "jad" * 4
    out = tmp_path / "out"
    report_dir = out / job_id
    report_dir.mkdir(parents=True)
    report = _build_report()
    (report_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps({"expected_fields": 4}), encoding="utf-8")
    (report_dir / "audit.json").write_text(
        json.dumps(_audit_for(report)), encoding="utf-8"
    )
    monkeypatch.setattr(web, "OUTPUT_DIR", out)
    monkeypatch.setattr(web, "REPORT_INDEX", out / "reports.json")
    web.app.config["TESTING"] = True
    return web.app.test_client()


def _audit_for(report):
    from meridian.audit import audit_report

    return audit_report(report, summary={"expected_fields": len(report["fields"])})


def test_report_loading_semantic_rows(client):
    r = client.get("/api/report/jadjadjadjad")
    assert r.status_code == 200
    j = r.get_json()
    assert j["ok"] is True
    rows = j["data"]["rows"]
    assert len(rows) == 4
    for row in rows:
        assert "field_id" in row
        assert "schema" in row
        assert "cells" in row
    assert any(x["field_id"] == "T001" for x in rows)


def test_semantic_columns_exposed(client):
    j = client.get("/api/report/jadjadjadjad").get_json()
    rows = {x["field_id"]: x for x in j["data"]["rows"]}
    assert rows["P002"]["schema"] == ["MCR", "MCR_%", "ENGINE"]
    assert rows["P001"]["schema"][:3] == ["XPERT", "TORSIOM.", "MOP"]
    nv = rows["T001"]
    assert sorted(nv["cells"].keys()) == ["VAL1", "VAL2", "VAL3"]
    assert [c["value"] for c in nv["cells"]["VAL1"]] == [2.07]


def test_cell_semantic_fields(client):
    j = client.get("/api/report/jadjadjadjad").get_json()
    rows = {x["field_id"]: x for x in j["data"]["rows"]}
    # Semantic column identity is the cells map key.
    cells = rows["P001"]["cells"]
    assert "TORSIOM." in cells
    cell = cells["TORSIOM."][0]
    assert cell["value"] == 21630.0
    assert cell["raw_text"] == "21630/@"
    assert "confidence" in cell
    assert "bbox" in cell


def test_field_filtering_row_identity(client):
    j = client.get("/api/report/jadjadjadjad").get_json()
    rows = j["data"]["rows"]
    ids = [x["field_id"] for x in rows]
    # field identity is field_id (not column index)
    assert len(set(ids)) == len(ids)


def test_graph_data_semantic_structure(client):
    # The graphs page consumes rows with schema + cells.
    j = client.get("/api/report/jadjadjadjad").get_json()
    for row in j["data"]["rows"]:
        assert "schema" in row
        assert "cells" in row
        assert "region" in row
        assert "unit" in row


def test_audit_endpoint_counts(client):
    r = client.get("/api/report/jadjadjadjad/audit")
    assert r.status_code == 200
    j = r.get_json()
    c = j["counts"]
    for key in [
        "registered_source_fields",
        "unregistered_source_fields",
        "fields_missing",
        "ocr_ambiguous",
        "column_mismatch",
        "text_values",
        "errors",
    ]:
        assert key in c
    assert j["integrity"]["expected_count_invariant_holds"] is True
    assert isinstance(j["field_status"], dict)


def test_graph_comparison_identity_semantics():
    # Comparison identity = field_id + semantic column; raw parser logic is
    # not duplicated in JS: cells are keyed by semantic column.
    report = _build_report()
    data = web.normalize_web_data(report)
    rows = {x["field_id"]: x for x in data["rows"]}
    # P001 "MOP" is keyed by its semantic column regardless of table position.
    assert "MOP" in rows["P001"]["cells"]
    assert [c["value"] for c in rows["P001"]["cells"]["MOP"]] == [21449.0]
    # NV is keyed by VAL1/VAL2/VAL3.
    assert sorted(rows["T001"]["cells"].keys()) == ["VAL1", "VAL2", "VAL3"]
