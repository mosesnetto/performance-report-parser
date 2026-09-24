from pathlib import Path

from meridian.registry import load_fields


def test_field_count():
    fields=load_fields(Path(__file__).resolve().parents[1]/"config"/"fields.yaml")
    assert len(fields) >= 120
