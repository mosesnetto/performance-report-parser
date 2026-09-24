"""Tests for MERIDIAN Voice (offline Piper briefings)."""
from pathlib import Path
from unittest.mock import patch

from meridian.voice import (
    briefing_text,
    synthesize,
    voice_available,
)

FIELDS = [
    {
        "field_id": "G001", "field_name": "Date and time of recording",
        "found": True, "row_confidence": 99.0,
        "cells": {"DATETIME": [{"value": "2025.04.10 08:30"}]},
    },
    {
        "field_id": "T001", "field_name": "Turbocharger inlet temperature",
        "found": True, "row_confidence": 98.5,
        "cells": {"VALUE": [{"value": 410.2}]},
    },
    {
        "field_id": "T002", "field_name": "Turbocharger compressor MAP",
        "found": False, "row_confidence": 0.0, "cells": {},
    },
    {
        "field_id": "C001", "field_name": "Crank-pin temperature",
        "found": True, "row_confidence": 72.0,
        "cells": {"VALUE": [{"value": 51.4}]},
    },
]

PAYLOAD = {
    "summary": {"filename": "2025__04__CE__01__ME_PERFORMANCE_REPORT_114.pdf"},
    "data": {"rows": FIELDS},
}


def test_briefing_mentions_counts_and_confidence():
    text = briefing_text(PAYLOAD)
    assert "3 of 4 measurement fields" in text
    assert "90 percent" in text  # average of 99.0, 98.5, 72.0
    assert "MERIDIAN engine intelligence briefing" in text


def test_briefing_flags_missing_and_low_confidence():
    text = briefing_text(PAYLOAD)
    assert "1 fields could not be matched" in text
    assert "turbocharger compressor map" in text
    assert "low OCR confidence" in text
    assert "crank-pin temperature" in text


def test_briefing_includes_date():
    text = briefing_text(PAYLOAD)
    assert "recording date is" in text


def test_briefing_empty_report_is_graceful():
    text = briefing_text({"summary": {}, "data": {"rows": []}})
    assert "MERIDIAN engine intelligence briefing" in text
    assert "No extracted measurements" in text


@patch("meridian.voice.shutil.which", return_value=None)
@patch.dict("os.environ", {}, clear=True)
def test_voice_unavailable_without_piper(mock_which):
    assert voice_available() is False


@patch.dict("os.environ", {
    "MERIDIAN_PIPER_BIN": "/opt/piper/piper",
    "MERIDIAN_PIPER_MODEL": "/opt/piper/en_US-lessac-medium.onnx",
}, clear=True)
def test_voice_available_when_configured():
    assert voice_available() is True


@patch.dict("os.environ", {}, clear=True)
def test_synthesize_returns_false_when_unavailable():
    out = Path("/tmp/opencode/www_never.wav")
    assert synthesize("hello", out) is False
    assert not out.exists()


@patch("meridian.voice.subprocess.run")
@patch.dict("os.environ", {
    "MERIDIAN_PIPER_BIN": "/opt/piper/piper",
    "MERIDIAN_PIPER_MODEL": "/opt/piper/en_US-lessac-medium.onnx",
}, clear=True)
def test_synthesize_pipes_text_to_piper(mock_run):

    class FakeProc:
        returncode = 0

    mock_run.return_value = FakeProc()
    out = Path("/tmp/opencode/www_ok.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"RIFF-fake-wav")
    try:
        assert synthesize("briefing text", out) is True
        args = mock_run.call_args.args[0]
        assert mock_run.call_args.kwargs["input"] == b"briefing text"
        assert "--model" in args
        assert "--output_file" in args
    finally:
        out.unlink(missing_ok=True)


def test_piper_binary_uses_env_when_no_which():
    import os
    env_bin = "/opt/piper/piper"
    with patch.dict(os.environ, {"MERIDIAN_PIPER_BIN": env_bin}, clear=True):
        from meridian.voice import piper_binary
        assert piper_binary() == [env_bin]