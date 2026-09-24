from pathlib import Path

import pytest

from meridian.extract import _scan_ce_datetime, parse_num, parse_numeric_ocr
from meridian.models import OCRRow, OCRWord
from meridian.registry import load_fields


def _ce_row(pairs):
    """Build a CE 'Date and time' row from (text, left) pairs."""
    words = [
        OCRWord(page=1, region="general", text=t, conf=90.0, left=l,
                top=100, width=30, height=20)
        for t, l in pairs
    ]
    return OCRRow(page=1, region="general", words=words)


def test_ce_datetime_ocr_split_time_pair():
    # OCR returned the clock time as two adjacent words on the G001 row,
    # with the recurring "[15]" box artifact (misread as "fis]") preceding it.
    row = _ce_row([("4/15/2025", 598), ("fis]", 737), ("21:", 803), ("40", 853),
                   ("Sea water temp.", 912)])
    assert _scan_ce_datetime(row) == "2025-04-15 21:40"


def test_ce_datetime_ocr_box_artifact_is_skipped():
    # The box artifact before the time must not end the time run; a complete
    # time token after it is still captured.
    row = _ce_row([("10/25/2025", 600), ("[15]", 750), ("08:50", 820), ("Sea", 880)])
    assert _scan_ce_datetime(row) == "2025-10-25 08:50"


def test_ce_datetime_complete_time_token():
    row = _ce_row([("10/25/2025", 600), ("08:50", 820), ("Sea", 880)])
    assert _scan_ce_datetime(row) == "2025-10-25 08:50"


@pytest.mark.parametrize("dotted", ["19.90", "20.60", "25.0"])
def test_ce_datetime_dotted_number_is_never_a_time(dotted):
    # Dotted values on the G001 row must never be reconstructed into times.
    row = _ce_row([("4/15/2025", 598), (dotted, 820), ("Sea water temp.", 912)])
    assert _scan_ce_datetime(row) == "2025-04-15"


def test_ce_datetime_split_pair_requires_same_row_adjacency():
    # The minute token must sit immediately after the hour fragment (adjacent
    # x positions); a far-away two-digit token is not part of the time.
    row = _ce_row([("4/15/2025", 598), ("21:", 803), ("40", 950), ("Sea", 1000)])
    assert _scan_ce_datetime(row) == "2025-04-15"


def test_ce_datetime_split_pair_minute_must_be_bare_two_digit():
    # A dotted measurement that happens to follow the hour fragment is not a
    # minute and must not be combined into a time.
    row = _ce_row([("4/15/2025", 598), ("21:", 803), ("25.0", 853), ("Sea", 912)])
    assert _scan_ce_datetime(row) == "2025-04-15"


def test_parse_num_debris_suffix_behavior():
    # Header/column suffixes are not numbers.
    assert parse_num("CYL10") is None
    assert parse_num("TC3") is None
    assert parse_num("BRG12") is None
    assert parse_num("10") == 10.0
    # A literal backslash followed by a letter/digit is NOT accepted by the
    # conservative strict parser (the numeric-debris suffix matcher only strips
    # non-word punctuation). "-2\0" / "0\e" therefore stay None -- reverting the
    # rejected loosening that caused B002 to gain spurious values.
    assert parse_num("-2\\0") is None
    assert parse_num("0\\e") is None
    # A bare alphabetic tail is rejected (not a number).
    assert parse_num("stable") is None
    assert parse_num("RPM84") is None
    # Colon time/date debris is rejected (Rule A).
    assert parse_num("21:") is None


def test_parse_numeric_ocr_cleans_control_glyph():
    # The debris-aware parser also cleans the real OCR NUL control glyph.
    word = OCRWord(page=1, region="r", text="-2\x00", conf=50.0, left=0, top=0, width=10, height=10)
    assert parse_numeric_ocr(word, {"name": "x", "id": "X"}) == -2.0


def test_registry_has_125_fields_with_nv_t001():
    fields = load_fields(Path(__file__).resolve().parents[1] / "config" / "fields.yaml")
    assert len(fields) == 125
    nv = [f for f in fields if f["id"] == "T001"]
    assert len(nv) == 1
    assert nv[0]["name"] == "NV (tot-tot)"
