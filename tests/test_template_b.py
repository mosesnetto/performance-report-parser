"""Regression tests for Template-B semantic correctness.

These are fast synthetic unit tests covering the two real defects fixed in
this pass:

* G001 "Date and time of recording" must never fabricate a datetime from
  unrelated page tokens (e.g. a footer date or an STW ship-speed "19.00").
* The TC / CYL value columns must be placed by physical x-geometry with blank
  slots preserved (a missing TC1 must not shift TC2/TC3 left), and T003
  "TC speed 298" must only match a row that actually carries the 298
  qualifier.
"""

from meridian.models import OCRRow, OCRWord
from meridian.template_b import (
    _collect_datetime,
    _enrich_schema_anchors,
    _parse,
    _scan_datetime,
    find_tb_row,
)


def _row(text_list, page=11):
    """Build an OCRRow from a list of (text, left) tuples."""
    words = []
    for i, (text, left) in enumerate(text_list):
        words.append(OCRWord(
            page=page, region="reading", text=text, conf=80.0,
            left=left, top=300, width=40, height=15,
        ))
    return OCRRow(page=page, region="reading", words=words)


def _g001_field():
    return {"id": "G001", "name": "Date and time of recording", "unit": ""}


def _t003_field():
    return {"id": "T003", "name": "TC speed 298", "unit": "rpm"}


def _t002_field():
    return {"id": "T002", "name": "TC speed", "unit": "rpm"}


# ---------------------------------------------------------------------------
# PART 1 - DATETIME must not be fabricated from unrelated page content
# ---------------------------------------------------------------------------

def test_datetime_dotted_stw_speed_is_not_a_time():
    # Same physical row as TB_A: date 5/24/2025 followed by STW 19.00 (a
    # dotted ship-speed reading).  A dotted number is NOT accepted as a time,
    # so only the date is emitted and the STW speed is never fabricated as a
    # recording time.
    row = _row([
        ("Date", 200), ("and", 260), ("time", 300), ("of", 340),
        ("recording", 370), ("5/24/2025", 600), ("STW", 1200), ("19.00", 1350),
    ])
    assert _scan_datetime(row.words) == "2025-05-24"


def test_datetime_colon_time_is_used():
    # A colon-separated time on the G001 row is an unambiguous clock time and
    # IS combined with the date on the same row.
    row = _row([
        ("Date", 200), ("and", 260), ("time", 300), ("of", 340),
        ("recording", 370), ("5/24/2025", 600), ("[15]", 700), ("10:34]", 760),
        ("STW", 1200), ("19.00", 1350),
    ])
    assert _collect_datetime(row, _g001_field(), None, None) == "2025-05-24 10:34"


def test_datetime_row_without_date_is_missing():
    # TB_B case: the matched G001 row carries no date token at all.  No
    # datetime is fabricated even though unrelated page rows (footer) hold a
    # date -- page_rows are never scanned.
    row = _row([
        ("Date", 200), ("and", 260), ("time", 300), ("of", 340),
        ("recording", 370), ("Sea", 900), ("water", 940), ("temp.", 980),
        ("28.0", 1030), ("STW", 1200), ("18.50", 1329),
    ])
    # An unrelated footer row that DOES contain a date/time.
    footer = _row([("11/24/2025", 437), ("02:19", 500)], page=11)
    dt = _collect_datetime(row, _g001_field(), None, None, page_rows=[footer])
    assert dt is None


def test_datetime_page_rows_cannot_leak_unrelated_date():
    # Even when the caller supplies page_rows containing a date (as the old
    # unsafe fallback did), that unrelated page date must never become G001.
    row = _row([("Date", 200), ("recording", 370)])
    footer = _row([("11/24/2025", 437), ("02:19", 500), ("some", 600), ("value", 640)])
    assert _collect_datetime(row, _g001_field(), None, None, page_rows=[footer]) is None


# ---------------------------------------------------------------------------
# PART 2/3 - TC semantics: blank slots preserved, qualifier matching
# ---------------------------------------------------------------------------

def _cyl_header(y, x0=640, dx=85):
    names = ["REF", "CALC", "AVG", "CYL1", "CYL2", "CYL3"] + [f"CYL{i}" for i in range(4, 11)]
    return {
        "y": y,
        "columns": [{"column": n, "x": x0 + i * dx} for i, n in enumerate(names)],
    }


def _meta_header(y, has_tc1):
    # A fragmented TC header missing TC1 (the real-world defect).
    cols = [
        {"column": "REF", "x": 640},
        {"column": "CALC", "x": 725},
        {"column": "AVG", "x": 810},
    ]
    if has_tc1:
        cols.append({"column": "TC1", "x": 895})
    cols.append({"column": "TC2", "x": 1010})
    cols.append({"column": "TC3", "x": 1120})
    return {"y": y, "columns": cols}


def test_enrich_tc1_reconstructed_from_sibling_cyl_header():
    # The chosen TC header is fragmented and missing TC1.  The generalized
    # mechanism must reconstruct TC1's x from the sibling CYL header by ordinal
    # rank (TC1 <-> CYL1 at the same x), preserving the blank TC1 slot instead
    # of letting a later value slide into it.
    schema = ["REF", "CALC", "AVG", "TC1", "TC2", "TC3"]
    chosen = _meta_header(1600, has_tc1=False)
    sibling = _cyl_header(1400)
    all_headers = [chosen, sibling]

    ax = _enrich_schema_anchors(schema, chosen, all_headers)
    # TC1 is reconstructed to CYL1's x (895).
    assert ax["REF"] == 640
    assert ax["CALC"] == 725
    assert ax["AVG"] == 810
    assert ax["TC1"] == 895
    assert ax["TC2"] == 1010
    assert ax["TC3"] == 1120
    # TC slots stay ordered and distinct.
    assert ax["TC1"] < ax["TC2"] < ax["TC3"]


def test_find_tb_row_t003_matches_only_298_row():
    # T003 "TC speed 298" must match only the row that contains the 298
    # qualifier, never the plain "TC speed" row.
    rows = [
        _row([("TC", 100), ("speed", 140), ("rpm", 190), ("7928", 600), ("7960", 700)]),
        _row([("TC", 100), ("speed", 140), ("298", 200), ("rpm", 250), ("8406", 600)]),
    ]
    row003, _ = find_tb_row(rows, _t003_field())
    assert row003 is not None
    assert "298" in row003.text
    # T002 matches only the non-298 row.
    row002, _ = find_tb_row(rows, _t002_field())
    assert row002 is not None
    assert "298" not in row002.text


def test_find_tb_row_t003_absent_when_no_298_row():
    # No "TC speed 298" row exists (both real Template-B docs) -> T003 must be
    # correctly not-found and must NOT fall back onto the plain TC speed row.
    rows = [
        _row([("TC", 100), ("speed", 140), ("rpm", 190), ("7928", 600)]),
    ]
    row003, score = find_tb_row(rows, _t003_field())
    assert row003 is None
    assert score < 0.6


# ---------------------------------------------------------------------------
# PART 6 - decimal ambiguity: never invent a decimal
# ---------------------------------------------------------------------------

def test_parse_integer_never_invents_decimal():
    # OCR only proves "610".  The parser must not turn it into 61.0; it keeps
    # the integer exactly as OCR provided.
    assert _parse("610") == 610.0
    assert _parse("610)") == 610.0
    # A literal decimal present in OCR is preserved literally.
    assert _parse("61.0") == 61.0


# ---------------------------------------------------------------------------
# PART 7 - DATETIME colon-lone split tokens (dpi=300 "10 : 34")
# ---------------------------------------------------------------------------

def test_datetime_colon_lone_split_tokens_combined():
    # At dpi=300 tesseract splits "10:34" into three word boxes "10", ":", "34".
    # The lone ":" colon flanked by a bare 1-2 digit hour and a bare 2-digit
    # minute is an unambiguous clock time and must combine with the date.
    row = _row([
        ("Date", 200), ("and", 260), ("time", 300), ("of", 340),
        ("recording", 370), ("5/24/2025", 600), ("10", 700), (":", 730),
        ("34", 760), ("Sea", 1200), ("water", 1260),
    ])
    assert _scan_datetime(row.words) == "2025-05-24 10:34"


def test_datetime_colon_lone_split_rejects_non_clock_flanks():
    # A ":" reaching into non-clock tokens (multi-digit hour or 3-digit minute)
    # must not be transcribed as a time; also a bare dotted number never is.
    row = _row([
        ("Date", 200), ("recording", 370), ("5/24/2025", 600),
        ("12", 700), (":", 730), ("345", 760),
    ])
    assert _scan_datetime(row.words) == "2025-05-24"
    row2 = _row([
        ("Date", 200), ("recording", 370), ("5/24/2025", 600),
        ("30.0", 700), (":", 730), ("00", 760),
    ])
    assert _scan_datetime(row2.words) == "2025-05-24"


# ---------------------------------------------------------------------------
# PART 8 - scalar protection: corrupted unit token is never a second value
# ---------------------------------------------------------------------------

from meridian.template_b import collect_values_tb  # noqa: E402


def _ambient_field():
    return {"id": "G006", "name": "Ambient air temp.", "unit": "°C"}


def _scalar_collect(field, text_pairs, headers):
    row = _row(text_pairs)
    return collect_values_tb(row, field, [_ambient_field()], headers, 11, page_rows=[])


def _simple_header(y=1600):
    return {"y": y, "columns": [{"column": "VALUE", "x": 1200}]}


def test_scalar_drops_corrupted_unit_token():
    # Source has a single "30.0 °C".  OCR misreads "°C" as "2€".  The scalar
    # must emit only the one well-formed value 30.0 and never a second "2".
    cells = _scalar_collect(
        _ambient_field(),
        [("Ambient", 200), ("air", 260), ("temp.", 300), ("30.0", 1251),
         ("2€", 1313)],
        [_simple_header()],
    )
    [(c.column, c.value) for c in cells]
    assert len(cells) == 1
    assert cells[0].value == 30.0
    assert "scalar_ambiguous" in cells[0].flags


def test_scalar_preserves_multi_column_row():
    # A row with several well-formed numbers (e.g. ENGINE power effective
    # "51070 51.7 26410") is a genuine structured row and must NOT be
    # collapsed by the scalar guard.
    field = {"id": "P003", "name": "ENGINE power effective", "unit": "kW"}
    cells = _scalar_collect(
        field,
        [("ENGINE", 200), ("power", 260), ("effective", 320), ("51070", 749),
         ("51.7", 861), ("%", 913), ("26410", 984)],
        [_simple_header()],
    )
    assert len([c for c in cells if c.value is not None]) >= 3


# ---------------------------------------------------------------------------
# PART 9 - B002 "Temp. deviation": a debris token must never become a
# next-field boundary that truncates a populated value row.
# ---------------------------------------------------------------------------

from meridian.extract import _compute_value_bounds  # noqa: E402


def _b002_field():
    return {"id": "B002", "name": "Temp. deviation", "unit": "°C",
            "anchor_after": "MAIN BEARING temp."}


def _l007_field():
    return {"id": "L007", "name": "Density @ 40 °C", "unit": "kg/m3"}


def _b002_bearing_deviation_row():
    """Real OCR row from the May run: 'temp. deviation °C ALL @ 16 |® -15/0
    38, @ -10 @ 4@ -15,@ 4@ -12/® -13,@ 7'.  The first value '16,@' sits at
    x=935; the neighboring label 'Density @ 40 °C' must NOT anchor a
    next-field boundary there."""
    return _row([
        ("temp.", 330), ("deviation", 375), ("°C", 554), ("ALL", 815),
        ("C)", 881), ("16,@", 935), ("-15/0", 1009), ("38", 1098),
        ("|®", 1123), ("-10|@", 1172), ("4@", 1271), ("-15/@", 1336),
        ("4@", 1435), ("-12/@", 1500), ("-13/@", 1582), ("7", 1682),
    ])


def test_b002_next_field_boundary_not_anchored_on_debris_token():
    # The reported next-field boundary must not start at the value/debris
    # token '16,@' (x=935).  Before the fix it was (375, 419, 935), which led
    # collect_values_tb to truncate every value off the row.
    row = _b002_bearing_deviation_row()
    fields = [_b002_field(), _l007_field()]
    _start, _end, next_field_left = _compute_value_bounds(row, _b002_field(), fields)
    assert next_field_left is None or next_field_left > 1682


def test_b002_all_ten_deviation_values_recovered():
    # With the bogus 935 boundary gone, all ten numeric deviation values are
    # recovered instead of zero cells.
    row = _b002_bearing_deviation_row()
    fields = [_b002_field(), _l007_field()]
    cells = collect_values_tb(row, _b002_field(), fields, [], 11, page_rows=[])
    vals = [c.value for c in cells]
    assert vals == [16.0, -15.0, 38.0, -10.0, 4.0, -15.0, 4.0, -12.0, -13.0, 7.0]


def test_fuzzy_next_field_still_detected_on_alphabetic_label():
    # Guard sanity check: a genuine next-field label that is only findable via
    # the tolerant fuzzy matcher (one OCR-mangled token, "BEAR" -> "b3ar") must
    # still be detected.  The reported boundary must anchor at the alphabetic
    # label word "CRANK", NOT be dropped and NOT drift onto a value/debris
    # token.  This proves the guard blocks only debris-token starts.
    words = [
        ("MAIN", 200, 60), ("BEARING", 270, 70), ("temp.", 350, 44),
        ("84.2", 600, 40), ("83.9", 660, 40),
        ("CRANK", 900, 60), ("PIN", 970, 40), ("b3ar", 1020, 44),
        ("temp", 1070, 44),
    ]
    row = _row([(t, l) for (t, l, _w) in words])
    b001 = {"id": "B001", "name": "MAIN BEARING temp.", "unit": "°C"}
    b003 = {"id": "B003", "name": "CRANK PIN BEAR. temp", "unit": "°C"}
    _start, _end, next_field_left = _compute_value_bounds(row, b001, [b001, b003])
    assert next_field_left == 900


# ---------------------------------------------------------------------------
# PART 10 - geometry-mismatch collision guard (B003/B005 style)
# A header whose column spacing does not match the value row must not collapse
# two values onto one column; the whole row falls back to schema order.
# ---------------------------------------------------------------------------

def _b003_field():
    return {"id": "B003", "name": "CRANK PIN BEAR. temp", "unit": "°C"}


def _wide_cyl_header(y):
    # REF/CALC/AVG/CYL1..CYL10/MOP at wide (~99px) spacing - the header the
    # max-overlap rule selects, whose geometry does NOT match the ~79px value
    # row (the production B003/B005 defect).
    names = ["REF", "CALC", "AVG"] + [f"CYL{i}" for i in range(1, 11)] + ["MOP"]
    xs = [760, 877, 994, 1102, 1202, 1300, 1398, 1496, 1595, 1694, 1792,
          1890, 1992, 2087]
    return {"y": y, "columns": [
        {"column": n, "x": x} for n, x in zip(names, xs)
    ]}


def test_geometry_collision_falls_back_to_schema_order():
    # 11 B003-like values at ~79px spacing under a wide (~99px) header.  The
    # selected header is complete for the AVG+CYL1..CYL10 schema, so values are
    # placed by x-geometry; that geometry collides (two values would snap to the
    # same CYL).  The row must be re-mapped in schema order: exact
    # AVG,CYL1..CYL10, no duplicate column, every value present.
    field = _b003_field()
    values = [52.6, 52.0, 53.0, 52.0, 52.0, 53.0, 56.0, 53.0, 52.0, 51.0, 52.0]
    lefts = [1011, 1089, 1168, 1247, 1325, 1404, 1483, 1562, 1640, 1719, 1800]
    row = _row(
        [("CRANK", 200), ("PIN", 260), ("BEAR.", 300), ("temp", 360)]
        + [(f"{v:g}", l) for v, l in zip(values, lefts)]
    )
    headers = [_wide_cyl_header(400)]
    cells = collect_values_tb(row, field, [field], headers, 14, page_rows=[])
    cols = [c.column for c in cells]
    assert cols == ["AVG"] + [f"CYL{i}" for i in range(1, 11)]
    assert len(cols) == len(set(cols))
    assert [c.value for c in cells] == values


def test_correctly_spaced_blank_middle_slot_preserved():
    # A correctly spaced CYL row with a missing middle value (CYL6) must keep
    # geometry authoritative: the absent slot stays blank and no later value is
    # shifted left.  No geometry collision occurs, so the fallback must NOT be
    # triggered (which would have collapsed this to CYL1..CYL8).
    field = {"id": "LW002", "name": "LINER WALL temp. fp", "unit": "°C"}
    schema = ["CYL1", "CYL2", "CYL3", "CYL4", "CYL5", "CYL6",
              "CYL7", "CYL8", "CYL9", "CYL10"]
    # Header exactly matches the value row spacing (79px); all 10 columns known.
    hx0 = 1098
    headers = [{
        "y": 300,
        "columns": [
            {"column": schema[i], "x": hx0 + i * 79}
            for i in range(len(schema))
        ],
    }]
    # Values in CYL1..CYL5 and CYL7..CYL10 (CYL6 blank -> 9 values; the CYL6
    # x-slot carries no word).
    vals = {1: 125.0, 2: 133.0, 3: 142.0, 4: 145.0, 5: 134.0,
            7: 127.0, 8: 122.0, 9: 132.0, 10: 129.0}
    row = _row(
        [("LINER", 200), ("WALL", 260), ("temp.", 300), ("fp", 360)]
        + [(f"{v:g}", hx0 + (k - 1) * 79 - 20)
           for k, v in sorted(vals.items())]
    )
    cells = collect_values_tb(row, field, [field], headers, 14, page_rows=[])
    cols = [c.column for c in cells]
    # CYL6 is blank: values map to their true columns, nothing shifts left.
    assert cols == ["CYL1", "CYL2", "CYL3", "CYL4", "CYL5",
                    "CYL7", "CYL8", "CYL9", "CYL10"]
    assert "CYL6" not in cols
