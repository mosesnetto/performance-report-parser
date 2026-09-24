import re

from .normalize import clean

# ------------------------------------------------------------
# Extended multi-word column vocabulary.
#
# The ME performance report uses a few structured header rows made
# of multi-word column labels that the per-word canonical_header()
# cannot recognize (each is split across two or three OCR words, e.g.
# "by" + "FI*rpm", "by" + "TC" + "rpm", and single tokens such as
# "XPERT" / "TORSIOM."). These are the report's own column names, not
# per-field patches, so they are recognized generally by normalizing a
# run of consecutive words and matching the merged token.
#
# Key: uppercase compact form (punctuation/whitespace removed).
# Value: the exact canonical column name used by extract.py schemas.
# ------------------------------------------------------------

_EXTENDED_COLUMNS = {
    "XPERT": "XPERT",
    "TORSIOM": "TORSIOM.",
    "MOP": "MOP",
    "BYFIRPM": "by FI*rpm",
    "BYFLRPM": "by FI*rpm",  # OCR reads FI*rpm as Fl*rpm (lowercase-L)
    "BYPSCAV": "by pscav",
    "BYTCRPM": "by TC rpm",
    "BYMEP": "by MEP",
    "BYRPM": "by rpm",
    "SEAWATERTEMP": "VALUE",  # e.g. "Sea water temp." row label
    "AMBIENTAIRTEMP": "VALUE",    # e.g. "Ambient air temp." row label
    "SHO": "SOG",              # "SHIP SOG" / "SOG" column header
    "STW": "STW",              # "STW" column header (often abbreviated)
    "FWD": "FWD",              # "Draft fwd" column header
    "AFT": "AFT",              # "Draft aft" column header
    "MID": "MID",              # "Draft mid" column header
    "TRIM": "TRIM",            # "Draft trim" column header
}


def _compact(token):
    return re.sub(r"[^A-Z0-9]", "", token.upper())


def _match_extended(words_sorted, i):
    """Try to match 1-3 consecutive words as one extended column label."""
    for n in (3, 2, 1):
        group = words_sorted[i:i + n]
        if len(group) != n:
            continue
        merged = "".join(w.text for w in group)
        key = _compact(merged)
        column = _EXTENDED_COLUMNS.get(key)
        if column is not None:
            cx = sum(float(w.cx) for w in group) / n
            return column, cx, n
    return None, None, 1


HEADER_RE = re.compile(
    r"""
    ^
    (?:
        REF\.?
        |CALC\.?
        |AVG(?:/ENG\.)?
        |MEAS\.?
        |MOP
        |ALL
        |EXPECT\.?
        |MCR
        |CYL\s*\d{1,2}
        |TC\s*\d{1,2}
        |BRG\s*\d{1,2}
        |ENGINE
        |ENGINE/CYL\s*\d+
        |PUMP\s*\d
    )
    $
    """,
    re.I | re.X,
)


def canonical_header(text):
    s = clean(str(text or "")).upper().strip()

    # Remove OCR garbage around common headers.
    s = re.sub(r"[^A-Z0-9/ .]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    compact = s.replace(" ", "")

    # Strip trailing dots that OCR commonly appends to abbreviations
    # (REF., CALC., AVG., ENG., EXPECT., MEAS.) so alias lookup works.
    compact = compact.rstrip(".")

    # Known OCR corruption corrections.
    _ocr_fix = {
        "CCYL1": "CYL1",
        "CCYL": "CYL",
        "T1": "TC1",
        "BRGS": "BRG5",
        "BRGY": "BRG9",
    }
    if compact in _ocr_fix:
        compact = _ocr_fix[compact]

    # Common OCR corrections.
    if compact.startswith("CYL") and compact[3:].isdigit():
        return f"CYL{int(compact[3:])}"

    if compact.startswith("TC") and compact[2:].isdigit():
        return f"TC{int(compact[2:])}"

    if compact.startswith("BRG") and compact[3:].isdigit():
        return f"BRG{int(compact[3:])}"

    if compact.startswith("PUMP") and compact[4:].isdigit():
        return f"PUMP{int(compact[4:])}"

    aliases = {
        "REF": "REF",
        "CALC": "CALC",
        "AVG": "AVG",
        "AVG/ENG": "AVG/ENG",
        "AVG./ENG.": "AVG/ENG",
        "MEAS": "MEAS",
        "MOP": "MOP",
        "ALL": "ALL",
        "EXPECT": "EXPECT",
        "ENGINE": "ENGINE",
        "MCR": "MCR",
        "VALUE": "VALUE",
        "SOG": "SOG",
        "STW": "STW",
        "FWD": "FWD",
        "AFT": "AFT",
        "MID": "MID",
        "TRIM": "TRIM",
        "TEMP": "VALUE",
        "DEG": "VALUE",
        "CEL": "VALUE",
        "RPM": "rpm",
        "KNOT": "kn",
        "METRE": "m",
        "BAR": "bar",
        "BARG": "barG",
        "BARA": "barA",
    }

    return aliases.get(compact)


def detect_headers(rows):
    """
    Return header records in the same structure expected by extract.py.

    Each record contains:
        {
            "y": float,
            "columns": [
                {
                    "column": "CYL1",
                    "x": float,
                },
                ...
            ]
        }
    """

    headers = []

    for row in rows:
        words_sorted = sorted(row.words, key=lambda w: w.left)
        tokens = []

        i = 0
        while i < len(words_sorted):
            word = words_sorted[i]
            header = canonical_header(word.text)

            # Handle split tokens: "CYL" followed by "10", "BRG" + "11", etc.
            # Check the raw uppercase text in case canonical_header returned None
            # for a bare prefix like "CYL" without a number.
            raw_upper = re.sub(r"[^A-Z0-9]", "", word.text.upper())
            if raw_upper in ("CYL", "TC", "BRG", "PUMP") and i + 1 < len(words_sorted):
                next_word = words_sorted[i + 1]
                next_text = next_word.text.strip()
                if next_text.isdigit():
                    header = f"{raw_upper}{int(next_text)}"
                    x = (float(word.cx) + float(next_word.cx)) / 2.0
                    tokens.append({"column": header, "x": x})
                    i += 2
                    continue

            # Handle "MCR" followed by "%" -> "MCR_%"
            if raw_upper == "MCR" and i + 1 < len(words_sorted):
                next_word = words_sorted[i + 1]
                next_raw = re.sub(r"[^A-Z0-9%]", "", next_word.text.upper())
                if next_raw == "%":
                    header = "MCR_%"
                    x = (float(word.cx) + float(next_word.cx)) / 2.0
                    tokens.append({"column": header, "x": x})
                    i += 2
                    continue

            if not header:

                # Generalized multi-word column headers (e.g. XPERT,
                # TORSIOM., "by FI*rpm", "by TC rpm", "by MEP", "by rpm").
                extended, ext_x, consumed = _match_extended(
                    words_sorted,
                    i,
                )

                if extended:
                    tokens.append(
                        {
                            "column": extended,
                            "x": ext_x,
                        }
                    )
                    i += consumed
                    continue

                i += 1
                continue

            tokens.append(
                {
                    "column": header,
                    "x": float(word.cx),
                }
            )
            i += 1

        if len(tokens) < 2:
            continue

        tokens.sort(key=lambda item: item["x"])

        headers.append(
            {
                "y": float(row.y),
                "columns": tokens,
            }
        )

    return headers


def nearest_header(x, header_columns):
    """
    Return the semantic column nearest to x.

    Accepts:
        [{"column": "CYL1", "x": 100}, ...]
    or:
        [("CYL1", 100), ...]
    or:
        ["CYL1", ...]  -> cannot spatially map
    """

    if not header_columns:
        return None

    candidates = []

    for item in header_columns:

        if isinstance(item, dict):
            name = item.get("column")
            hx = item.get("x")

        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            hx, name = item[0], item[1]

        else:
            continue

        if name is None or hx is None:
            continue

        try:
            distance = abs(float(hx) - float(x))
        except Exception:
            continue

        candidates.append((distance, name))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    return candidates[0][1]