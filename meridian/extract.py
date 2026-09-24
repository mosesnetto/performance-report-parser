import re

from .headers import nearest_header
from .models import Cell, FieldRecord
from .normalize import clean

# ============================================================
# NUMBER PARSING
# ============================================================

NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)$"
)


# Conservative OCR-debris matcher: a clean numeric prefix followed only by
# non-word, non-percent punctuation/symbols (e.g. "/@"). Fails on any
# alphabetic or digit tail, so "RPM84", "PUMP50", "CYL10", "abc123",
# "stable" are never accepted.
_OCR_DEBRIS_PREFIX_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+))[^\w%]*$"
)


# A clean numeric prefix followed by OCR noise, a "%" unit, and optional
# trailing OCR noise ("32.0%", "32.0|/%"). Used only to recover a value for
# %-declared fields, where the trailing "%" is the field's own unit.
_PERCENT_SUFFIX_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+))[^\w]*%[^\w]*$"
)


# Date in the CE form "M/D/YYYY" or "M-D-YYYY" (a single OCR word).
_CE_DATE_MD = re.compile(
    r"(?:^|[^\d])(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})(?:[^\d]|$)"
)


# An unambiguous colon-joined clock time "HH:MM" (or "H:MM") in one OCR word.
_CE_TIME = re.compile(
    r"(\d{1,2}):(\d{2})"
)


# A clock-time-only OCR token (digits, separators and OCR noise), used to
# recognise the structurally-associated time that follows a date on the same
# logical row without touching letters of neighbouring fields.
_CE_TIME_TOKEN = re.compile(
    r"^[0-9:.\[\]()|/@\\\\\-]*$"
)


# An OCR-split clock-time hour fragment: a bare "HH:" (or "H:") token with a
# trailing colon but no minutes yet, e.g. "21:" / "08:".
_CE_HOUR_FRAG = re.compile(
    r"^(\d{1,2}):$"
)


# The matching minute fragment of an OCR-split clock time: a bare two-digit
# token (no dots, no letters), e.g. "40" / "50".  A dotted measurement such as
# "19.90", "20.60" or "25.0" never matches, so it can never be (re)used as a
# time.
_CE_MIN_FRAG = re.compile(
    r"^(\d{2})$"
)


UNIT_TOKENS = {
    "%",
    "rpm",
    "kw",
    "kW",
    "bar",
    "barG",
    "barA",
    "°C",
    "°c",
    "g/kWh",
    "kg/s",
    "kg/m³",
    "kg/m3",
    "mmWG",
    "mm",
    "m",
    "m³/h",
    "m3/h",
    "cSt",
    "kg/kg",
    "USD/t",
    "h",
    "°CA",
    "l/h",
}


DISTINGUISHING_WORDS = {
    "aft",
    "fp",
    "fwd",
    "exh",
}


# ============================================================
# KNOWN MULTI-COLUMN FIELDS
# ============================================================

COLUMN_SCHEMAS = {
    "engine power estimated": [
        "XPERT",
        "TORSIOM.",
        "MOP",
        "by FI*rpm",
        "by pscav",
        "by TC rpm",
        "by MEP",
        "by rpm",
    ],

    "hydr oil pump swash plate pos": [
        "PUMP2",
        "PUMP3",
        "PUMP4",
        "PUMP5",
    ],

    "water press sac in": [
        "AVG",
        "TC1",
        "TC2",
        "TC3",
    ],

    "lub oil temp tc in": [
        "AVG",
        "TC1",
        "TC2",
        "TC3",
    ],

    "lub oil temp tc out": [
        "AVG",
        "TC1",
        "TC2",
        "TC3",
    ],

    "liner wall temp aft": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "liner wall temp fp": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "liner wall temp fwd": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "liner wall temp exh": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "cool water temp cyl in": [
        "AVG",
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "cool water temp cyl out": [
        "AVG",
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "main bearing temp": [
        "AVG",
        "BRG1",
        "BRG2",
        "BRG3",
        "BRG4",
        "BRG5",
        "BRG6",
        "BRG7",
        "BRG8",
        "BRG9",
        "BRG10",
        "BRG11",
        "BRG12",
        "BRG13",
    ],
}


EXPLICIT_COLUMNS = {

    # ENGINE POWER ESTIMATED
    #
    # Actual table structure:
    #
    # XPERT | TORSIOM. | MOP | by FI*rpm |
    # by pscav | by TC rpm | by MEP | by rpm
    #

    "engine power estimated": [
        "XPERT",
        "TORSIOM.",
        "MOP",
        "by FI*rpm",
        "by pscav",
        "by TC rpm",
        "by MEP",
        "by rpm",
    ],

    # ENGINE SPEED
    #
    # Row: "ENGINE speed rpm" followed by three values that map
    # positionally (left to right) to:
    #
    #     MCR | MCR_% | ENGINE

    "engine speed": [
        "MCR",
        "MCR_%",
        "ENGINE",
    ],

    # ENGINE POWER EFFECTIVE
    #
    # Row: "ENGINE power effective kW" followed by three values:
    #     MCR | MCR_% | ENGINE

    "engine power effective": [
        "MCR",
        "MCR_%",
        "ENGINE",
    ],

    # HYDRAULIC PUMP

    "hydr oil pump swash plate pos": [
        "PUMP2",
        "PUMP3",
        "PUMP4",
        "PUMP5",
    ],

    # TURBOCHARGER

    "water press sac in": [
        "AVG",
        "TC1",
        "TC2",
        "TC3",
    ],

    "lub oil temp tc in": [
        "AVG",
        "TC1",
        "TC2",
        "TC3",
    ],

    "lub oil temp tc out": [
        "AVG",
        "TC1",
        "TC2",
        "TC3",
    ],

    # LINER WALL

    "liner wall temp aft": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "liner wall temp fp": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "liner wall temp fwd": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "liner wall temp exh": [
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    # CYLINDER TABLES

    "cool water temp cyl in": [
        "AVG",
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    "cool water temp cyl out": [
        "AVG",
        "CYL1",
        "CYL2",
        "CYL3",
        "CYL4",
        "CYL5",
        "CYL6",
        "CYL7",
        "CYL8",
        "CYL9",
        "CYL10",
    ],

    # BEARINGS

    "main bearing temp": [
        "AVG",
        "BRG1",
        "BRG2",
        "BRG3",
        "BRG4",
        "BRG5",
        "BRG6",
        "BRG7",
        "BRG8",
        "BRG9",
        "BRG10",
        "BRG11",
        "BRG12",
        "BRG13",
    ],

    # ELECTRONIC CONTROL
    #
    # Table structure on page 13:
    # REF. | CALC. | ALL
    # Fuel index ECU % | 67.4 | 67.4 | 63.0
    # Applied fuel quality offset % | 0.0 | 0.0 | 0.0
    # Running mode | Economy | RPM | ISO CORRECTED
    # Governor mode | RPM | | RECALL

    "fuel index ecu": [
        "REF",
        "CALC",
        "ALL",
    ],

    "applied fuel quality offset": [
        "REF",
        "CALC",
        "ALL",
    ],

    "running mode": [
        "REF",
        "CALC",
        "ALL",
    ],

    "governor mode": [
        "REF",
        "CALC",
        "ALL",
    ],
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    s = str(text or "").lower()

    s = s.replace("®c", " ")
    s = s.replace("®", " ")
    s = s.replace("°e", " ")
    s = s.replace("°c", " ")
    s = s.replace("°", " ")

    s = s.replace(".", " ")
    s = s.replace(":", " ")
    s = s.replace(",", " ")

    # OCR corruption: "viv" read instead of "vlv" (valve).
    s = re.sub(r"\bviv\b", "vlv", s)

    s = re.sub(
        r"\b\d+(?:\.\d+)?\b",
        " ",
        s,
    )

    s = re.sub(
        r"[^a-z0-9*@/\-\s]+",
        " ",
        s,
    )

    return " ".join(
        s.split()
    )


def field_key(text):
    s = normalize_text(
        text
    )

    s = s.replace(
        "*",
        "",
    )

    s = s.replace(
        "/",
        " ",
    )

    s = re.sub(
        r"\s+",
        " ",
        s,
    )

    return s.strip()


def label_tokens(text):
    return normalize_text(
        text
    ).split()


# ============================================================
# EXPLICIT COLUMN HELPERS
# ============================================================

def get_explicit_columns(field):
    """
    Return the semantic columns for fields that are known to be
    columnar in the ME performance report.

    Priority:
        1. Explicit EXPLICIT_COLUMNS entry.
        2. Field-level YAML `columns` when present.
        3. Known report-region schema.
        4. None for scalar/unknown layouts.
    """

    name = field.get(
        "name",
        "",
    )

    key = field_key(name)

    # ------------------------------------------------------------
    # 1. Existing explicit schema.
    # ------------------------------------------------------------

    explicit = EXPLICIT_COLUMNS.get(key)

    if explicit:
        return explicit

    # ------------------------------------------------------------
    # 2. YAML-provided columns.
    # ------------------------------------------------------------

    configured = field.get("columns")

    if isinstance(configured, list) and configured:
        return [
            str(column)
            for column in configured
            if str(column).strip()
        ]

    # ------------------------------------------------------------
    # 3. Field ID specific schemas.
    # ------------------------------------------------------------

    field_id = str(
        field.get("id", "")
    ).upper()

    # Cylinder-pressure rows.
    if field_id in {
        "C001",
        "C003",
        "C004",
        "C011",
    }:
        return [
            "REF",
            "CALC",
            "AVG",
            *[
                f"CYL{i}"
                for i in range(1, 11)
            ],
        ]

    if field_id in {
        "C002",
        "C005",
        "C008",
        "C009",
        "C010",
    }:
        return [
            f"CYL{i}"
            for i in range(1, 11)
        ]

    # Cylinder pressure rows with REF, CALC, AVG, CYL1-10
    if field_id in {
        "C001",
        "C003",
        "C004",
        "C006",
        "C007",
        "C011",
        "C012",
        "C013",
    }:
        return [
            "REF",
            "CALC",
            "AVG",
            *[f"CYL{i}" for i in range(1, 11)],
        ]

    # Crank pin bearing and X-head bearing.
    if field_id in {
        "B003",
        "B004",
        "B005",
        "B006",
    }:
        return [
            "AVG",
            *[
                f"CYL{i}"
                for i in range(1, 11)
            ],
        ]

    # Liner wall rows.
    if field_id in {
        "LW001",
        "LW002",
        "LW003",
        "LW004",
    }:
        return [
            f"CYL{i}"
            for i in range(1, 11)
        ]

    # Liner wall AVG (single value)
    if field_id == "LW005":
        return ["AVG"]

    # Cylinder-condition rows.
    if field_id in {
        "CC001",
        "CC002",
    }:
        return [
            "AVG",
            *[
                f"CYL{i}"
                for i in range(1, 11)
            ],
        ]

    # Turbocharger.
    if field_id in {
        "T002",
        "T003",
        "T006",
        "T008",
    }:
        return [
            "REF",
            "CALC",
            "AVG",
            "TC1",
            "TC2",
            "TC3",
        ]

    if field_id == "T007":
        return ["AVG"]

    # Scavenge air.
    if field_id in {
        "S002",
        "S003",
        "S006",
        "S007",
        "S008",
    }:
        return [
            "REF",
            "CALC",
            "AVG",
            "TC1",
            "TC2",
            "TC3",
        ]

    if field_id == "S011":
        return ["REF", "CALC", "AVG", "TC1"]

    if field_id == "S012":
        return ["REF", "CALC", "AVG", "TC2"]

    # Exhaust gas cylinder rows.
    if field_id == "X009":
        return ["REF", "CALC", "AVG", "CYL1", "CYL2", "CYL3", "CYL4", "CYL5"]

    # Exhaust gas cylinder rows.
    if field_id in {
        "X001",
        "X002",
    }:
        return [
            "REF",
            "CALC",
            "AVG",
            *[f"CYL{i}" for i in range(1, 11)],
        ]

    # Exhaust gas TC rows (turbocharger)
    if field_id in {
        "X005",
        "X006",
        "X007",
        "X008",
    }:
        return [
            "REF",
            "CALC",
            "AVG",
            "TC1",
            "TC2",
            "TC3",
        ]

    # Fuel oil.
    if field_id == "F002":
        return ["REF", "CALC", "MEAS"]

    if field_id in {
        "F005",
        "F008",
    }:
        return ["REF", "CALC"]

    if field_id == "F012":
        return ["REF", "CALC", "EXPECT"]

    # Scavenge air receiver.
    if field_id == "S004":
        return ["REF", "CALC"]

    # Engine-media hydraulic pump.
    if field_id == "EM016":
        return [
            "PUMP2",
            "PUMP3",
            "PUMP4",
            "PUMP5",
        ]

    # ------------------------------------------------------------
    # 4. Name-based schemas for cases where the IDs may change.
    # ------------------------------------------------------------

    if (
        "crank pin bear" in key
        or "x-head bear" in key
    ):
        return [
            "AVG",
            *[
                f"CYL{i}"
                for i in range(1, 11)
            ],
        ]

    if (
        "liner wall temp" in key
    ):
        return [
            f"CYL{i}"
            for i in range(1, 11)
        ]

    if (
        "firing press" in key
        or "pmax deviation" in key
        or "pmax offset" in key
        or "compression press" in key
        or "pcomp deviation" in key
        or "pcomp offset" in key
        or "power indicated" in key
    ):
        return [
            "REF",
            "CALC",
            "AVG",
            *[
                f"CYL{i}"
                for i in range(1, 11)
            ],
        ]

    if (
        key.startswith("exh gas temp cyl")
        or key.startswith("exh gas pressure cyl")
    ):
        return [
            "REF",
            "AVG",
            *[
                f"CYL{i}"
                for i in range(1, 11)
            ],
        ]

    if (
        "water press sac" in key
        or "lub oil temp tc" in key
    ):
        return [
            "AVG",
            "TC1",
            "TC2",
            "TC3",
        ]

    if (
        "main bearing temp" in key
    ):
        return [
            "AVG",
            *[
                f"BRG{i}"
                for i in range(1, 14)
            ],
        ]

    return None


def is_multi_column_field(field):
    return bool(
        get_explicit_columns(
            field
        )
    )


# Known paired scalar fields (SOG/STW, FWD/AFT, MID/TRIM) hold exactly two
# named values; they are not single-value scalars, so Rule B / same-value
# dedup must not collapse them.
_PAIRED_COLUMN_KEYS = {
    "ship sog stw",
    "draft fwd aft",
    "draft mid trim",
}


def _field_is_scalar(field):
    """
    True when the field resolves to a single VALUE slot (no explicit
    multi-column schema, no region-default columns, no known pair). Rule B
    (scalar same-value dedup) is restricted to these fields so that
    genuinely columnar tables (e.g. X002, C008, B001, P001) are untouched.
    """
    if is_multi_column_field(field):
        return False

    key = field_key(
        field.get(
            "name",
            "",
        )
    )

    if key in _PAIRED_COLUMN_KEYS:
        return False

    if default_columns_for_region(
        field.get(
            "region",
            "",
        ),
        field.get(
            "name",
            "",
        ),
    ):
        return False

    return True


# ============================================================
# FIELD ROW MATCHING
# ============================================================

def find_field_row(
    rows,
    field,
):
    target = normalize_text(
        field["name"]
    )

    target_tokens = target.split()

    if not target_tokens:
        return None, 0.0

    target_special = (
        set(target_tokens)
        & DISTINGUISHING_WORDS
    )

    target_is_avg = (
        "avg" in target_tokens
    )

    # ------------------------------------------------------------
    # anchor_after: this field's label is on a row *after* the row
    # containing the anchor label (e.g. "Temp. deviation CRANK PIN"
    # appears below the "CRANK PIN BEAR. temp" row). Find the anchor
    # row first, then restrict the search to rows below it.
    # ------------------------------------------------------------

    anchor = field.get(
        "anchor_after"
    )

    anchor_min_y = None

    if anchor:

        anchor_tokens = normalize_text(
            anchor
        ).split()

        for row in sorted(
            rows,
            key=lambda r: r.y,
        ):

            candidate_tokens = normalize_text(
                row.text
            ).split()

            valid = True

            search_start = 0

            for token in anchor_tokens:

                try:

                    pos = candidate_tokens.index(
                        token,
                        search_start,
                    )

                except ValueError:

                    valid = False
                    break

                search_start = pos + 1

            if valid:
                anchor_min_y = (
                    float(row.y) + 1.0
                )
                break

        # The anchor row already identifies this field's location. The
        # field name repeats words from the anchor (e.g. "Temp. deviation
        # CRANK PIN" after the "CRANK PIN BEAR. temp" row), but those words
        # are not present on the deviation row itself. Drop any target token
        # that also appears in the anchor so the remainder ("temp deviation")
        # matches the labelled row below the anchor.
        anchor_set = set(anchor_tokens)
        target_tokens = [
            t for t in target_tokens
            if t not in anchor_set
        ]

        target_special = (
            set(target_tokens)
            & DISTINGUISHING_WORDS
        )

        target_is_avg = (
            "avg" in target_tokens
        )

    def _search(
        target_tokens,
        target_special,
        target_is_avg,
        anchor_min_y,
    ):

        candidates = []

        for row in rows:

            if (
                anchor_min_y is not None
                and float(row.y) < anchor_min_y
            ):
                continue

            candidate = normalize_text(
                row.text
            )

            candidate_tokens = candidate.split()

            if not candidate_tokens:
                continue

            candidate_special = (
                set(candidate_tokens)
                & DISTINGUISHING_WORDS
            )

            positions = []

            search_start = 0

            valid = True

            for token in target_tokens:

                try:

                    pos = candidate_tokens.index(
                        token,
                        search_start,
                    )

                except ValueError:

                    # Exact match required - no substring fallback.
                    # OCR word mergers (e.g. "Seawater" → "sea" + "water")
                    # are handled by the token ordering and gap penalties
                    # below, not by allowing fields to match each other's rows.
                    valid = False
                    break

                positions.append(
                    pos
                )

                search_start = (
                    pos + 1
                )

            if not valid:
                continue

            label_span = (
                positions[-1]
                - positions[0]
                + 1
            )

            # Distinguished rows must match the same qualifier.
            #
            # A distinguishing word (fwd/aft/fp/exh) only disqualifies this
            # candidate when it sits on the same physical label as the target.
            # On packed rows an unrelated adjacent field (e.g. "Draft fwd | aft")
            # may contribute a distinguishing word far to the right of the target
            # label, so a generic target must still be allowed to match.
            if target_special:

                if not target_special.issubset(
                    candidate_special
                ):
                    continue

            else:

                if candidate_special:

                    # Check whether a distinguishing word occurs adjacent to the
                    # target label span (i.e. it belongs to the same label). Only
                    # then is this a competing, more-specific field to reject.
                    last_label_idx = positions[-1]

                    adjacent = any(
                        abs(didx - last_label_idx) <= 1
                        or
                        abs(didx - positions[0]) <= 1
                        for didx in range(len(candidate_tokens))
                        if candidate_tokens[didx] in candidate_special
                    )

                    if adjacent:
                        continue

            # AVG field must contain AVG.

            if target_is_avg:

                if "avg" not in candidate_tokens:
                    continue

            span = label_span

            gaps = (
                span -
                len(target_tokens)
            )

            score = 1.0

            score -= min(
                0.20,
                gaps * 0.03,
            )

            extra = max(
                0,
                len(candidate_tokens)
                - len(target_tokens),
            )

            score -= min(
                0.15,
                extra * 0.02,
            )

            if (
                span ==
                len(target_tokens)
            ):
                score = 1.0

            candidates.append(
                (
                    score,
                    row,
                )
            )

        if not candidates:
            return None, 0.0

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        score, row = candidates[0]

        if score < 0.60:
            return None, score

        return row, score

    # Try the anchor-restricted search first. If it finds nothing (e.g. the
    # anchor and the label share the same physical row), fall back to the
    # original unanchored search so previously-working fields do not regress.
    if anchor_min_y is not None:

        row, score = _search(
            target_tokens,
            target_special,
            target_is_avg,
            anchor_min_y,
        )

        if row is not None:
            return row, score

        orig_tokens = normalize_text(
            field["name"]
        ).split()

        return _search(
            orig_tokens,
            set(orig_tokens) & DISTINGUISHING_WORDS,
            "avg" in orig_tokens,
            None,
        )

    return _search(
        target_tokens,
        target_special,
        target_is_avg,
        None,
    )


# ============================================================
# NUMBER PARSING
# ============================================================

def _strip_ocr_trailing(text):
    raw = str(
        text or ""
    ).strip()

    if not raw:
        return raw

    while raw and raw[-1] in ";,|)]}@®~`^-=><":

        raw = raw[:-1]

    return raw


def _strip_ocr_leading(text):
    raw = str(
        text or ""
    ).strip()

    if not raw:
        return raw

    while raw and raw[0] in "~`^=><":

        raw = raw[1:]

    return raw


def _clean_ocr_artifacts(text):
    raw = str(
        text or ""
    ).strip()

    if not raw:
        return raw

    raw = _strip_ocr_leading(raw)
    raw = _strip_ocr_trailing(raw)

    return raw


def _parse_num_internal(text):
    """
    Context-free number parser. Returns ``(value, debris_cleaned)`` where
    ``debris_cleaned`` is True only when the token required the conservative
    OCR-debris suffix strip to become numeric (e.g. ``"21630/@"`` -> ``21630``
    is debris-cleaned; ``"40"`` is not).

    Rule A (time/date debris exclusion): a debris-cleaned token whose
    discarded suffix contains a colon ``:`` is rejected, because a leading
    number followed by ``:`` is a time/date fragment (``21:``, ``2025:``),
    not a measurement. This never rejects genuine numeric debris such as
    ``21630/@``, ``20.60)``, ``594.0)``, ``115.2@`` or ``-0.02/@``.
    """
    raw = str(
        text or ""
    ).strip()

    if not raw:
        return None, False

    raw = raw.strip(
        "()[]{}"
    )

    raw = _clean_ocr_artifacts(raw)

    raw = raw.replace(
        ",",
        ".",
    )

    if NUMBER_RE.fullmatch(
        raw
    ):
        try:
            return float(raw), False
        except ValueError:
            return None, False

    # ----------------------------------------------------
    # Conservative OCR-debris cleanup.
    #
    # OCR occasionally merges a numeric value with an abutting
    # non-numeric glyph (e.g. "21630/@"). Such tokens fail the
    # strict full-match above and were silently dropped, with the
    # value shifting the whole column mapping. Accept the token
    # only when a clean numeric PREFIX is followed exclusively by
    # obvious OCR punctuation/symbols (no letters, digits or "%").
    # This intentionally never accepts alphabetic suffixes such as
    # "abc123", "RPM84", "PUMP50", "CYL10" or "stable".
    # ----------------------------------------------------

    debris = _OCR_DEBRIS_PREFIX_RE.match(
        raw
    )

    if not debris:
        return None, False

    numeric = debris.group(1)

    discarded = raw[len(numeric):]

    # Rule A: colon-based time/date debris is not a measurement.
    if ":" in discarded:
        return None, False

    try:
        value = float(numeric)
    except ValueError:
        return None, False

    return value, True


def parse_num(text):
    """
    Public context-free number parser. Returns a float (or None). Debris
    status is preserved for callers that need it via ``_parse_num_internal``
    / ``parse_numeric_ocr``.
    """
    value, _ = _parse_num_internal(
        text
    )

    return value


def parse_numeric_ocr(
    word,
    field,
):
    raw = str(
        getattr(
            word,
            "text",
            "",
        )
    ).strip()

    raw = _clean_ocr_artifacts(raw)

    unit = str(
        field.get("unit", "")
    ).strip()

    # A value for a %-declared field legitimately ends with a "%" unit that
    # may be glued to OCR noise (e.g. "32.0%", "32.0|/%"). The generic debris
    # matcher deliberately rejects any "%" tail so percentages never leak into
    # non-% fields. For a %-unit field the "%" is the field's own unit, so
    # strip it and parse the cleaned numeric. Non-% fields keep the strict
    # behaviour (so a stray "%" never fabricates a value).
    if unit == "%":

        percent_tail = _PERCENT_SUFFIX_RE.match(
            raw
        )

        if percent_tail:

            raw = percent_tail.group(
                1
            )

    value, debris_cleaned = _parse_num_internal(
        raw
    )

    try:
        word.debris_cleaned = bool(
            debris_cleaned
        )
    except Exception:
        pass

    if value is None:
        return None

    key = field_key(
        field.get(
            "name",
            "",
        )
    )

    # --------------------------------------------------------
    # PERCENTAGE POSITION GAUGES (OCR decimal-point loss)
    #
    # A 0-100 percentage position gauge (hydraulic-pump swash plate,
    # exhaust wastegate position) is printed with one decimal place. OCR
    # frequently drops that decimal, reading "50.0" as "500" and "59.4" as
    # "594". Recover the decimal ONLY for such a gauge when the raw integer
    # is outside the valid 0-100 range and a tens/hundreds shift lands it
    # inside it. This mirrors the source's fixed one-decimal convention and
    # never touches non-position percentage fields (fuel index, fuel offset,
    # sulfur/water content), which carry no such convention.
    # --------------------------------------------------------

    is_position_gauge = (
        "pos" in key
    )

    if (
        unit == "%"
        and is_position_gauge
        and value > 100
    ):

        if value > 1000:

            value /= 100.0

        else:

            value /= 10.0

        if 0 <= value <= 100:

            return value

        return None

    return value


# ============================================================
# NUMERIC OCR WORDS
# ============================================================

def _match_tokens_to_words(
    normalized_words,
    tokens,
):
    """
    Match an ordered token sequence against per-word normalized token
    streams, returning a list of matched word-index position lists (one per
    distinct start word), or an empty list when no match exists.

    Unlike a plain per-word equality check, a single OCR word such as
    "spec.air", "f.r.@current", "consumpt.*" or "Exh.wastegate" normalizes
    to MULTIPLE tokens (normalize_text replaces '.'/'@'/'*' with a space).
    The whole-row find_field_row() path already handles this because it
    normalizes the entire row text; this stream match keeps find_value_words
    consistent so labels containing such words are still recognised.

    A bounded window (default 6 words) is kept so unrelated words cannot
    extend a match arbitrarily.
    """

    if not tokens:
        return []

    word_tokens = [
        nw.split()
        for nw in normalized_words
    ]

    n_words = len(word_tokens)

    all_matches = []

    for start_index in range(n_words):

        positions = []

        wi = start_index
        ti = 0

        ok = True

        for token in tokens:

            found = False

            wj = wi
            tj = ti
            steps = 0

            while wj < n_words and steps <= 6:

                wt = word_tokens[wj]

                if (
                    tj < len(wt)
                    and wt[tj] == token
                ):
                    found = True
                    break

                tj += 1

                if tj >= len(wt):

                    wj += 1
                    tj = 0
                    steps += 1

            if not found:

                ok = False
                break

            positions.append(wj)

            wi = wj
            ti = tj + 1

            if ti >= len(word_tokens[wi]):

                wi += 1
                ti = 0

        if ok and len(positions) == len(tokens):
            all_matches.append(positions)

    return all_matches


def _match_tokens_to_words_fuzzy(
    normalized_words,
    tokens,
):
    """
    Like `_match_tokens_to_words` but tolerant of a single corrupt/absent
    label token. OCR often drops or mangles one character inside a word
    (e.g. "f.r.@current" -> "f..@current"), which makes the strict per-token
    match fail. This is used only for next-field (right-bound) detection so
    that a mangled neighbouring label does not leave the current field
    unbounded.

    Returns a list of start word indices (or empty) for labels that match
    with at most one token missing.
    """

    if not tokens:
        return []

    word_tokens = [
        nw.split()
        for nw in normalized_words
    ]

    n_words = len(word_tokens)

    starts = []

    for start_index in range(n_words):

        wi = start_index
        ti = 0

        misses = 0
        positions = []

        for token in tokens:

            found = False

            wj = wi
            tj = ti
            steps = 0

            while wj < n_words and steps <= 6:

                wt = word_tokens[wj]

                if (
                    tj < len(wt)
                    and wt[tj] == token
                ):
                    found = True
                    break

                tj += 1

                if tj >= len(wt):

                    wj += 1
                    tj = 0
                    steps += 1

            if not found:

                misses += 1

                if misses > 1:
                    break

                continue

            positions.append(wj)

            wi = wj
            ti = tj + 1

            if ti >= len(word_tokens[wi]):

                wi += 1
                ti = 0

        if misses <= 1 and positions:

            # A neighboring FIELD LABEL cannot begin at an OCR debris token
            # that carries no alphabetic character (a bare "@", digit-only
            # text, "|®", "-15/0", ...).  Such tokens are value-row artifacts,
            # not label starts; anchoring a next-field boundary on them would
            # truncate an otherwise real value row (e.g. B002 "Temp.
            # deviation").  Because a token may be missing, the label can
            # start at an alphabetic word and yet land on a debris token --
            # reject the reported start unless it is itself alphabetic.
            if positions[0] < n_words and not any(
                re.search(r"[A-Za-z]", tok)
                for tok in word_tokens[positions[0]]
            ):
                continue

            starts.append(positions[0])

    return starts


def _compute_value_bounds(
    row,
    field,
    region_fields=None,
):
    """
    Determine the horizontal segment (label_end .. next_field_left) that
    bounds the numeric values belonging to `field` on `row`.

    Returns (label_end, next_field_left); label_end is None when the
    field's label cannot be located on the row.
    """

    words = sorted(
        row.words,
        key=lambda word: word.left,
    )

    if not words:
        return None, None

    field_name = str(
        field.get("name", "")
    ).strip()

    target_tokens = label_tokens(
        field_name
    )

    # Mirror find_field_row(): for a field declared with `anchor_after`, the
    # distinguishing words that repeat the anchor (e.g. "CRANK PIN" / "X-HEAD"
    # in "Temp. deviation CRANK PIN") live on the anchor row, NOT on the value
    # row that this field actually reports from. Drop the anchor tokens so the
    # remaining label (e.g. "temp deviation") can be located on the value row;
    # otherwise value bounds become (None, None, None) and every value is lost.
    anchor = field.get(
        "anchor_after"
    )

    if anchor:

        anchor_tokens = label_tokens(
            anchor
        )

        anchor_set = set(
            anchor_tokens
        )

        target_tokens = [
            t
            for t in target_tokens
            if t not in anchor_set
        ]

    if not target_tokens:
        return None, None, None

    normalized_words = [
        normalize_text(word.text)
        for word in words
    ]

    label_matches = _match_tokens_to_words(
        normalized_words,
        target_tokens,
    )

    if not label_matches:
        return None, None, None

    label_positions = min(
        label_matches,
        key=lambda p: p[0],
    )

    label_start_index = label_positions[0]
    label_end_index = label_positions[-1]

    label_start = words[
        label_start_index
    ].left

    label_end = max(
        words[i].right
        for i in range(
            label_start_index,
            label_end_index + 1,
        )
    )

    # ------------------------------------------------------------
    # Find the next field label to the right.
    # ------------------------------------------------------------

    next_field_left = None

    if region_fields:

        next_positions = []

        current_id = field.get(
            "id"
        )

        for other in region_fields:

            if other.get("id") == current_id:
                continue

            other_name = str(
                other.get("name", "")
            ).strip()

            other_tokens = label_tokens(
                other_name
            )

            if not other_tokens:
                continue

            all_matches = _match_tokens_to_words(
                normalized_words,
                other_tokens,
            )

            for positions in all_matches:

                other_left = words[
                    positions[0]
                ].left

                if other_left <= label_end:
                    continue

                next_positions.append(
                    other_left
                )

                break

            if all_matches:

                continue

            for other_start in _match_tokens_to_words_fuzzy(
                normalized_words,
                other_tokens,
            ):

                other_left = words[
                    other_start
                ].left

                if other_left <= label_end:
                    continue

                next_positions.append(
                    other_left
                )

                break

        if next_positions:
            next_field_left = min(
                next_positions
            )

    return label_start, label_end, next_field_left


def _collect_numeric_values(
    row,
    field,
    label_end,
    next_field_left,
    left_bound=None,
):
    """
    Collect the numeric OCR words on `row` that fall inside the given
    horizontal segment. Returns a list of (word, value) tuples.

    Numeric words with `left <= left_bound` are excluded. When `left_bound`
    is not provided it defaults to `label_end` (same-row extraction: values
    lie to the right of the field's label words).
    """

    if left_bound is None:
        left_bound = label_end

    words = sorted(
        row.words,
        key=lambda word: word.left,
    )

    if label_end is None or not words:
        return []

    normalized_words = [
        normalize_text(word.text)
        for word in words
    ]

    values = []

    for idx, (
        word,
        normalized,
    ) in enumerate(
        zip(
            words,
            normalized_words,
        )
    ):

        if word.left <= left_bound:
            continue

        if (
            next_field_left is not None
            and word.left >= next_field_left
        ):
            continue

        raw = str(
            word.text or ""
        ).strip()

        if not raw:
            continue

        # OCR with zero confidence is unreliable noise (e.g. a "2" captured
        # from a "°C" unit glyph in the left margin). Reject it so it cannot
        # shift the column assignment of a multi-column table.
        try:
            if float(word.conf) <= 0.0:
                continue
        except Exception:
            pass

        if raw in UNIT_TOKENS:
            continue

        if normalized in {
            "ref",
            "calc",
            "avg",
            "avg/eng",
            "meas",
            "mop",
            "all",
            "expect",
            "engine",
            "cyl",
            "tc",
            "brg",
            "pump",
        }:
            continue

        if re.fullmatch(
            r"(?:cyl|tc|brg|pump)\s*\d{1,2}",
            normalized,
            re.I,
        ):
            continue

        value = parse_numeric_ocr(
            word,
            field,
        )

        if value is None:
            continue

        # A numeric immediately followed (within a small gap) by a "%" unit
        # token is a percentage measurement, not a reading in this field's own
        # units. For a columnar field whose explicit schema has a single unit
        # (e.g. CYL1..CYL10 of a bar "Mean effective pressure"), such a
        # percentage cannot belong to any of its slots and must not leak in
        # (e.g. a "% of MCR" engine-load value becoming a spurious extra
        # cylinder column). Fields that legitimately carry a percentage column
        # (an explicit schema containing "%", such as MCR_%) are left alone, as
        # are scalar fields without an explicit schema.
        schema_cols = get_explicit_columns(
            field
        )

        if (
            str(field.get("unit", "")).strip() != "%"
            and schema_cols
            and not any(
                "%" in c
                for c in schema_cols
            )
        ):

            next_word = (
                words[idx + 1]
                if idx + 1 < len(words)
                else None
            )

            next_is_percent = False

            if next_word is not None:

                next_raw = str(
                    next_word.text or ""
                ).strip()

                # The trailing "%" unit is often glued to OCR noise (e.g.
                # "%", "%|", "%}"), so reduce the next token to its %-glyphs.
                next_is_percent = (
                    next_raw != ""
                    and \
                    re.sub(r"[^\w%]", "", next_raw) == "%"
                )

            if (
                next_is_percent
                and (
                    next_word.left - word.right
                ) < 80
            ):
                continue

        values.append(
            (
                word,
                value,
            )
        )

    # ------------------------------------------------------------
    # Remove duplicate OCR detections.
    # ------------------------------------------------------------

    deduped = []

    for word, value in values:

        duplicate = False

        for old_word, old_value in deduped:

            if (
                abs(
                    word.cx - old_word.cx
                ) < 30
                and
                abs(
                    word.cy - old_word.cy
                ) < 20
                and
                abs(
                    value - old_value
                ) < 0.0001
            ):
                duplicate = True
                break

        if not duplicate:
            deduped.append(
                (
                    word,
                    value,
                )
            )

    # ------------------------------------------------------------
    # Rule B (scalar same-value dedup).
    #
    # A scalar field normally yields ONE value, so a debris-cleaned variant
    # of a value that is ALSO present cleanly in the same field is an OCR
    # double detection (e.g. E003 "Fuel index ECU": clean `67.4` at conf .96
    # plus a debris-cleaned `67.4` at conf .30). In that case drop the
    # debris-cleaned duplicate and keep the clean one.
    #
    # Crucially, a scalar field can still legitimately hold MULTIPLE CLEAN
    # readings of the same value on one row (e.g. X009 "Exh.wastegate (EGB)
    # pos.": `0 0 1` -- two distinct clean zero positions). Those are NOT
    # collapsed; only a debris-cleaned token colliding with a CLEAN
    # same-value token is removed.
    # ------------------------------------------------------------

    if _field_is_scalar(
        field
    ):

        by_value = {}

        for word, value in deduped:

            key = round(
                value,
                6,
            )

            entry = by_value.setdefault(
                key,
                {
                    "clean": [],
                    "debris": [],
                },
            )

            if getattr(
                word,
                "debris_cleaned",
                False,
            ):
                entry["debris"].append(
                    (
                        word,
                        value,
                    )
                )
            else:
                entry["clean"].append(
                    (
                        word,
                        value,
                    )
                )

        rebuilt = []

        for entry in by_value.values():

            if (
                entry["debris"]
                and entry["clean"]
            ):

                # Debris is a corrupted double-detection of a clean value:
                # keep the clean occurrences, discard the debris ones.
                rebuilt.extend(
                    entry["clean"]
                )
            else:

                # No clean/debris collision: keep everything (genuine
                # multi-readings, or all-clean, or all-debris).
                rebuilt.extend(
                    entry["clean"]
                )

                rebuilt.extend(
                    entry["debris"]
                )

        deduped = rebuilt

    return deduped


def _collect_text_value(
    row,
    field,
    region_fields=None,
    label_row=None,
):
    """
    Return the OCR word that constitutes a text-valued field's value.

    Used only for fields flagged ``allow_text_values: true``. The value is the
    first non-empty, non-unit OCR word located strictly to the right of the
    field label and before the next registered field boundary (or numeric
    value), i.e. the same horizontal segment the numeric path would use for
    the same field. Returns ``None`` when the source carries no text value.

    The captured word is returned as a tuple:

        (word, text_value, raw_text, confidence)
    """

    if label_row is None:
        label_row = row

    words = sorted(
        row.words,
        key=lambda word: word.left,
    )

    if not words:
        return None

    label_start, label_end, next_field_left = _compute_value_bounds(
        label_row,
        field,
        region_fields,
    )

    if label_end is None:
        return None

    for word in words:

        if word.left <= label_end:
            continue

        if (
            next_field_left is not None
            and word.left >= next_field_left
        ):
            break

        raw = str(
            word.text or ""
        ).strip()

        if not raw:
            continue

        if raw in UNIT_TOKENS:
            continue

        text = clean(raw)

        if not text:
            continue

        try:
            confidence = round(
                float(word.conf or 0) / 100,
                4,
            )
        except Exception:
            confidence = 0.0

        return (
            word,
            text,
            raw,
            confidence,
        )

    return None


def _row_at_offset(
    rows,
    row,
    offset,
):
    """
    Return the OCR row at a relative y offset from `row` (e.g. -1 = the row
    immediately above, +1 = immediately below), or None when out of range.
    """

    if offset == 0:
        return row

    ordered = sorted(
        rows,
        key=lambda r: r.y,
    )

    try:
        index = ordered.index(row)
    except ValueError:
        return None

    target = index + int(offset)

    if (
        target < 0
        or target >= len(ordered)
    ):
        return None

    return ordered[target]


def find_value_words(
    row,
    field,
    region_fields=None,
    label_row=None,
):
    """
    Extract numeric values belonging to the selected field.

    By default values are collected on the same physical OCR row used to
    identify the field, limited to the horizontal area between this field's
    label and the next registered field label.

    When `label_row` is provided, the label segment is located on that row
    while numeric values are collected from `row`. This supports fields
    declared with `value_row_offsets` whose measurements live on a different
    physical OCR row than the label (e.g. P001 above, L007 below).
    """

    words = sorted(
        row.words,
        key=lambda word: word.left,
    )

    if not words:
        return []

    if label_row is None:

        label_row = row

    label_start, label_end, next_field_left = _compute_value_bounds(
        label_row,
        field,
        region_fields,
    )

    return _collect_numeric_values(
        row,
        field,
        label_end,
        next_field_left,
        left_bound=(label_start if label_row is not row else label_end),
    )


def unwrap_header(
    entry
):
    """
    Convert any supported header representation into:

        {
            "y": ...,
            "columns": [...]
        }

    Supported input forms:

        dict
        (score, dict)
        [score, dict]
        (dict,)
    """

    if isinstance(
        entry,
        dict
    ):

        return entry

    if isinstance(
        entry,
        (tuple, list)
    ):

        # Search from the end for a header dictionary.

        for item in reversed(
            entry
        ):

            if isinstance(
                item,
                dict
            ):

                return item

    return None


# ============================================================
# HEADER SELECTION
# ============================================================

def find_header_for_row(
    row,
    headers,
    maximum_distance=180,
):
    """
    Find the closest header above the selected field row.

    IMPORTANT:
    `headers` in this project may contain tuples such as:

        (score, header)

    so entries are unwrapped before accessing `["y"]`.
    """

    if not headers:
        return None

    candidates = []

    for entry in headers:

        header = unwrap_header(
                entry
            )

        if header is None:
            continue

        if "y" not in header:
            continue

        if "columns" not in header:
            continue

        try:

            header_y = float(
                    header["y"]
                )

        except (
            TypeError,
            ValueError,
        ):

            continue

        distance = (
            row.y -
            header_y
        )

        if (
            distance >= 0
            and
            distance <=
                maximum_distance
        ):

            candidates.append(
                (
                    distance,
                    header,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item:
            item[0]
    )

    return candidates[0][1]


# ============================================================
# FALLBACK COLUMNS
# ============================================================

def default_columns_for_region(
    region,
    field_name="",
):
    region = str(
        region or ""
    ).lower()

    key = field_key(
        field_name
    )

    explicit = EXPLICIT_COLUMNS.get(
            key
        )

    if explicit:
        return explicit

    if region in {
        "cylinder_pressure",
        "cylinder_condition",
        "cylinder_lubrication",
        "liner_wall",
    }:

        return [
            f"CYL{i}"
            for i in range(
                1,
                11,
            )
        ]

    if region == "crankcase":

        if (
            "main bearing" in key
            or
            "bearing" in key
        ):

            return [
                f"BRG{i}"
                for i in range(
                    1,
                    14,
                )
            ]

        return [
            f"CYL{i}"
            for i in range(
                1,
                11,
            )
        ]

    if region in {
        "turbocharger",
        "tc_sac_media",
    }:

        return [
            "REF",
            "CALC",
            "AVG",
            "TC1",
            "TC2",
            "TC3",
        ]

    return []


# ============================================================
# COLUMN MAPPING
# ============================================================

def explicit_column_for_value(
    field,
    index,
):
    columns = get_explicit_columns(
            field
        )

    if not columns:
        return None

    if index >= len(columns):
        return None

    return columns[index]


def sparse_geo_remap(
    values,
    explicit,
    header,
    row,
):
    """
    Build a guarded geometric column remap for a sparse multi-column row.

    Returns a list parallel to ``values`` (one semantic column per value)
    when the remap is trustworthy, otherwise ``None`` to signal that the
    caller should fall back to plain positional assignment.

    The remap is accepted only when ALL of the following hold:
        1. A physical header with column x-positions was detected.
        2. The header row is at or above the value row (header geometry is
           independent of the values), allowing a small vertical tolerance
           for OCR positioning artifacts.
        3. The row is sparse (fewer surviving values than schema columns).
        4. Every value maps (nearest-header) to a schema column.
        5. No schema column is assigned more than once.
        6. The leading value maps onto the first schema column, or onto
           the second when exactly one leading column is blank.

    A fully-populated row (positional == geometric) is left to the plain
    positional path -- this remap is only for genuine gaps.
    """

    if not values:
        return None

    cols = (
        header
        and header.get("columns")
    ) or []

    if not cols:
        return None

    if len(values) >= len(explicit):
        return None

    # Independent geometry: header row must sit at or above the value row.
    # Allow a small tolerance for OCR vertical positioning artifacts.
    try:
        hdr_y = float(header.get("y"))
        row_y = float(row.y)
    except Exception:
        return None

    # Relaxed check: header can be at or slightly below the value row
    # (within 5 pixels tolerance) due to OCR row positioning artifacts.
    # The key is that the header geometry is from the same logical row
    # structure, not derived from the value words themselves.
    if hdr_y > row_y + 5.0:
        return None

    remap = []
    try:
        for word, _ in values:
            remap.append(
                nearest_header(
                    word.cx,
                    cols,
                )
            )
    except Exception:
        return None

    if len(remap) != len(values):
        return None

    if any(c not in explicit for c in remap):
        return None

    if len(set(remap)) != len(remap):
        return None

    # The surviving values must anchor to the leading schema columns,
    # tolerating a single blank leading column (e.g. a blank REF/XPERT).
    if remap[0] not in (explicit[0], (explicit[1] if len(explicit) > 1 else explicit[0])):
        return None

    return remap


def determine_column(
    field,
    word,
    header,
    value_index,
    total_values,
    region,
    geo_remap=None,
):
    """
    Resolve the semantic column for one extracted measurement.

    Rules:
        1. Explicit schemas are authoritative for multi-column fields.
        2. Ordinary scalar fields always use VALUE.
        3. Paired scalar fields use their explicit pair names.
        4. Header geometry is used only for genuinely multi-column fields.
        5. Never assign a scalar field to REF/CALC/MEAS/ENGINE/etc.
    """

    field_name = field.get(
        "name",
        "",
    )

    key = field_key(
        field_name
    )

    # ------------------------------------------------------------
    # 1. Explicit schema.
    # ------------------------------------------------------------

    explicit = get_explicit_columns(
        field
    )

    if explicit:

        # ----------------------------------------------------
        # Sparse-row header remap (row-level, guarded).
        #
        # A multi-column field is normally mapped positionally (the
        # i-th OCR value -> the i-th declared column). That mapping is
        # correct when the row is fully populated. But when one or more
        # values are skipped/blank -- e.g. a zero-confidence word that
        # was rejected, a CALC/AVG slot holding a stray non-numeric
        # glyph, or a genuinely empty source column -- positional
        # mapping shifts every remaining value one column left and drops
        # the trailing column(s) (the last column appears "missing").
        #
        # When the row is sparse, a physical header with column
        # x-positions was detected independently of the value row, and
        # the surviving values land geometrically under the LEADING
        # schema columns (a single leading blank is tolerated), we
        # resolve each value against its nearest header x-coordinate so
        # it lands in the true semantic column. This remap is only
        # trusted when it is faithful: all mapped columns belong to the
        # schema, no column is reused, and the header geometry is
        # independent (a real header row above the value row). Any
        # unfaithful mapping falls back to positional assignment.
        # ----------------------------------------------------

        if (
            geo_remap
            and value_index < len(geo_remap)
        ):
            return geo_remap[value_index]

        if value_index < len(explicit):
            return explicit[value_index]

        return None

    # ------------------------------------------------------------
    # 2. Known paired scalar fields.
    # ------------------------------------------------------------

    paired_columns = {
        "ship sog stw": [
            "SOG",
            "STW",
        ],
        "draft fwd aft": [
            "FWD",
            "AFT",
        ],
        "draft mid trim": [
            "MID",
            "TRIM",
        ],
    }

    pair = paired_columns.get(key)

    if pair:
        if value_index < len(pair):
            return pair[value_index]
        return None

    # ------------------------------------------------------------
    # 3. Determine whether this is a genuinely columnar field.
    # ------------------------------------------------------------

    multi_column = is_multi_column_field(
        field
    )

    # Fields which are explicitly known to contain multiple
    # measurements may use header geometry.
    #
    # For ordinary scalar fields, NEVER use the detected PDF
    # header because the nearest header may be something such as
    # ENGINE, REF, CALC or MEAS belonging to another table.

    if not multi_column:

        if total_values == 1:
            return "VALUE"

        # A scalar field unexpectedly producing multiple values
        # is unresolved rather than being mapped to an unrelated
        # header.
        return None

    # ------------------------------------------------------------
    # 4. Multi-column fields: header geometry.
    # ------------------------------------------------------------

    if header:

        raw_columns = header.get(
            "columns",
            [],
        )

        # Extract semantic names from header records.
        header_names = []

        for item in raw_columns:

            if isinstance(
                item,
                dict,
            ):

                name = item.get(
                    "column"
                )

                if name:
                    header_names.append(
                        name
                    )

            elif isinstance(
                item,
                str,
            ):

                header_names.append(
                    item
                )

        # Exact width is safe for a fully populated row.
        if (
            len(header_names)
            == total_values
            and value_index < len(header_names)
        ):
            return header_names[
                value_index
            ]

        # Otherwise use the physical x coordinate.
        try:
            mapped = nearest_header(
                word.cx,
                raw_columns,
            )
        except Exception:
            mapped = None

        if mapped:
            return mapped

    # ------------------------------------------------------------
    # 5. Explicit region fallback.
    # ------------------------------------------------------------

    fallback = default_columns_for_region(
        region,
        field_name,
    )

    if fallback:

        if (
            len(fallback)
            == total_values
            and value_index < len(fallback)
        ):
            return fallback[
                value_index
            ]

        if (
            value_index < len(fallback)
            and multi_column
        ):
            return fallback[
                value_index
            ]

    return None


def _scan_ce_datetime(row):
    """Assemble a structured datetime from a CE "Date and time of recording" row.

    A datetime is only emitted from tokens that are structurally associated
    with the field's own logical row: a well-formed ``M/D/YYYY`` (or
    ``M-D-YYYY``) date must appear on the row. A clock time is only appended
    when it appears on that row as an unambiguous colon-joined ``HH:MM`` token,
    or as an OCR-split ``HH:`` + ``MM`` pair of adjacent tokens (the recurring
    leading ``[15]`` OCR box artifact is treated as noise). Dotted values such
    as ``19.90``, ``20.60`` and ``25.0`` are never accepted as times, and an
    OCR-split pair is only combined when the minute token is a bare two-digit
    token placed immediately after the hour fragment on the same row. Arbitrary
    page tokens (footer dates, speeds, title lines, neighbouring measurements)
    are never scanned, so a datetime cannot be fabricated from unrelated
    content.
    Returns ``YYYY-MM-DD``, ``YYYY-MM-DD HH:MM``, or ``None`` (OCR_MISSING).
    """

    words = sorted(
        row.words,
        key=lambda w: w.left,
    )

    month = day = year = None
    date_right = None

    for w in words:
        m = _CE_DATE_MD.search(
            str(w.text)
        )
        if m:
            month, day, year = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
            )
            date_right = w.right
            break

    if month is None:
        return None

    window = (
        date_right + 550
    )

    hh = mm = None

    for i in range(len(words)):
        w = words[i]

        if w.left <= date_right:
            continue

        if w.left >= window:
            break

        token = str(
            w.text or ""
        ).strip()

        # The recurring leading OCR box artifact (e.g. "[15]" / "fis]")
        # immediately precedes the clock time on the G001 row.  It is pure
        # noise (never a measurement or a label); skip it transparently so it
        # does not end the structurally-associated time run.
        if "[" in token or "]" in token:
            continue

        if not _CE_TIME_TOKEN.match(
            token
        ):
            # First non-time-looking token ends the structurally-associated
            # time run (e.g. "Sea water temp.", "SHIP", "SOG", "Voy.").
            break

        m = _CE_TIME.search(
            token
        )

        if m:
            h, mm_t = (
                int(m.group(1)),
                int(m.group(2)),
            )
            if (
                0 <= h <= 23
                and 0 <= mm_t <= 59
            ):
                hh, mm = h, mm_t
                break

        # Not a complete "HH:MM" token.  If it is an OCR-split hour fragment
        # ("HH:") whose immediately following token is a bare two-digit minute
        # placed right next to it on the same row, combine them into one clock
        # time.  Anything else (dotted measurements, labels, non-adjacent
        # tokens) is never treated as a time.
        frag = _CE_HOUR_FRAG.match(token) if token else None
        if frag and i + 1 < len(words):
            nxt = words[i + 1]
            if nxt.left > w.right and nxt.left >= date_right and nxt.left < window:
                nxt_tok = str(nxt.text or "").strip()
                gap = nxt.left - w.right
                if (
                    gap <= w.width
                    and _CE_MIN_FRAG.match(nxt_tok)
                ):
                    h = int(frag.group(1))
                    mm_t = int(nxt_tok)
                    if (
                        0 <= h <= 23
                        and 0 <= mm_t <= 59
                    ):
                        hh, mm = h, mm_t
                        break

    yy = int(year)
    if yy < 100:
        yy += 2000
    date_str = (
        f"{yy:04d}-{int(month):02d}-{int(day):02d}"
    )

    if hh is not None and mm is not None:
        return f"{date_str} {hh:02d}:{mm:02d}"

    return date_str



def extract_field(
    field,
    rows,
    region_fields=None,
    field_index=None,
    headers=None,
    page=None,
    region=None,
):
    """
    Extract one registered field.

    Compatible with the current crop_pipeline.py call:

        extract_field(
            field,
            rows,
            region_fields,
            field_index,
            headers,
            page,
            region,
        )
    """

    if headers is None:
        headers = []

    if region is None:

        region = field.get(
                "region",
                "",
            )

    if page is None:

        page = field.get(
                "page",
                "",
            )

    # --------------------------------------------------------
    # Find row.
    # --------------------------------------------------------

    row, row_score = find_field_row(
            rows,
            field,
        )

    # --------------------------------------------------------
    # Create record.
    # --------------------------------------------------------

    record = FieldRecord(
            field_id = field["id"],

            field_name = field["name"],

            page = page,

            region = region,

            unit = field.get(
                    "unit",
                    "",
                ),

            found = row is not None,

            row_text = (
                    row.text
                    if row
                    else ""
                ),

            row_confidence = (
                    (
                        sum(
                            word.conf
                            for word in
                            row.words
                        )
                        /
                        len(
                            row.words
                        )
                        /
                        100
                    )
                    if (
                        row
                        and
                        row.words
                    )
                    else 0
                ),

            cells=[],

            flags=[],
        )

    if row is None:

        record.flags.append(
            "field_row_not_found"
        )

        return record

    # --------------------------------------------------------
    # Find numeric words.
    # --------------------------------------------------------

    values = find_value_words(
            row,
            field,
            region_fields,
        )

    # --------------------------------------------------------
    # Structured datetime fields (e.g. "Date and time of recording").
    #
    # These carry a date and a clock time on their own row; they must NEVER be
    # collected as raw scalar numerics (which previously surfaced the time
    # fragments as VAL1/VAL2 and let a neighbouring measurement such as the sea
    # water temperature bleed in as a value). When a well-formed date (with an
    # optional structurally-associated time) is present, a single DATETIME cell
    # is emitted; otherwise the field yields no cells (OCR_MISSING).
    # --------------------------------------------------------

    is_datetime_field = (
        field_key(
            field.get("name", "")
        ).startswith("date and time")
    )

    if is_datetime_field:

        dt = _scan_ce_datetime(
            row
        )

        if dt:

            record.cells.append(
                Cell(
                    field_id=field["id"],
                    field_name=field["name"],
                    page=page,
                    region=region,
                    column="DATETIME",
                    raw_text=dt,
                    value=dt,
                    unit="",
                    confidence=1.0,
                    bbox=[0, 0, 0, 0],
                    source_row=(
                        row.text
                        if row
                        else ""
                    ),
                    flags=["datetime"],
                )
            )

            return record

        record.flags.append(
            "no_value_cells"
        )

        return record

    # --------------------------------------------------------
    # value_row_offsets: some fields carry their measurements on a
    # different physical OCR row than their label (e.g. P001 on the row
    # above, L007 on the row below). When the label row yields no values,
    # try each configured relative offset, keeping the label row solely
    # for identifying the field (label segment).
    # --------------------------------------------------------

    if not values:

        value_row_offsets = field.get(
            "value_row_offsets"
        )

        if value_row_offsets:

            for offset in value_row_offsets:

                offset_row = _row_at_offset(
                    rows,
                    row,
                    offset,
                )

                if offset_row is None:
                    continue

                values = find_value_words(
                        offset_row,
                        field,
                        region_fields,
                        label_row=row,
                    )

                if values:
                    break

    # --------------------------------------------------------
    # Cap collection to the number of declared semantic columns.
    #
    # A multi-column field (one with an explicit schema) cannot
    # legitimately hold more values than it has named columns. When
    # the physical row contains later values belonging to other fields
    # to the right (e.g. P002 "ENGINE speed" also shares its row with
    # "Power kW", "Light running", "Theoretic."), keep only the first
    # `len(schema)` values so they map positionally 1:1 to the schema.
    # --------------------------------------------------------

    schema = get_explicit_columns(field)

    if schema and len(values) > len(schema):

        values = values[:len(schema)]

    if not values:

        # ----------------------------------------------------
        # Text-valued fields (allow_text_values): when the numeric
        # path finds no measurements, capture the field's textual
        # value (e.g. "stable", "Economy", "off", "Laboratory") as a
        # single VALUE cell. This is additive and does not affect
        # numeric extraction.
        # ----------------------------------------------------

        if field.get("allow_text_values"):

            text_result = _collect_text_value(
                row,
                field,
                region_fields,
            )

            if text_result is not None:

                word, text_value, raw_text, confidence = text_result

                record.cells.append(
                    Cell(
                        field_id = field["id"],

                        field_name = field["name"],

                        page = page,

                        region = region,

                        column = "VALUE",

                        raw_text = clean(
                                raw_text
                            ),

                        value = text_value,

                        unit = field.get(
                                "unit",
                                "",
                            ),

                        confidence = confidence,

                        bbox=[
                            word.left,
                            word.top,
                            word.width,
                            word.height,
                        ],

                        source_row = row.text,

                        flags = [],
                    )
                )

                return record

        record.flags.append(
            "no_numeric_cells"
        )

        return record

    # --------------------------------------------------------
    # Header.
    # --------------------------------------------------------

    header = find_header_for_row(
            row,
            headers,
        )

    if header is None:

        record.flags.append(
            "header_not_found"
        )

    # --------------------------------------------------------
    # Multi-column status.
    # --------------------------------------------------------

    multi_column = is_multi_column_field(
            field
        )

    used_columns = set()

    # --------------------------------------------------------
    # Row-level guarded geometric remap for sparse explicit
    # multi-column rows (see sparse_geo_remap). None => positional.
    # --------------------------------------------------------

    geo_remap = None

    explicit_schema = get_explicit_columns(
            field
        )

    if explicit_schema:

        geo_remap = sparse_geo_remap(
                values,
                explicit_schema,
                header,
                row,
            )

    # --------------------------------------------------------
    # Rule C (clean-value precedence within the same logical slot).
    #
    # A token that needed OCR-debris cleanup to become numeric is only
    # admitted when it does not collide with a CLEAN candidate occupying
    # the same physical cell/slot. When a debris-cleaned candidate and a
    # clean candidate resolve to the SAME column and are positionally the
    # same cell (< 60px apart), keep the clean one and discard the debris
    # one. This is NOT a confidence threshold -- a debris token with no
    # same-slot clean competitor (e.g. P001 TORSIOM. `21630/@`, F013
    # `107,@` vs `112` in separate cells) is always kept.
    #
    # 60px is narrower than a column pitch, so distinct values in adjacent
    # columns are never conflated.
    # --------------------------------------------------------

    if values:

        skip_indices = set()

        plans = []

        for index, (
            word,
            value,
        ) in enumerate(values):

            col = determine_column(
                    field = field,

                    word = word,

                    header = header,

                    value_index = index,

                    total_values = len(values),

                    region = region,

                    geo_remap = geo_remap,
                )

            plans.append(
                (
                    index,
                    word,
                    value,
                    col,
                )
            )

        for i, (
            index,
            word,
            value,
            col,
        ) in enumerate(plans):

            if not getattr(
                word,
                "debris_cleaned",
                False,
            ):
                continue

            for j, (
                index2,
                word2,
                value2,
                col2,
            ) in enumerate(plans):

                if i == j:
                    continue

                if getattr(
                    word2,
                    "debris_cleaned",
                    False,
                ):
                    continue

                if (
                    str(col) == str(col2)
                    and
                    abs(
                        word.cx - word2.cx
                    ) < 60
                ):

                    skip_indices.add(
                        index
                    )

                    break

        if skip_indices:

            values = [
                entry
                for idx, entry in enumerate(
                    values
                )
                if idx not in skip_indices
            ]

    # --------------------------------------------------------
    # Create one Cell per OCR measurement.
    # --------------------------------------------------------

    for index, (
        word,
        value,
    ) in enumerate(values):

        column = determine_column(
                field = field,

                word = word,

                header = header,

                value_index = index,

                total_values = len(values),

                region = region,

                geo_remap = geo_remap,
            )

        flags = []

        if column is None:

            column = f"VAL{index + 1}"

            flags.append(
                "column_unresolved"
            )

        # ----------------------------------------------------
        # Never silently create VALUE for explicit multi-
        # column fields.
        # ----------------------------------------------------

        if (
            multi_column
            and
            str(
                column
            )
            .strip()
            .upper()
            == "VALUE"
        ):

            column = f"X_{int(word.cx)}"

            flags.append(
                "generic_value_blocked"
            )

            flags.append(
                "column_unresolved"
            )

        # ----------------------------------------------------
        # Duplicate logical column.
        # ----------------------------------------------------

        if column in used_columns:

            flags.append(
                "duplicate_column_assignment"
            )

        used_columns.add(
            column
        )

        # ----------------------------------------------------
        # Confidence.
        # ----------------------------------------------------

        if word.conf < 60:

            flags.append(
                "low_ocr_confidence"
            )

        # ----------------------------------------------------
        # Create cell.
        # ----------------------------------------------------

        record.cells.append(
            Cell(
                field_id = field["id"],

                field_name = field["name"],

                page = page,

                region = region,

                column = column,

                raw_text = clean(
                        word.text
                    ),

                value = value,

                unit = field.get(
                        "unit",
                        "",
                    ),

                confidence = round(
                        word.conf /
                        100,
                        4,
                    ),

                bbox=[
                    word.left,
                    word.top,
                    word.width,
                    word.height,
                ],

                source_row = row.text,

                flags = flags,
            )
        )

    return record
