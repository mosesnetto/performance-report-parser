"""MERIDIAN Voice: offline best-effort spoken briefings via Piper TTS.

Piper (https://github.com/rhasspy/piper) synthesises natural voice from a
small neural model on-device — no cloud, no audio data leaves the machine.
This module degrades gracefully: if the ``piper`` binary or a voice model is
not installed, calls return ``None`` and the web layer reports that voice is
unavailable instead of failing the page.

Environment overrides
---------------------
MERIDIAN_PIPER_BIN     path to the ``piper`` executable
MERIDIAN_PIPER_MODEL   path to a ``.onnx`` voice model (or a directory
                       containing ``model.onnx`` + ``config.json``)
MERIDIAN_PIPER_CONFIG  path to the matching piper ``config.json``
"""

import os
import shutil
import subprocess
from pathlib import Path

PIPER_BIN_ENV = "MERIDIAN_PIPER_BIN"
PIPER_MODEL_ENV = "MERIDIAN_PIPER_MODEL"
PIPER_CONFIG_ENV = "MERIDIAN_PIPER_CONFIG"

LOW_CONFIDENCE_THRESHOLD = 80.0


def piper_binary() -> list[str] | None:
    """Return ``[path-to-piper-binary]`` if a piper executable is reachable."""
    explicit = os.environ.get(PIPER_BIN_ENV)
    if explicit:
        return [explicit]
    found = shutil.which("piper")
    return [found] if found else None


def piper_model() -> str | None:
    """Return the configured piper voice model path, if any."""
    return os.environ.get(PIPER_MODEL_ENV) or None


def voice_available() -> bool:
    """True when speech synthesis can run right now."""
    return bool(piper_binary() and piper_model())


def briefing_text(payload: dict) -> str:
    """Build a short natural-language briefing from a normalized report.

    ``payload`` is the output of :func:`meridian.web.get_report_data`
    (``{"summary": {...}, "data": {"rows": [...], ...}}``).
    """
    summary = payload.get("summary", {}) or {}
    data = payload.get("data", {}) or {}
    rows = data.get("rows", []) or []

    name = summary.get("filename") or "this report"
    stem = Path(str(name)).stem.replace("_", " ").replace("-", " ").strip()

    found = [r for r in rows if r.get("found")]
    missing = [r for r in rows if not r.get("found")]

    confidences = [
        float(r.get("row_confidence", 0.0))
        for r in found
        if r.get("row_confidence") is not None
    ]
    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    lines = [f"MERIDIAN engine intelligence briefing for {stem or name}."]
    if rows:
        lines.append(
            f"{len(found)} of {len(rows)} measurement fields were extracted, "
            f"with an average extraction confidence of {avg_conf:.0f} percent."
        )
    else:
        lines.append("No extracted measurements were found in this report.")

    if missing:
        names = ", ".join(r.get("field_name", "unknown field").lower()
                          for r in missing[:5])
        extra = " and others" if len(missing) > 5 else ""
        lines.append(
            f"Attention: {len(missing)} fields could not be matched to source "
            f"rows{extra}, including {names}."
        )

    weak = [
        r for r in found
        if r.get("row_confidence") is not None
        and float(r.get("row_confidence", 0)) < LOW_CONFIDENCE_THRESHOLD
    ]
    if weak:
        names = ", ".join(r.get("field_name", "unknown field").lower()
                          for r in weak[:5])
        lines.append(
            f"{len(weak)} fields were recovered with low OCR confidence, "
            f"including {names}. Review them in the validator."
        )

    date_value = _extract_date(payload)
    if date_value:
        lines.append(f"Report recording date is {_speak_date(date_value)}.")

    total = len(lines)
    _ = total  # kept for easy extension; no unused-import warnings
    return " ".join(lines)


def _extract_date(payload: dict) -> str | None:
    for row in payload.get("data", {}).get("rows", []):
        if str(row.get("field_name", "")).strip().lower() == "date and time of recording":
            for col in ("DATETIME", "VAL1", "VALUE"):
                for cell in row.get("cells", {}).get(col, []):
                    value = cell.get("value")
                    if value is not None:
                        return str(value)
    return None


def _speak_date(value: str) -> str:
    """Make a machine date (YYYY.MM.DD ...) friendly to read aloud."""
    return value.replace(".", " ").replace("/", " ").replace("-", " ").strip()


def synthesize(text: str, out_wav: Path, timeout: int = 60) -> bool:
    """Synthesize ``text`` to ``out_wav`` using Piper.

    Returns True on success, False when piper or the configured voice model
    is unavailable or the run fails. Never raises.
    """
    bin_cmd = piper_binary()
    model = piper_model()
    if not bin_cmd or not model:
        return False
    try:
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        command = [*bin_cmd, "--model", model]
        config = os.environ.get(PIPER_CONFIG_ENV)
        if config:
            command += ["--config", config]
        command += ["--output_file", str(out_wav)]
        proc = subprocess.run(
            command,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        return False