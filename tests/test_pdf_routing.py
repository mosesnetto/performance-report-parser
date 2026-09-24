"""Routing regression tests for _dispatch() in pdf_pipeline.

These fast unit tests verify that the engine argument correctly selects the
V4 zone pipeline vs. the V3 process_tb path for Template B, and that
Template A routing is unaffected.
"""

from unittest.mock import MagicMock, patch

from meridian.pdf_pipeline import _dispatch

# ───────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────

_FAKE_PDF = "dummy.pdf"
_FAKE_FIELDS = "/fake/config/fields.yaml"
_FAKE_OUTPUT = "/tmp/test_routing_out"


def _make_summary(source="v3"):
    return {"source": source, "expected_fields": 0}


# ───────────────────────────────────────────────────────────
# Template B routing
# ───────────────────────────────────────────────────────────

@patch("meridian.pdf_pipeline.load_fields")
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.template_routing.reading_pages_for", return_value=[11, 12])
@patch("meridian.template_routing.detect_template", return_value="B")
@patch("meridian.template_b.process_tb")
def test_tb_v3_routes_to_process_tb(mock_tb, mock_detect, mock_rpf,
                                     mock_path_cls, mock_shutil, mock_load):
    """Template B + engine=v3 must call process_tb (V3 path)."""
    mock_load.return_value = []
    mock_tb.return_value = _make_summary("v3")
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out

    result = _dispatch(
        _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
        dpi=300, psm=11, engine="v3",
    )
    mock_tb.assert_called_once()
    assert result["source"] == "v3"


@patch("meridian.pdf_pipeline.load_fields")
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.template_routing.reading_pages_for", return_value=[11, 12])
@patch("meridian.template_routing.detect_template", return_value="B")
def test_tb_v4_routes_to_process_tb_v4(mock_detect, mock_rpf,
                                        mock_path_cls, mock_shutil, mock_load):
    """Template B + engine=v4 must call _process_tb_v4 (V4 zone path)."""
    mock_load.return_value = []
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out

    fake_summary = _make_summary("v4")
    with patch("meridian.pdf_pipeline._process_tb_v4",
               return_value=fake_summary) as mock_v4:
        result = _dispatch(
            _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
            dpi=300, psm=11, engine="v4",
        )
        mock_v4.assert_called_once()
        assert result["source"] == "v4"


@patch("meridian.pdf_pipeline.load_fields")
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.template_routing.reading_pages_for", return_value=[11, 12])
@patch("meridian.template_routing.detect_template", return_value="B")
def test_tb_v4_does_not_call_process_tb(mock_detect, mock_rpf,
                                         mock_path_cls, mock_shutil, mock_load):
    """When engine=v4, process_tb must never be called."""
    mock_load.return_value = []
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out

    with patch("meridian.template_b.process_tb") as mock_tb:
        with patch("meridian.pdf_pipeline._process_tb_v4",
                   return_value=_make_summary("v4")):
            _dispatch(
                _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
                dpi=300, psm=11, engine="v4",
            )
            mock_tb.assert_not_called()


@patch("meridian.pdf_pipeline.load_fields")
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.template_routing.reading_pages_for", return_value=[11, 12])
@patch("meridian.template_routing.detect_template", return_value="B")
def test_tb_v3_strategy_json(mock_detect, mock_rpf, mock_path_cls,
                              mock_shutil, mock_load):
    """Template B + v3 writes strategy='template-b-whole-page' with engine key."""
    mock_load.return_value = []
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out
    mock_routing = MagicMock()
    mock_out.__truediv__ = MagicMock(return_value=mock_routing)

    with patch("meridian.template_b.process_tb",
               return_value=_make_summary()):
        _dispatch(
            _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
            dpi=300, psm=11, engine="v3",
        )
        mock_routing.write_text.assert_called()
        import json
        call_args = mock_routing.write_text.call_args
        written = json.loads(call_args[0][0])
        assert written["strategy"] == "template-b-whole-page"
        assert written["engine"] == "v3"


@patch("meridian.pdf_pipeline.load_fields")
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.template_routing.reading_pages_for", return_value=[11, 12])
@patch("meridian.template_routing.detect_template", return_value="B")
def test_tb_v4_strategy_json(mock_detect, mock_rpf, mock_path_cls,
                              mock_shutil, mock_load):
    """Template B + v4 writes strategy='template-b-whole-page-v4' with engine key."""
    mock_load.return_value = []
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out
    mock_routing = MagicMock()
    mock_out.__truediv__ = MagicMock(return_value=mock_routing)

    with patch("meridian.pdf_pipeline._process_tb_v4",
               return_value=_make_summary("v4")):
        _dispatch(
            _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
            dpi=300, psm=11, engine="v4",
        )
        mock_routing.write_text.assert_called()
        import json
        call_args = mock_routing.write_text.call_args
        written = json.loads(call_args[0][0])
        assert written["strategy"] == "template-b-whole-page-v4"
        assert written["engine"] == "v4"


# ───────────────────────────────────────────────────────────
# Template A routing (unchanged)
# ───────────────────────────────────────────────────────────

@patch("meridian.pdf_pipeline.process_crops", return_value=_make_summary("ta"))
@patch("meridian.pdf_pipeline.detect_reading_pages", return_value=[13, 14])
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.pdf_pipeline.pymupdf")
@patch("meridian.template_routing.detect_template", return_value="A")
@patch("PIL.Image.open")
def test_ta_v4_unchanged(mock_img_open, mock_detect, mock_mupdf, mock_path_cls,
                          mock_shutil, mock_dr, mock_crops):
    """Template A + engine=v4 still calls process_crops with engine=v4."""
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out
    mock_pil = MagicMock()
    mock_img_open.return_value = mock_pil
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG"
    mock_doc = MagicMock()
    mock_mupdf.open.return_value.__enter__ = MagicMock(return_value=mock_doc)
    mock_mupdf.open.return_value.__exit__ = MagicMock(return_value=False)

    with patch("meridian.pdf_pipeline._render_page", return_value=mock_pix):
        with patch("meridian.pdf_pipeline.boxes_for_page", return_value=[]):
            _dispatch(
                _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
                dpi=300, psm=11, engine="v4",
            )
            call_kwargs = mock_crops.call_args
            assert call_kwargs[1].get("engine") == "v4"


@patch("meridian.pdf_pipeline.process_crops", return_value=_make_summary("ta"))
@patch("meridian.pdf_pipeline.detect_reading_pages", return_value=[13, 14])
@patch("meridian.pdf_pipeline.shutil")
@patch("meridian.pdf_pipeline.Path")
@patch("meridian.pdf_pipeline.pymupdf")
@patch("meridian.template_routing.detect_template", return_value="A")
@patch("PIL.Image.open")
def test_ta_v3_unchanged(mock_img_open, mock_detect, mock_mupdf, mock_path_cls,
                          mock_shutil, mock_dr, mock_crops):
    """Template A + engine=v3 still calls process_crops with engine=v3."""
    mock_out = MagicMock()
    mock_out.exists.return_value = False
    mock_path_cls.return_value = mock_out
    mock_pil = MagicMock()
    mock_img_open.return_value = mock_pil
    mock_pix = MagicMock()
    mock_pix.tobytes.return_value = b"\x89PNG"
    mock_doc = MagicMock()
    mock_mupdf.open.return_value.__enter__ = MagicMock(return_value=mock_doc)
    mock_mupdf.open.return_value.__exit__ = MagicMock(return_value=False)

    with patch("meridian.pdf_pipeline._render_page", return_value=mock_pix):
        with patch("meridian.pdf_pipeline.boxes_for_page", return_value=[]):
            _dispatch(
                _FAKE_PDF, None, _FAKE_FIELDS, _FAKE_OUTPUT,
                dpi=300, psm=11, engine="v3",
            )
            call_kwargs = mock_crops.call_args
            assert call_kwargs[1].get("engine") == "v3"
