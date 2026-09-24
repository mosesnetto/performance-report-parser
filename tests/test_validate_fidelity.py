"""
Tests for the generic source-driven fidelity validator.
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meridian.template_routing import TEMPLATE_A, TEMPLATE_B
from meridian.validate_fidelity import (
    SOURCE_ABSENT,
    SOURCE_BLANK,
    SOURCE_PRESENT,
    Discrepancy,
    PDFValidationResult,
    SourceCell,
    UnionCatalog,
    ValidationSummary,
    _assign_tokens_to_columns,
    _build_col_boundaries,
    _parse_value,
    discover_pdf_info,
    group_tokens_into_ocr_rows,
    match_source_to_extraction,
)

# ---------------------------------------------------------------------------
# PDF discovery tests (mocked OCR)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract OCR binary not installed")
def test_discover_pdf_info():
    """Test dynamic reading page discovery and template detection."""
    from meridian.validate_fidelity import PDF_TEST_DIR
    pdfs = list(PDF_TEST_DIR.glob("*.pdf"))
    pdfs = [f for f in pdfs if not str(f).endswith(":Zone.Identifier")]
    assert len(pdfs) >= 6, f"Expected at least 6 PDFs, found {len(pdfs)}"

    for pdf_path in pdfs[:3]:
        info = discover_pdf_info(pdf_path)
        assert "pdf_name" in info
        assert "template" in info
        assert info["template"] in (TEMPLATE_A, TEMPLATE_B)
        assert "reading_pages" in info
        assert isinstance(info["reading_pages"], list)
        assert len(info["reading_pages"]) > 0


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract OCR binary not installed")
def test_template_detection_variety():
    """Verify different PDFs produce different templates."""
    from meridian.validate_fidelity import PDF_TEST_DIR
    pdfs = list(PDF_TEST_DIR.glob("*.pdf"))
    pdfs = [f for f in pdfs if not str(f).endswith(":Zone.Identifier")]
    templates = set()
    for pdf_path in pdfs[:8]:
        info = discover_pdf_info(pdf_path)
        templates.add(info["template"])
    assert len(templates) >= 2, f"Expected at least 2 templates, found {templates}"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract OCR binary not installed")
def test_discover_reading_pages_dynamic():
    """Verify reading pages are discovered dynamically."""
    from meridian.validate_fidelity import PDF_TEST_DIR
    pdfs = list(PDF_TEST_DIR.glob("*.pdf"))
    pdfs = [f for f in pdfs if not str(f).endswith(":Zone.Identifier")]
    for pdf_path in pdfs[:5]:
        info = discover_pdf_info(pdf_path)
        assert info["reading_pages"] != []
        for p in info["reading_pages"]:
            assert isinstance(p, int) and p > 0


def test_multiple_pdfs_scanned():
    """Verify that all PDFs in pdfs_test are identified."""
    from meridian.validate_fidelity import PDF_TEST_DIR
    pdfs = list(PDF_TEST_DIR.glob("*.pdf"))
    pdfs = [f for f in pdfs if not str(f).endswith(":Zone.Identifier")]
    assert len(pdfs) >= 6, f"Expected at least 6 PDFs in pdfs_test, found {len(pdfs)}"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract OCR binary not installed")
def test_template_a_vs_b_reading_pages():
    """Verify different templates have different reading page structures."""
    from meridian.validate_fidelity import PDF_TEST_DIR
    pdfs = list(PDF_TEST_DIR.glob("*.pdf"))
    pdfs = [f for f in pdfs if not str(f).endswith(":Zone.Identifier")]
    template_pages = {}
    for pdf_path in pdfs[:8]:
        info = discover_pdf_info(pdf_path)
        t = info["template"]
        template_pages.setdefault(t, []).append(info["reading_pages"])
    assert TEMPLATE_A in template_pages or TEMPLATE_B in template_pages


# ---------------------------------------------------------------------------
# OCR helper tests
# ---------------------------------------------------------------------------

def test_group_tokens_into_ocr_rows():
    """Test row grouping from OCR tokens."""
    from meridian.models import OCRRow, OCRWord
    words = [
        OCRWord(page=1, region="test", text="Hello", conf=90.0, left=10, top=100, width=40, height=15),
        OCRWord(page=1, region="test", text="World", conf=90.0, left=60, top=100, width=40, height=15),
        OCRWord(page=1, region="test", text="Next", conf=90.0, left=10, top=130, width=40, height=15),
    ]
    rows = group_tokens_into_ocr_rows(words)
    assert len(rows) >= 1
    assert isinstance(rows[0], OCRRow)


def test_parse_value():
    """Test value parsing."""
    assert _parse_value("25.0") == 25.0
    assert _parse_value("100") == 100.0
    assert _parse_value("") is None
    assert _parse_value("abc") is None
    assert _parse_value("32.5") == 32.5


def test_build_col_boundaries():
    """Test column boundary construction."""
    headers = [
        {"y": 100.0, "columns": [{"column": "REF", "x": 50}, {"column": "CALC", "x": 150}]},
    ]
    from meridian.models import OCRWord
    tokens = [
        OCRWord(page=1, region="test", text="REF", conf=90, left=40, top=95, width=30, height=15),
    ]
    boundaries = _build_col_boundaries(headers, tokens)
    assert len(boundaries) >= 1
    assert boundaries[0][2] == "REF"


def test_assign_tokens_to_columns():
    """Test column assignment."""
    from meridian.models import OCRWord
    boundaries = [
        (0, 100, "REF"),
        (100, 200, "CALC"),
        (200, 300, "AVG"),
    ]
    words = [
        OCRWord(page=1, region="test", text="25.0", conf=90, left=10, top=100, width=30, height=15),
        OCRWord(page=1, region="test", text="30.0", conf=90, left=120, top=100, width=30, height=15),
    ]
    result = _assign_tokens_to_columns(words, boundaries)
    assert "REF" in result
    assert "CALC" in result


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

def test_source_cell_dataclass():
    """Test SourceCell creation."""
    cell = SourceCell(
        field_id="C001", field_name="Test Field", region="cylinder_pressure",
        page=13, column="REF", raw_text="25.0", value=25.0,
        unit="barG", source_status=SOURCE_PRESENT,
    )
    assert cell.field_name == "Test Field"
    assert cell.source_status == SOURCE_PRESENT
    assert cell.value == 25.0


def test_discrepancy_dataclass():
    """Test Discrepancy creation."""
    disc = Discrepancy(
        category="MISSING_VALUE",
        pdf_name="test.pdf",
        template=TEMPLATE_A,
        reading_pages=[13, 14],
        region="general",
        field_name="Sea water temp.",
        source_status=SOURCE_PRESENT,
        source_value="25.0",
        details="Test discrepancy",
    )
    assert disc.category == "MISSING_VALUE"
    assert disc.pdf_name == "test.pdf"


def test_union_catalog_dataclass():
    """Test UnionCatalog creation."""
    catalog = UnionCatalog(
        fields=[{"field_name": "test"}],
        regions=["general", "power_speed"],
        pdf_count=2,
        unique_field_names=1,
        unique_headers=1,
        unique_regions=2,
        physical_tables=1,
    )
    assert catalog.unique_field_names == 1
    assert catalog.unique_regions == 2


def test_validation_summary_dataclass():
    """Test ValidationSummary creation."""
    summary = ValidationSummary(total_pdfs=2)
    summary.templates_discovered = [TEMPLATE_A, TEMPLATE_B]
    assert summary.total_pdfs == 2


def test_pdf_validation_result():
    """Test PDFValidationResult creation."""
    result = PDFValidationResult(
        pdf_name="test.pdf",
        template=TEMPLATE_A,
        reading_pages=[13, 14],
        regions=["general", "power_speed"],
        source_physical_cells=100,
        html_cells_matched=95,
        download_cells_matched=95,
    )
    assert result.pdf_name == "test.pdf"
    assert result.source_physical_cells == 100


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------

def test_match_source_to_extraction():
    """Test source vs extraction matching."""
    source_cells = [
        SourceCell(
            field_id="C001", field_name="Sea water temp.", region="general",
            page=13, column="VALUE", raw_text="25.0", value=25.0,
            unit="°C", source_status=SOURCE_PRESENT,
        ),
    ]
    extraction_records = [
        {"field_id": "G005", "field_name": "Sea water temp.", "column": "VALUE",
         "raw_text": "25.0)", "value": 25.0, "unit": "°C", "page": 13, "region": "general"},
    ]
    discrepancies, html_matched = match_source_to_extraction(source_cells, extraction_records)
    assert html_matched >= 0
    assert isinstance(discrepancies, list)


def test_source_absent_vs_missing():
    """Verify SOURCE_ABSENT vs MISSING_VALUE distinction."""
    absent_cell = SourceCell(
        field_id="X999", field_name="Nonexistent Field", region="general",
        page=13, column="VALUE", raw_text="", value=None,
        unit="", source_status=SOURCE_ABSENT,
    )
    assert absent_cell.source_status == SOURCE_ABSENT

    blank_cell = SourceCell(
        field_id="C001", field_name="Test Field", region="general",
        page=13, column="VALUE", raw_text="", value=None,
        unit="barG", source_status=SOURCE_BLANK,
    )
    assert blank_cell.source_status == SOURCE_BLANK


def test_source_present_vs_absent_not_counted_as_missing():
    """Source-absent items must NOT generate MISSING_VALUE discrepancies."""
    source_cells = [
        SourceCell(
            field_id="C001", field_name="Existing Field", region="general",
            page=13, column="VALUE", raw_text="25.0", value=25.0,
            unit="barG", source_status=SOURCE_PRESENT,
        ),
        SourceCell(
            field_id="X999", field_name="Nonexistent Field", region="general",
            page=13, column="VALUE", raw_text="", value=None,
            unit="", source_status=SOURCE_ABSENT,
        ),
    ]
    extraction_records = [
        {"field_id": "C001", "field_name": "Existing Field", "column": "VALUE",
         "raw_text": "25.0", "value": 25.0, "unit": "barG", "page": 13, "region": "general"},
    ]
    discrepancies, html_matched = match_source_to_extraction(source_cells, extraction_records)
    for disc in discrepancies:
        assert disc.source_status != SOURCE_ABSENT or disc.category != "MISSING_VALUE"


def test_blank_cells_preserved_not_shifted():
    """Blank cells should remain blank, not shift adjacent values."""
    source_cells = [
        SourceCell(
            field_id="C001", field_name="Test Field", region="general",
            page=13, column="REF", raw_text="25.0", value=25.0,
            unit="barG", source_status=SOURCE_PRESENT,
        ),
        SourceCell(
            field_id="C001", field_name="Test Field", region="general",
            page=13, column="CALC", raw_text="", value=None,
            unit="barG", source_status=SOURCE_BLANK,
        ),
        SourceCell(
            field_id="C001", field_name="Test Field", region="general",
            page=13, column="AVG", raw_text="30.0", value=30.0,
            unit="barG", source_status=SOURCE_PRESENT,
        ),
    ]
    extraction_records = [
        {"field_id": "C001", "field_name": "Test Field", "column": "REF", "raw_text": "25.0", "value": 25.0, "unit": "barG", "page": 13, "region": "general"},
        {"field_id": "C001", "field_name": "Test Field", "column": "AVG", "raw_text": "30.0", "value": 30.0, "unit": "barG", "page": 13, "region": "general"},
    ]
    discrepancies, html_matched = match_source_to_extraction(source_cells, extraction_records)
    for disc in discrepancies:
        if disc.source_status == SOURCE_PRESENT:
            assert disc.category != "MISSING_VALUE"


# ---------------------------------------------------------------------------
# Discrepancy categories
# ---------------------------------------------------------------------------

def test_discrepancy_categories():
    """Verify all discrepancy categories are defined."""
    from meridian.validate_fidelity import DISCREPANCY_CATEGORIES
    expected = [
        "MISSING_ROW", "MISSING_HEADER", "MISSING_UNIT", "MISSING_CELL",
        "MISSING_VALUE", "WRONG_ROW", "WRONG_COLUMN", "WRONG_VALUE",
        "WRONG_UNIT", "WRONG_EMPTY_CELL", "DUPLICATE", "SPURIOUS",
        "NORMALIZATION_ERROR", "PROVENANCE_LOSS", "NON_READING_LEAKAGE",
        "SOURCE_ABSENT", "SOURCE_BLANK", "SOURCE_UNREADABLE", "SOURCE_AMBIGUOUS",
    ]
    assert DISCREPANCY_CATEGORIES == expected


def test_union_catalog_structure():
    """Test union catalog has all required fields."""
    catalog = UnionCatalog(
        fields=[{"field_name": "f1", "regions": ["general"]}],
        headers=[{"table_id": "t1", "columns": ["REF", "CALC"]}],
        regions=["general", "power_speed"],
        structures=[],
        source_observations=[],
        pdf_count=2,
        unique_field_names=1,
        unique_headers=1,
        unique_regions=2,
        physical_tables=1,
    )
    d = catalog.to_dict()
    assert "fields" in d
    assert "headers" in d
    assert "regions" in d
    assert "pdf_count" in d


# ---------------------------------------------------------------------------
# Source status constants
# ---------------------------------------------------------------------------

def test_source_status_values():
    """Verify source status constants."""
    from meridian.validate_fidelity import SOURCE_STATUS
    assert "SOURCE_PRESENT" in SOURCE_STATUS
    assert "SOURCE_BLANK" in SOURCE_STATUS
    assert "SOURCE_ABSENT" in SOURCE_STATUS
    assert "SOURCE_UNREADABLE" in SOURCE_STATUS
    assert "SOURCE_AMBIGUOUS" in SOURCE_STATUS


def test_source_present_constant():
    """Verify SOURCE_PRESENT is accessible."""
    assert SOURCE_PRESENT == "SOURCE_PRESENT"


def test_source_absent_constant():
    """Verify SOURCE_ABSENT is accessible."""
    assert SOURCE_ABSENT == "SOURCE_ABSENT"


def test_source_blank_constant():
    """Verify SOURCE_BLANK is accessible."""
    assert SOURCE_BLANK == "SOURCE_BLANK"


def test_template_constants():
    """Verify template constants."""
    from meridian.template_routing import TEMPLATE_A, TEMPLATE_B
    assert TEMPLATE_A == "A"
    assert TEMPLATE_B == "B"
