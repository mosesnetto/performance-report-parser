from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class OCRWord:
    page: int
    region: str
    text: str
    conf: float
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self): return self.left + self.width
    @property
    def bottom(self): return self.top + self.height
    @property
    def cx(self): return self.left + self.width / 2
    @property
    def cy(self): return self.top + self.height / 2

    def to_dict(self):
        d = asdict(self)
        d.update({"right":self.right,"bottom":self.bottom,"cx":self.cx,"cy":self.cy})
        return d

@dataclass
class OCRRow:
    page: int
    region: str
    words: List[OCRWord] = field(default_factory=list)

    @property
    def text(self):
        return " ".join(w.text for w in sorted(self.words,key=lambda x:x.left))

    @property
    def y(self):
        return sum(w.cy for w in self.words)/len(self.words) if self.words else 0

@dataclass
class Cell:
    field_id: str
    field_name: str
    page: int
    region: str
    column: str
    raw_text: str
    value: Optional[float]
    unit: str
    confidence: float
    bbox: List[int]
    source_row: str
    flags: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

@dataclass
class FieldRecord:
    field_id: str
    field_name: str
    page: int
    region: str
    unit: str
    found: bool
    row_text: str
    row_confidence: float
    cells: List[Cell] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    # Canonical field-level value: propagated from the single VALUE cell when
    # the field carries exactly one non-absent cell.  For multi-column or
    # absent fields these remain None; the cell-level raw_text is always
    # the authoritative source.
    raw_text: Optional[str] = None
    value: Optional[float] = None

    def to_dict(self):
        d = asdict(self)
        d["cells"] = [c.to_dict() for c in self.cells]
        return d
