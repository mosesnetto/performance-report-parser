from pathlib import Path

import fitz


def render(pdf, pages, out_dir, dpi=300):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    scale=dpi/72
    matrix=fitz.Matrix(scale,scale)
    doc=fitz.open(pdf)
    result={}
    for pno in pages:
        page=doc[pno-1]
        pix=page.get_pixmap(matrix=matrix,alpha=False)
        path=out/f"page_{pno}.png"
        pix.save(str(path))
        result[pno]=path
    doc.close()
    return result
