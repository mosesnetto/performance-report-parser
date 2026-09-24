from .models import OCRRow


def cluster_rows(words, tolerance=None):
    if not words:
        return []
    hs=sorted(w.height for w in words if w.height>0)
    med=hs[len(hs)//2] if hs else 20
    tol=tolerance or max(5,med*0.70)
    rows=[]
    for w in sorted(words,key=lambda x:(x.cy,x.left)):
        best=None
        dist=10**9
        for row in rows:
            d=abs(w.cy-row.y)
            if d<=tol and d<dist:
                best=row;dist=d
        if best is None:
            rows.append(OCRRow(page=w.page,region=w.region,words=[w]))
        else:
            best.words.append(w)
    for r in rows:
        r.words.sort(key=lambda x:x.left)
    return rows

def row_height(row):
    vals=[w.height for w in row.words if w.height>0]
    return sorted(vals)[len(vals)//2] if vals else 20
