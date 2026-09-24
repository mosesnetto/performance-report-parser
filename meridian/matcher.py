from difflib import SequenceMatcher

from .normalize import norm_for_match


def similarity(a,b):
    a=norm_for_match(a); b=norm_for_match(b)
    if not a or not b: return 0.0
    if a==b: return 1.0
    if a in b or b in a:
        return min(len(a),len(b))/max(len(a),len(b))
    return SequenceMatcher(None,a,b).ratio()

def match_field(row_text, field_index):
    r=norm_for_match(row_text)
    best=None; best_score=0.0
    for field,aliases in field_index:
        for alias in aliases:
            if not alias: continue
            # Strong exact/contained match. This prevents short generic fields
            # such as "Density" or "Temp. deviation" from stealing other rows.
            if r==alias:
                return field,1.0
            if alias in r:
                score=min(0.995, 0.80 + 0.19*(len(alias)/max(1,len(r))))
            else:
                score=SequenceMatcher(None,r,alias).ratio()
                if score < 0.88:
                    continue
            if score>best_score:
                best,best_score=field,score
    return best,best_score
