from pathlib import Path

import yaml

from .normalize import norm_for_match


def load_fields(path):
    data=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return data.get("fields",[])

def build_alias_index(fields):
    idx=[]
    for f in fields:
        aliases=[f.get("name","")]+f.get("aliases",[])
        aliases=[norm_for_match(x) for x in aliases if x]
        idx.append((f,aliases))
    return idx
