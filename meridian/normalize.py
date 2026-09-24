import re
import unicodedata

OCR_REPLACEMENTS = {
    "CYLS":"CYL5",
    "CYS":"CYL5",
    "Tes":"TC3",
    "Tcs":"TC3",
    "1c":"TC",
    "°G":"°C",
    "°c":"°C",
    "oc":"°C",
    "kw":"kW",
    "KW":"kW",
    "viv":"vlv",
    "Viv":"vlv",
    "VIV":"vlv",
}

def clean(text):
    s=unicodedata.normalize("NFKC",str(text or ""))
    s=" ".join(s.replace("|"," ").split())
    for a,b in OCR_REPLACEMENTS.items():
        s=s.replace(a,b)
    return s.strip()

def norm_for_match(text):
    s=clean(text).lower()
    s=s.replace("°","").replace("*","")
    s=re.sub(r"[^a-z0-9%./@ -]"," ",s)
    s=re.sub(r"\s+"," ",s).strip()
    return s

def slug(text):
    s=norm_for_match(text)
    return re.sub(r"[^a-z0-9]+","_",s).strip("_")
