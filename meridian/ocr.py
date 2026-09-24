import csv
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from pytesseract import Output

from .models import OCRWord


def run_ocr(image_path, page, region, out_json=None, out_tsv=None, psm=11, preprocess="none"):
    img = Image.open(image_path).convert("RGB")
    if preprocess == "gray":
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(1.8)
    elif preprocess == "threshold":
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.point(lambda p: 0 if p < 200 else 255)
    try:
        data = pytesseract.image_to_data(
            img, lang="eng", config=f"--oem 1 --psm {psm}",
            output_type=Output.DICT, timeout=30
        )
    except RuntimeError as exc:
        # Tesseract can occasionally stall on malformed/complex crops.
        # Return an empty OCR result so the caller can continue to its
        # alternate OCR mode instead of hanging the whole PDF job.
        print(f"warning: OCR timeout for {region} psm={psm}: {exc}")
        data = {k: [] for k in ["text", "conf", "left", "top", "width", "height"]}
    words=[]
    for i,t in enumerate(data["text"]):
        text=(t or "").strip()
        try:
            conf=float(data["conf"][i])
        except Exception:
            conf=-1
        if not text or conf < 0:
            continue
        words.append(OCRWord(
            page=page, region=region, text=text, conf=conf,
            left=int(data["left"][i]), top=int(data["top"][i]),
            width=int(data["width"][i]), height=int(data["height"][i])
        ))

    if out_json:
        Path(out_json).parent.mkdir(parents=True,exist_ok=True)
        import json
        Path(out_json).write_text(
            json.dumps([x.to_dict() for x in words],indent=2,ensure_ascii=False),
            encoding="utf-8"
        )
    if out_tsv:
        Path(out_tsv).parent.mkdir(parents=True,exist_ok=True)
        with open(out_tsv,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f,delimiter="\t")
            w.writerow(["page","region","text","conf","left","top","width","height"])
            for x in words:
                w.writerow([x.page,x.region,x.text,x.conf,x.left,x.top,x.width,x.height])
    return words
