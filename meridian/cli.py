import argparse
from pathlib import Path

from .crop_pipeline import process as process_crops
from .pdf_pipeline import process as process_pdf


def parse_pages(s):
    if not s or s.lower() in {"auto", "all"}:
        return None
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(description="Performance Report Parser")
    p.add_argument("--mode", choices=["crops", "pdf"], default="pdf")
    p.add_argument("--crops")
    p.add_argument("--pdf")
    p.add_argument("--pages", default="auto")
    p.add_argument("--output", default="output")
    p.add_argument("--psm", type=int, default=11)
    p.add_argument("--engine", choices=["v3", "v4"], default="v3")
    args = p.parse_args()

    base = Path(__file__).resolve().parent.parent
    fields_yaml = base / "config" / "fields.yaml"

    if args.mode == "crops":
        if not args.crops:
            p.error("--crops is required for crops mode")
        summary = process_crops(args.crops, fields_yaml, args.output, args.psm, (13, 14), engine=args.engine)
    else:
        if not args.pdf:
            p.error("--pdf is required for pdf mode")
        summary = process_pdf(args.pdf, parse_pages(args.pages), fields_yaml, args.output, 300, args.psm, engine=args.engine)

    print(f"\n=== Performance Report Parser v3 (engine={args.engine}) ===")
    for k, v in summary.items():
        print(f"{k:22}: {v}")


if __name__ == "__main__":
    main()
