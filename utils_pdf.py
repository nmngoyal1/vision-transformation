# utils_pdf.py
from pdf2image import convert_from_path
from pathlib import Path

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 200):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    for i, p in enumerate(pages):
        fp = out / f"{Path(pdf_path).stem}_p{i+1}.png"
        p.save(fp)
        paths.append(str(fp))
    return paths
