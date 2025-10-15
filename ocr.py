from __future__ import annotations
import os
from pathlib import Path
import cv2
import pytesseract
from dotenv import load_dotenv

load_dotenv()
exe = os.getenv("TESSERACT_EXE")
if exe and Path(exe).exists():
    pytesseract.pytesseract.tesseract_cmd = exe

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def ocr_image(path: str, lang: str = "eng") -> str:
    """
    Simple, robust OCR pipeline for images using Tesseract.
    - converts to grayscale
    - adaptive threshold
    - slight denoise
    """
    p = Path(path)
    if p.suffix.lower() not in IMG_EXTS:
        raise ValueError(f"Not an image: {path}")

    img = cv2.imread(str(p))
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    config = "--oem 3 --psm 6"  # LSTM, block of text
    text = pytesseract.image_to_string(thr, lang=lang, config=config)
    return text.strip()
