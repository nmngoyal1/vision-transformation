# classify_folder.py
import argparse, csv, shutil
from pathlib import Path
from typing import Tuple
import torch
from PIL import Image, ImageSequence
from pdf2image import convert_from_path
from torchvision import transforms
from transformers import ViTForImageClassification

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
PDF_EXT = ".pdf"

def load_checkpoint(model_path: str) -> Tuple[ViTForImageClassification, list, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        ignore_mismatched_sizes=True,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, device

def pil_from_any(path: Path) -> Image.Image:
    suf = path.suffix.lower()
    if suf == PDF_EXT:
        return convert_from_path(str(path), dpi=200)[0].convert("RGB")
    if suf in IMG_EXTS:
        im = Image.open(path)
        if getattr(im, "is_animated", False):
            im = next(ImageSequence.Iterator(im))
        return im.convert("RGB")
    raise ValueError(f"Unsupported file type: {path}")

def tfm(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])

@torch.no_grad()
def predict(model, x, device):
    logits = model(pixel_values=x.to(device)).logits
    probs = logits.softmax(-1)
    conf, pred = probs.max(dim=-1)
    return pred.item(), conf.item()

def ensure_unique(dest: Path) -> Path:
    if not dest.exists(): return dest
    stem, ext, parent = dest.stem, dest.suffix, dest.parent
    i = 1
    while True:
        cand = parent / f"{stem}({i}){ext}"
        if not cand.exists(): return cand
        i += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best.pt", help="Path to ViT checkpoint")
    ap.add_argument("--input_dir", required=True, help="Folder with mixed docs")
    ap.add_argument("--out_dir", default="sorted_output", help="Where to copy/move")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    model, classes, device = load_checkpoint(args.model)
    transform = tfm(args.img_size)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in classes: (out_dir / c).mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.rglob("*")
             if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower()==PDF_EXT)]
    if not files:
        print("No images/PDFs found."); return

    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["file","pred_label","confidence"])
        for fp in files:
            try:
                x = transform(pil_from_any(fp)).unsqueeze(0)
                pred_idx, conf = predict(model, x, device)
                label = classes[pred_idx]
                writer.writerow([str(fp), label, f"{conf:.4f}"])
                dst = ensure_unique(out_dir / label / fp.name)
                shutil.move(str(fp), dst) if args.move else shutil.copy2(str(fp), dst)
                print(f"{fp.name:40s} → {label:22s} ({conf:.2%})")
            except Exception as e:
                print(f"[skip] {fp} — {e}")

    print(f"\nDone. CSV: {csv_path}")
    print(f"Sorted files under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
