# train_and_classify.py
import argparse, csv, os, shutil
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup

from PIL import Image, ImageSequence
from pdf2image import convert_from_path

from sklearn.metrics import f1_score, accuracy_score

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
PDF_EXT = ".pdf"

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

def tfm_train(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.2)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def tfm_eval(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def ensure_unique(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem, ext, parent = dest.stem, dest.suffix, dest.parent
    i = 1
    while True:
        cand = parent / f"{stem}({i}){ext}"
        if not cand.exists():
            return cand
        i += 1

class DocDataset(Dataset):
    """
    Expects labeled folders like:
      data/train/<class>/*.*
      data/val/<class>/*.*
    """
    def __init__(self, root: str, classes: List[str], train: bool, img_size: int = 224):
        self.root = Path(root)
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.paths, self.labels = [], []
        self.t = tfm_train(img_size) if train else tfm_eval(img_size)

        for cls in classes:
            p = self.root / cls
            if not p.exists():  # allow missing during early experiments
                continue
            for f in p.rglob("*"):
                if f.is_file() and (f.suffix.lower() in IMG_EXTS or f.suffix.lower()==PDF_EXT):
                    self.paths.append(f)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.t(pil_from_any(self.paths[i])), self.labels[i]

@torch.no_grad()
def evaluate(model, dl, device) -> Tuple[float, float]:
    y_true, y_pred = [], []
    model.eval()
    for x, y in dl:
        x = x.to(device); y = torch.tensor(y).to(device)
        pred = model(pixel_values=x).logits.argmax(-1)
        y_true.extend(y.tolist()); y_pred.extend(pred.tolist())
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    f1  = f1_score(y_true, y_pred, average='macro') if y_true else 0.0
    return acc, f1

def train_vit(train_dir: str, val_dir: str, classes: List[str],
              out_ckpt: str = "best.pt", img_size: int = 224,
              batch_size: int = 32, epochs: int = 8, lr: float = 3e-5,
              weight_decay: float = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = DocDataset(train_dir, classes, train=True, img_size=img_size)
    val_ds   = DocDataset(val_dir,   classes, train=False, img_size=img_size)
    if len(train_ds)==0 or len(val_ds)==0:
        raise RuntimeError("Empty train/val. Put images under data/train/<class> and data/val/<class>.")

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    va = DataLoader(val_ds,   batch_size=max(32,batch_size), shuffle=False, num_workers=2, pin_memory=True)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        ignore_mismatched_sizes=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, len(tr))*epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1*steps), steps)

    best_f1 = -1.0
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x,y in tr:
            x = x.to(device); y = torch.tensor(y).to(device)
            out = model(pixel_values=x, labels=y)
            loss = out.loss
            loss.backward(); opt.step(); opt.zero_grad(); sched.step()
            running += float(loss.item())

        acc, f1 = evaluate(model, va, device)
        print(f"[ep {ep}] loss={running/len(tr):.4f}  val_acc={acc:.4f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"state_dict": model.state_dict(), "classes": classes}, out_ckpt)
            print(f"  saved best → {out_ckpt} (macroF1={best_f1:.4f})")

def load_model(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(classes),
        ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model, classes, device

@torch.no_grad()
def predict(model, x, device):
    logits = model(pixel_values=x.to(device)).logits
    probs = logits.softmax(-1)
    conf, pred = probs.max(dim=-1)
    return pred.item(), conf.item()

def classify_folder(ckpt_path: str, input_dir: str, out_dir: str = "sorted_output",
                    img_size: int = 224, move: bool = False):
    model, classes, device = load_model(ckpt_path)
    tf = tfm_eval(img_size)
    in_dir = Path(input_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    for c in classes: (out_dir_p / c).mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.rglob("*")
             if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower()==PDF_EXT)]
    if not files:
        print("No images/PDFs found."); return

    csv_path = out_dir_p / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["file","pred_label","confidence"])
        for fp in files:
            try:
                x = tf(pil_from_any(fp)).unsqueeze(0)
                idx, conf = predict(model, x, device)
                label = classes[idx]
                writer.writerow([str(fp), label, f"{conf:.4f}"])
                dst = ensure_unique(out_dir_p / label / fp.name)
                shutil.move(str(fp), dst) if move else shutil.copy2(str(fp), dst)
                print(f"{fp.name:40s} → {label:22s}  ({conf:.2%})")
            except Exception as e:
                print(f"[skip] {fp} — {e}")

    print(f"\nDone. CSV: {csv_path}")
    print(f"Sorted files under: {out_dir_p.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="Train ViT (with aug) and/or classify a mixed folder.")
    ap.add_argument("--mode", choices=["train","classify","both"], required=True)
    # training
    ap.add_argument("--train_dir", default="data/train")
    ap.add_argument("--val_dir",   default="data/val")
    ap.add_argument("--classes",   nargs="+",
                    default=["certificate","claim_form","invoice","policy","renewal"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--ckpt", default="best.pt")
    # inference
    ap.add_argument("--input_dir", help="Folder with mixed docs for classification")
    ap.add_argument("--out_dir", default="sorted_output")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()

    if args.mode in ("train","both"):
        train_vit(args.train_dir, args.val_dir, args.classes, args.ckpt,
                  args.img_size, args.batch_size, args.epochs, args.lr, args.weight_decay)

    if args.mode in ("classify","both"):
        if not args.input_dir:
            raise SystemExit("--input_dir is required for classify/both")
        classify_folder(args.ckpt, args.input_dir, args.out_dir, args.img_size, args.move)

if __name__ == "__main__":
    main()
