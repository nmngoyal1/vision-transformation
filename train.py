# train.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, get_cosine_schedule_with_warmup
from pathlib import Path
from PIL import Image, ImageSequence
from pdf2image import convert_from_path
from classes import CLASSES

CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif"}

def pil_from_any(p: Path):
    if p.suffix.lower()==".pdf":
        return convert_from_path(str(p), dpi=200)[0].convert("RGB")
    im = Image.open(p)
    if getattr(im, "is_animated", False):
        im = next(ImageSequence.Iterator(im))
    return im.convert("RGB")

class DocDS(Dataset):
    def __init__(self, root, class_to_idx, train=True, size=224):
        self.paths,self.labels=[],[]
        self.root = Path(root)
        self.t = transforms.Compose([
            transforms.Resize(int(size*1.2)),
            transforms.RandomResizedCrop(size,scale=(0.8,1.0)),
            transforms.RandomRotation(3),
            transforms.ColorJitter(0.1,0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ]) if train else transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        for cls, idx in class_to_idx.items():
            for p in (self.root/cls).rglob("*"):
                if p.suffix.lower() in IMG_EXTS or p.suffix.lower()==".pdf":
                    self.paths.append(p); self.labels.append(idx)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.t(pil_from_any(self.paths[i])), self.labels[i]

def evaluate(model, dl, device):
    model.eval(); import numpy as np
    correct,total=0,0
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device); y = torch.tensor(y).to(device)
            pred = model(pixel_values=x).logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/total if total else 0.0

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = DocDS("data/train", CLASS_TO_IDX, train=True)
    val_ds   = DocDS("data/val",   CLASS_TO_IDX, train=False)
    tr = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    va = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", num_labels=len(CLASSES),
        ignore_mismatched_sizes=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    steps = max(1, len(tr))*8
    sched = get_cosine_schedule_with_warmup(opt, int(0.1*steps), steps)

    best, path = 0.0, "best.pt"
    for ep in range(1,9):
        model.train()
        for x,y in tr:
            x = x.to(device); y = torch.tensor(y).to(device)
            out = model(pixel_values=x, labels=y)
            out.loss.backward(); opt.step(); opt.zero_grad(); sched.step()
        acc = evaluate(model, va, device)
        print(f"epoch {ep} val_acc={acc:.4f}")
        if acc>best:
            best=acc
            torch.save({"state_dict":model.state_dict(), "classes":CLASSES}, path)
            print("saved", path)
