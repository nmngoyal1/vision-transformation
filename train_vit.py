from __future__ import annotations
import argparse
from pathlib import Path
import torch, timm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import yaml, json

CFG = yaml.safe_load(open("app/config.yaml","r"))

def get_loaders(train_dir, val_dir, img_size, bs=16):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=tfm)
    val_ds   = datasets.ImageFolder(val_dir,   transform=tfm)
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2),
            DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2),
            train_ds.classes)

def evaluate(model, dl, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0

def main(train_dir, val_dir, out, epochs, lr, bs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, classes = get_loaders(train_dir, val_dir, CFG["vit"]["image_size"], bs)
    model = timm.create_model(CFG["vit"]["model_name"], pretrained=True, num_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc, outp = 0.0, Path(out)
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
        val_acc = evaluate(model, val_dl, device)
        print(f"Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), outp)
    with open("app/labels.json","w") as f:
        json.dump({"classes": classes}, f, indent=2)
    print(f"Saved best weights to {outp} (val_acc={best_acc:.4f})")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="data/prepared/train")
    ap.add_argument("--val_dir",   default="data/prepared/val")
    ap.add_argument("--out",       default="models/vit/best.pt")
    ap.add_argument("--epochs",    type=int, default=10)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--bs",        type=int, default=16)
    args = ap.parse_args()
    main(args.train_dir, args.val_dir, args.out, args.epochs, args.lr, args.bs)
