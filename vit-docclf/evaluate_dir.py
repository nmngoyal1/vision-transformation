# evaluate_dir.py
import argparse, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from train_and_classify import DocDS, load_model  # reuse your dataset + loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--dir",  default="data/val", help="Labeled folder with subfolders per class")
    ap.add_argument("--classes", nargs="+", default=["certificate","claim_form","invoice","policy","renewal"])
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    model, classes_ckpt, device = load_model(args.ckpt)
    # if you change --classes, make sure order matches training
    classes = args.classes

    ds = DocDS(args.dir, classes, train=False, size=args.img_size)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(pixel_values=x).logits
            pred = logits.argmax(-1).cpu().tolist()
            # y is a list of ints from the dataset
            y_true.extend(y if isinstance(y, list) else y.tolist())
            y_pred.extend(pred)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\nAccuracy: {acc:.4f}   Macro-F1: {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))))

if __name__ == "__main__":
    main()
