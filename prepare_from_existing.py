import argparse, shutil, random
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def copy_tree(src_class_dir: Path, dst_dir: Path, files):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copy2(p, dst_dir / p.name)

def main(src_root, train_root, val_root, val_ratio, seed):
    random.seed(seed)
    src_root = Path(src_root)
    classes = [d.name for d in src_root.iterdir() if d.is_dir()]
    print(f"Found classes: {classes}")

    for cls in classes:
        files = [p for p in (src_root/cls).glob("**/*") if p.suffix.lower() in IMG_EXTS]
        if not files:
            print(f"[WARN] no images in {cls}, skipping")
            continue
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_ratio)) if len(files) > 5 else 1
        val_files = files[:n_val]
        train_files = files[n_val:]

        copy_tree(src_class_dir=src_root/cls, dst_dir=Path(train_root)/cls, files=train_files)
        copy_tree(src_class_dir=src_root/cls, dst_dir=Path(val_root)/cls, files=val_files)
        print(f"{cls}: train={len(train_files)}, val={len(val_files)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   default="data/source", help="source folder with class subfolders")
    ap.add_argument("--train", default="data/prepared/train")
    ap.add_argument("--val",   default="data/prepared/val")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.src, args.train, args.val, args.val_ratio, args.seed)
