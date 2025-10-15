import argparse, time, shutil
from pathlib import Path
from PIL import Image
from app.vit_infer import VitClassifier
import yaml

CFG = yaml.safe_load(open("app/config.yaml","r"))
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def main(in_dir: str):
    ts = time.strftime("%Y-%m-%d-%H-%M-%S")
    out_root = Path(CFG["paths"]["run_root"]) / ts / CFG["paths"]["sorted_dir_name"]
    out_root.mkdir(parents=True, exist_ok=True)

    clf = VitClassifier()
    files = [p for p in Path(in_dir).glob("**/*") if p.suffix.lower() in IMG_EXTS]
    if not files:
        print(f"No images found in {in_dir}")
        return

    for p in files:
        img = Image.open(p)
        preds = clf.predict(img, topk=1)
        cls = preds[0][0]
        dest = out_root / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / p.name)
        print(f"{p.name} -> {cls}")

    print(f"\nSorted output: {out_root}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Folder with images")
    args = ap.parse_args()
    main(args.in_dir)
