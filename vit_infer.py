from __future__ import annotations
import json, yaml, sys
from pathlib import Path
from typing import List, Tuple
import torch, timm
from PIL import Image

CFG = yaml.safe_load(open("app/config.yaml", "r"))

def _load_labels(path: str) -> list[str]:
    with open(path, "r") as f:
        return json.load(f)["classes"]

class VitClassifier:
    def __init__(self, weights_path: str | None = None, labels_path: str | None = None, image_size: int | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.labels = _load_labels(labels_path or CFG["paths"]["labels_json"])
        model_name = CFG["vit"]["model_name"]
        num_classes = len(self.labels)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        wpath = weights_path or CFG["vit"]["weights_path"]
        wfile = Path(wpath)
        if wfile.exists():
            self.model.load_state_dict(torch.load(str(wfile), map_location="cpu"))
        else:
            print(f"[WARN] Weights not found at {wpath}. Using pretrained backbone with random head.", flush=True)
        self.model.eval().to(self.device)
        self.image_size = image_size or CFG["vit"]["image_size"]

    def _prep(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.image_size, self.image_size))
        x = torch.from_numpy(
            (torch.tensor(list(img.getdata())).float()/255.0)
            .view(img.size[1], img.size[0], 3).permute(2,0,1).numpy()
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        x = (x - mean) / std
        return x.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, img: Image.Image, topk: int = 3) -> List[Tuple[str, float]]:
        x = self._prep(img)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(topk, len(self.labels))
        vals, idxs = torch.topk(probs, k=k)
        return [(self.labels[i.item()], float(v.item())) for v, i in zip(vals, idxs)]

def _cli():
    if len(sys.argv) < 2:
        print("Usage: python -m app.vit_infer <image_path>")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"Not found: {p}"); sys.exit(2)
    clf = VitClassifier()
    img = Image.open(p)
    preds = clf.predict(img, topk=CFG["vit"]["topk"])
    for cls, prob in preds:
        print(f"{cls}: {prob:.3f}")

if __name__ == "__main__":
    _cli()
