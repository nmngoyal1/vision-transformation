# streamlit_app.py
# Run: streamlit run streamlit_app.py

import shutil
from pathlib import Path
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageSequence
from pdf2image import convert_from_path
from torchvision import transforms
from transformers import ViTForImageClassification

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
PDF_EXT = ".pdf"

def tfm(img_size=224):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", num_labels=len(classes), ignore_mismatched_sizes=True
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, device

def pil_from_any(path: Path):
    suf = path.suffix.lower()
    if suf == PDF_EXT:
        return convert_from_path(str(path), dpi=200)[0].convert("RGB")
    if suf in IMG_EXTS:
        im = Image.open(path)
        if getattr(im, "is_animated", False):
            im = next(ImageSequence.Iterator(im))
        return im.convert("RGB")
    raise ValueError(f"Unsupported type: {path}")

@torch.no_grad()
def predict(model, x, device):
    logits = model(pixel_values=x.to(device)).logits
    probs = logits.softmax(-1)
    conf, pred = probs.max(dim=-1)
    return pred.item(), conf.item()

st.set_page_config(page_title="ViT Doc Classifier", layout="centered")
st.title("📄 ViT Document Classifier — Folder Sorter")

with st.sidebar:
    st.markdown("**Settings**")
    default_input = r"C:\Users\91807\Desktop\hackathon\hospital\vision transformer\images"
    input_dir = st.text_input("Input folder (mixed docs)", value=default_input)
    out_dir = st.text_input("Output folder", value="sorted_output")
    model_path = st.text_input("Model checkpoint (best.pt)", value="best.pt")
    img_size = st.slider("Image size", 192, 384, 224, step=32)
    move_files = st.checkbox("Move files instead of copy", value=False)
    go = st.button("Start classification")

st.info("Tip: If you use PDFs, install Poppler and add its `bin` to PATH so `pdf2image` works on Windows.")

if go:
    try:
        model, classes, device = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    tf = tfm(img_size)
    in_dir = Path(input_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (out_dir_p / c).mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.rglob('*') if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower()==PDF_EXT)]
    if not files:
        st.warning("No images/PDFs found in the input folder.")
        st.stop()

    rows = []
    prog = st.progress(0)
    for i, fp in enumerate(files, 1):
        try:
            img = pil_from_any(fp)
            x = tf(img).unsqueeze(0)
            pred_idx, conf = predict(model, x, device)
            label = classes[pred_idx]
            # copy/move
            dst = out_dir_p / label / fp.name
            if dst.exists():
                stem, ext = dst.stem, dst.suffix
                k = 1
                while dst.exists():
                    dst = dst.with_name(f"{stem}({k}){ext}")
                    k += 1
            if move_files:
                shutil.move(str(fp), dst)
            else:
                shutil.copy2(str(fp), dst)
            rows.append({"file": str(fp), "pred_label": label, "confidence": round(conf,4)})
        except Exception as e:
            rows.append({"file": str(fp), "pred_label": "__error__", "confidence": 0.0, "error": str(e)})
        prog.progress(i/len(files))

    df = pd.DataFrame(rows)
    st.success("Done!")
    st.dataframe(df, use_container_width=True)
    counts = df["pred_label"].value_counts().rename_axis("label").reset_index(name="count")
    st.caption("Summary")
    st.table(counts)

    # save CSV + allow download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions.csv", data=csv_bytes, file_name="predictions.csv")

    st.write("Output folder:", str(out_dir_p.resolve()))
