import streamlit as st
from PIL import Image
from app.vit_infer import VitClassifier
import yaml, io, time
from pathlib import Path

CFG = yaml.safe_load(open("app/config.yaml","r"))
st.set_page_config(page_title="ViT Doc Classifier", layout="centered")

if "clf" not in st.session_state:
    st.session_state.clf = VitClassifier()

st.title("📄 ViT Document Classifier")
file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","tiff"])
if file:
    img = Image.open(io.BytesIO(file.read()))
    st.image(img, caption="Preview", use_container_width=True)
    preds = st.session_state.clf.predict(img, topk=CFG["vit"]["topk"])
    st.subheader("Predictions")
    for cls, prob in preds:
        st.write(f"- **{cls}**: {prob:.3f}")

    if st.button("Save to sorted_output"):
        ts = time.strftime("%Y-%m-%d-%H-%M-%S")
        out = Path(CFG["paths"]["run_root"])/ts/CFG["paths"]["sorted_dir_name"]/preds[0][0]
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / f"upload_{int(time.time())}.png")
        st.success(f"Saved under {out}")
