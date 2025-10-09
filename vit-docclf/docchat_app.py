# docchat_app.py
# Chat with documents in a folder (images + PDFs) using local OCR + embeddings + QA.
# Works fully offline (no API keys needed). Sources are shown for every answer.

import os, io, re, json, time, hashlib
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import joblib
import streamlit as st
from PIL import Image
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline

# ---------- OPTIONAL: hardcode tesseract path on Windows ----------
# If you get "TesseractNotFoundError", uncomment and set your install path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- Configuration ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast CPU
QA_MODEL_NAME    = "deepset/roberta-base-squad2"             # extractive local QA
CHUNK_CHARS      = 800
CHUNK_OVERLAP    = 120
TOP_K_RETRIEVE   = 5

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
PDF_EXT  = ".pdf"

# ---------- Utilities ----------
def text_cleanup(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def file_fingerprint(p: Path) -> str:
    stat = p.stat()
    return f"{p}|{stat.st_size}|{int(stat.st_mtime)}"

def ocr_image(img: Image.Image) -> str:
    # PSM 6 works well for dense documents
    return pytesseract.image_to_string(img, config="--psm 6")

def extract_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_number, text) for a PDF. Try text first, OCR fallback."""
    out = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                txt = text_cleanup(txt)
                out.append((i, txt))
    except Exception:
        out = []  # fall through to OCR

    # If PDF text is empty/weak, OCR each page
    if sum(len(t) for _, t in out) < 50:
        try:
            images = convert_from_path(str(pdf_path), dpi=250)
            out = []
            for i, img in enumerate(images, start=1):
                txt = text_cleanup(ocr_image(img.convert("RGB")))
                out.append((i, txt))
        except Exception as e:
            out = []
    return out

def extract_from_image(img_path: Path) -> List[Tuple[int, str]]:
    try:
        img = Image.open(img_path).convert("RGB")
        txt = text_cleanup(ocr_image(img))
        return [(1, txt)]
    except Exception:
        return []

def chunk_text(txt: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    txt = txt.strip()
    if not txt:
        return []
    chunks = []
    start = 0
    while start < len(txt):
        end = min(len(txt), start + size)
        chunk = txt[start:end]
        chunks.append(chunk)
        if end == len(txt):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------- Index builder ----------
def build_or_load_index(base_dir: Path, index_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    """
    Build embeddings + metadata for all files under base_dir.
    Cache to index_dir. If nothing changed, load from cache.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path  = index_dir / "meta.json"
    vecs_path  = index_dir / "embeddings.npy"
    sign_path  = index_dir / "signature.txt"

    # compute signature of current folder contents
    sig_hasher = hashlib.sha1()
    files = []
    for p in base_dir.rglob("*"):
        if p.is_file() and (p.suffix.lower() in IMG_EXTS or p.suffix.lower() == PDF_EXT):
            sig_hasher.update(file_fingerprint(p).encode())
            files.append(p)
    current_sig = sig_hasher.hexdigest()

    # if cached and unchanged → load
    if meta_path.exists() and vecs_path.exists() and sign_path.exists():
        if sign_path.read_text() == current_sig:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vecs = np.load(vecs_path)
            return vecs, meta

    # otherwise, (re)build
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    meta: List[Dict] = []
    all_texts: List[str] = []

    for f in files:
        if f.suffix.lower() == PDF_EXT:
            pages = extract_from_pdf(f)
        else:
            pages = extract_from_image(f)

        for (page_no, txt) in pages:
            if not txt or len(txt) < 10:
                continue
            for i, chunk in enumerate(chunk_text(txt)):
                meta.append({
                    "file": str(f),
                    "page": page_no,
                    "chunk_id": i,
                    "text": chunk
                })
                all_texts.append(chunk)

    if not all_texts:
        return np.zeros((0, 384), dtype="float32"), meta

    vecs = embedder.encode(all_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(vecs_path, vecs)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    sign_path.write_text(current_sig)
    return vecs, meta

# ---------- Retrieval + QA ----------
def top_k(query: str, vecs: np.ndarray, meta: List[Dict], k: int = TOP_K_RETRIEVE) -> List[Dict]:
    if vecs.shape[0] == 0:
        return []
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    q = embedder.encode([query], normalize_embeddings=True)
    # cosine distance via NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(meta)), metric="cosine")
    nn.fit(vecs)
    dist, idx = nn.kneighbors(q)
    idx = idx[0].tolist()
    ranked = []
    for i in idx:
        item = dict(meta[i])
        item["score"] = 1 - float(dist[0][idx.index(i)])  # cosine similarity
        ranked.append(item)
    return ranked

@st.cache_resource(show_spinner=False)
def get_qa_pipeline():
    return pipeline("question-answering", model=QA_MODEL_NAME)

def answer_query(question: str, retrieved: List[Dict]) -> Tuple[str, List[Dict]]:
    """Run extractive QA over top contexts; pick best-scoring span."""
    if not retrieved:
        return "I couldn't find relevant text in your documents.", []
    qa = get_qa_pipeline()
    best_ans = {"answer": "", "score": -1.0, "ctx": None}
    for item in retrieved:
        ctx = item["text"]
        try:
            out = qa(question=question, context=ctx)
        except Exception:
            continue
        if out and out.get("score", 0) > best_ans["score"]:
            best_ans = {"answer": out.get("answer", ""), "score": float(out.get("score", 0)), "ctx": item}
    if best_ans["score"] < 0:
        return "I couldn't extract an answer. Try rephrasing your question.", retrieved[:3]
    return best_ans["answer"], [best_ans["ctx"]] + [r for r in retrieved if r is not best_ans["ctx"]][:2]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="DocChat — chat with your documents", layout="wide")
st.title("🗂️ DocChat — chat with your documents")

with st.sidebar:
    st.subheader("Index settings")
    default_base = r"C:\Users\91807\Desktop\hackathon\hospital\vision transformer\vit-docclf\sorted_output"
    base_dir = Path(st.text_input("Documents folder", value=default_base))
    index_dir = Path(st.text_input("Index cache folder", value="doc_index"))
    rebuild = st.checkbox("Force rebuild index", value=False)
    st.caption("Tip: point to your sorted_output folder (it scans subfolders too).")

if rebuild and index_dir.exists():
    for f in index_dir.rglob("*"):
        try: f.unlink()
        except: pass
    try: index_dir.rmdir()
    except: pass

with st.spinner("Building / loading index…"):
    vecs, meta = build_or_load_index(base_dir, index_dir)

if len(meta) == 0:
    st.warning("No text found. Make sure your folder has PDFs/images and Tesseract/Poppler are installed.")
else:
    st.success(f"Indexed {len(meta)} text chunks from {len(set(m['file'] for m in meta))} files.")

# Chat box
if "history" not in st.session_state: st.session_state["history"] = []

q = st.text_input("Ask a question about your documents…", placeholder="e.g., What is the policy number and expiry date?")
if st.button("Ask") and q.strip():
    with st.spinner("Searching…"):
        retrieved = top_k(q.strip(), vecs, meta, k=TOP_K_RETRIEVE)
    with st.spinner("Answering…"):
        answer, sources = answer_query(q.strip(), retrieved)
    st.session_state["history"].append({"q": q.strip(), "a": answer, "sources": sources})

# Display history
for turn in st.session_state["history"][::-1]:
    st.markdown(f"**You:** {turn['q']}")
    st.markdown(f"**Assistant:** {turn['a']}")
    if turn["sources"]:
        st.caption("Sources:")
        for s in turn["sources"]:
            st.write(f"• {Path(s['file']).name} (page {s['page']}), score={s.get('score',0):.2f}")
            with st.expander("View chunk"):
                st.code(s["text"][:1200])
    st.markdown("---")