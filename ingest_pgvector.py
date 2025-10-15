import argparse, hashlib
from pathlib import Path
from pypdf import PdfReader
from app.rag.chunking import simple_chunk
from app.rag.embeddings import embed_texts
from app.db import upsert_chunk

def sha256_text(t: str) -> str: return hashlib.sha256(t.encode("utf-8")).hexdigest()

def extract_text_pdf(p: Path) -> str:
    r = PdfReader(str(p))
    return "\n".join(page.extract_text() or "" for page in r.pages)

def main(in_dir: str):
    files = list(Path(in_dir).glob("**/*.pdf"))
    if not files:
        print("No PDFs found."); return
    for p in files:
        text = extract_text_pdf(p)
        chunks = simple_chunk(text)
        embs = embed_texts(chunks)
        doc_sha = sha256_text(str(p.resolve()))
        for i, (c, e) in enumerate(zip(chunks, embs)):
            upsert_chunk(doc_sha, i, c, e)
        print(f"Ingested {p.name} with {len(chunks)} chunks")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data/raw/pdfs")
    args = ap.parse_args()
    main(args.in_dir)
