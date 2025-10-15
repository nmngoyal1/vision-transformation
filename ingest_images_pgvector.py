import argparse, hashlib
from pathlib import Path
from app.rag.ocr import ocr_image, IMG_EXTS
from app.rag.chunking import simple_chunk
from app.rag.embeddings import embed_texts
from app.db import get_conn

def sha256_text(t: str) -> str: 
    import hashlib; return hashlib.sha256(t.encode("utf-8")).hexdigest()

def main(in_dir: str, lang: str, size: int, overlap: int, min_chars: int, dry: bool):
    in_dir = Path(in_dir)
    files = [p for p in in_dir.glob("**/*") if p.suffix.lower() in IMG_EXTS]
    if not files:
        print("No images found."); return

    for p in files:
        text = ocr_image(str(p), lang=lang)
        if len(text) < min_chars:
            print(f"[SKIP] {p.name}: too little text ({len(text)} chars)")
            continue

        chunks = simple_chunk(text, size=size, overlap=overlap)
        embs = embed_texts(chunks)
        doc_sha = sha256_text(str(p.resolve()))

        if dry:
            print(f"[DRY] {p.name} -> {len(chunks)} chunks")
            continue

        with get_conn() as conn, conn.cursor() as cur:
            # ensure docs row
            cur.execute("""
              INSERT INTO docs (doc_sha256, source_path, source_name, mime_type)
              VALUES (%s,%s,%s,%s)
              ON CONFLICT (doc_sha256) DO NOTHING
            """, (doc_sha, str(p.resolve()), p.name, "image"))
            # upsert chunks
            for i, (c, e) in enumerate(zip(chunks, embs)):
                cur.execute("""
                  INSERT INTO rag_docs (doc_sha256, chunk_id, content, embedding)
                  VALUES (%s,%s,%s,%s)
                  ON CONFLICT (doc_sha256, chunk_id)
                  DO UPDATE SET content=EXCLUDED.content, embedding=EXCLUDED.embedding
                """, (doc_sha, i, c, e))
            conn.commit()
        print(f"[OK] {p.name}: {len(chunks)} chunks -> pgvector")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data/raw/images_final")
    ap.add_argument("--lang", default="eng")
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    main(args.in_dir, args.lang, args.size, args.overlap, args.min_chars, args.dry)
