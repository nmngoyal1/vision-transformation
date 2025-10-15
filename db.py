import os, psycopg2, yaml
from contextlib import contextmanager
from dotenv import load_dotenv

CFG = yaml.safe_load(open("app/config.yaml","r"))
load_dotenv()

@contextmanager
def get_conn():
    dsn = os.getenv(CFG["pg"]["dsn_env"])
    if not dsn:
        raise RuntimeError("PG_DSN not set in environment (.env)")
    conn = psycopg2.connect(dsn)
    try:
        yield conn
    finally:
        conn.close()

def upsert_chunk(doc_sha: str, chunk_id: int, content: str, embedding: list[float]):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO rag_docs (doc_sha256, chunk_id, content, embedding)
            VALUES (%s,%s,%s,%s)
            ON CONFLICT (doc_sha256, chunk_id)
            DO UPDATE SET content=EXCLUDED.content, embedding=EXCLUDED.embedding
        """, (doc_sha, chunk_id, content, embedding))
        conn.commit()
