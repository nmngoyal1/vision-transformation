# app/rag/query.py
import psycopg2, os
from .fusion import rrf, weighted_scores

def connect():
    return psycopg2.connect(os.environ["PG_DSN"])

def search_vector(cur, q_emb, topk=20):
    cur.execute("""
      SELECT id, 1 - (embedding <=> %s) AS score
      FROM rag_docs
      ORDER BY embedding <=> %s
      LIMIT %s
    """, (q_emb, q_emb, topk))
    return [(r[0], float(r[1])) for r in cur.fetchall()]

def search_bm25(cur, text_query, topk=20):
    # assuming you added a tsvector column + index on rag_docs(content_tsv)
    cur.execute("""
      SELECT id, ts_rank(content_tsv, plainto_tsquery(%s)) AS score
      FROM rag_docs
      WHERE content_tsv @@ plainto_tsquery(%s)
      ORDER BY score DESC
      LIMIT %s
    """, (text_query, text_query, topk))
    return [(r[0], float(r[1])) for r in cur.fetchall()]

def query(q_text, q_emb, mode="rrf", weights=None, k=60, topk_each=20):
    with connect() as con, con.cursor() as cur:
        vec = search_vector(cur, q_emb, topk_each)
        bm25 = search_bm25(cur, q_text, topk_each)

    if mode == "rrf":
        rank_lists = [
            [doc_id for doc_id, _ in sorted(vec, key=lambda x: -x[1])],
            [doc_id for doc_id, _ in sorted(bm25, key=lambda x: -x[1])]
        ]
        fused = rrf(rank_lists, k=k)  # [(doc_id, rrf_score), ...]
        return fused

    # weighted re-rank on normalized scores
    # (ensure both lists are normalized 0..1 before combining, if needed)
    candidates = {"vector": vec, "bm25": bm25}
    weights = weights or {"vector": 0.6, "bm25": 0.4}
    return weighted_scores(candidates, weights)
