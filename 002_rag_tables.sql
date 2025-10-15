CREATE TABLE IF NOT EXISTS rag_docs (
  doc_sha256 TEXT NOT NULL,
  chunk_id   INT  NOT NULL,
  content    TEXT NOT NULL,
  embedding  vector(384) NOT NULL,
  PRIMARY KEY (doc_sha256, chunk_id)
);
CREATE INDEX IF NOT EXISTS rag_docs_idx
  ON rag_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
