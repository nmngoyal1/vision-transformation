CREATE TABLE IF NOT EXISTS docs (
  doc_sha256 TEXT PRIMARY KEY,
  source_path TEXT NOT NULL,
  source_name TEXT NOT NULL,
  mime_type   TEXT NOT NULL DEFAULT 'image',
  created_at  TIMESTAMP DEFAULT now()
);
