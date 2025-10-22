1. Training the Vision Transformer

You have two different training pipelines:

🧩 Option A — HuggingFace ViT (train.py)

Loads the ViT-Base model from HuggingFace.

Automatically reads images and the first page of each PDF from your training folders.

Applies preprocessing, augmentation, and fine-tuning.

Evaluates accuracy and F1-score per epoch, then saves the best model (best.pt) along with the list of class names.

Use this when you want a HuggingFace transformer-centric approach.

🧠 Option B — timm-based ViT (train_vit.py)

Uses the timm library (great for speed and simplicity).

Reads parameters from config.yaml — model name, image size, learning rate, etc.

Trains the model using AdamW optimizer, saves weights at models/vit/best.pt, and writes class labels to app/labels.json.

Perfect for fast, lightweight training on custom datasets.

2. Inference and Classification

Once you have a trained model, there are multiple ways to use it:

🧮 vit_infer.py

A reusable Python class (VitClassifier) that loads the model and performs top-k prediction on any image.

It handles resizing, normalization, and returns class probabilities.

This is what powers both your Streamlit app and your batch classifier.

🗂️ classify_folder.py

Takes an entire folder of images, classifies each one, and copies them into sub-folders based on predicted labels.

Great for organizing raw, mixed-category image sets.

💻 train_and_classify.py

An “all-in-one” pipeline.

Can train, classify, or do both in a single run — controlled by the --mode argument.

Tracks macro-F1 score, saves predictions to CSV, and can move or copy files into result folders.

Handles both images and PDFs, so it’s your most flexible script.

3. Interactive Web UI (Streamlit)
🌐 streamlit_app.py

Launches a clean Streamlit interface.

Upload an image (or scanned document) and instantly see the model’s prediction probabilities.

You can optionally store results in timestamped folders (runs/<timestamp>/sorted_output/<predicted_class>).

Designed for quick demos, business showcases, or client-facing prototypes.

4. Utilities and Preprocessing
📂 prepare_from_existing.py

Automatically splits a single dataset folder into training and validation sets with a ratio you define.

📑 utils_pdf.py

Converts each page of a PDF into an image — useful for datasets or inference on multi-page files.

🧾 classes.py and labels.json

Contain the list of categories used during training.

train.py uses classes.py, while the timm-based setup uses labels.json.

⚙️ config.yaml

The project’s central configuration file.

Stores model parameters, weight paths, top-k prediction count, label locations, and RAG/pgvector settings.

The entire pipeline reads from here — so editing this file changes the app’s behavior consistently.

5. OCR + RAG (Retrieval-Augmented Generation) Integration

This is the next-level module of your project — where image classification meets semantic search.

📖 ocr.py

Extracts text from images using Tesseract OCR (path set via .env).

Useful for scanned documents or form-based inputs.

🗃️ db.py

Connects to PostgreSQL (with the pgvector extension) and provides an upsert_chunk helper to store embeddings.

📥 ingest_pgvector.py & ingest_images_pgvector.py

Extracts text from PDFs or images.

Splits text into chunks, generates embeddings (e.g., MiniLM-L6-v2 → 384-dim vectors), and stores them in rag_docs.

Enables semantic retrieval — letting you query “similar” documents or sections later.

Depends on your own modules: app.rag.chunking and app.rag.embeddings (which you can easily add).

🧱 SQL files (001_init_pgvector.sql, 002_rag_tables.sql, 003_docs_table.sql)

Create tables and enable the vector extension.

rag_docs stores embeddings, and docs keeps metadata (file name, type, etc.).

🔄 Full Workflow Summary
Stage	Script	Description
1. Data Prep	prepare_from_existing.py	Split data into train/val folders
2. Train (Option A)	train.py	HuggingFace ViT fine-tuning
2. Train (Option B)	train_vit.py	timm-based ViT training
3. Classify Single/Batch	vit_infer.py, classify_folder.py, train_and_classify.py	Predict and organize outputs
4. Interactive Demo	streamlit_app.py	Web UI for real-time classification
5. OCR & RAG	ocr.py, ingest_pgvector.py, SQL files	Text extraction + semantic search in pgvector DB
🚀 How to Run (Simplified Steps)

Set up environment

pip install torch timm transformers streamlit pdf2image pytesseract pyyaml psycopg2-binary sentence-transformers


Prepare dataset

python prepare_from_existing.py --src data/source --train data/train --val data/val --val-ratio 0.2


Train the model

python train_vit.py  # or train.py for HuggingFace path


Run Streamlit

streamlit run streamlit_app.py


(Optional) Enable RAG

psql -d yourdb -f 001_init_pgvector.sql
python ingest_pgvector.py --in data/pdfs

🧩 Conceptually

Stage 1: The model learns to “see” — ViT classifies document images.

Stage 2: OCR lets the system “read” — extracting textual content.

Stage 3: pgvector allows it to “understand” — enabling semantic retrieval, search, and hybrid reasoning (visual + textual).

Together, it becomes a hybrid RAG pipeline — where your AI can reason both on how a document looks and what it says.
