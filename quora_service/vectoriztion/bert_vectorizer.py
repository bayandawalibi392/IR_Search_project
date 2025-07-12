import os
import sqlite3
from tqdm import tqdm
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

# --- إعدادات ---
SOURCES = ["quora"]
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# تحميل نموذج BERT
print("🧠 Loading BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# الاتصال بقاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

for source in SOURCES:
    print(f"\n📥 Loading preprocessed documents from source: {source}")
    cursor.execute("SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (source,))
    rows = cursor.fetchall()

    doc_ids = []
    texts = []

    for doc_id, content in tqdm(rows, desc=f"{source} docs"):
        if content.strip():
            doc_ids.append(doc_id)
            texts.append(content.strip())

    print(f"🔢 Encoding {len(texts)} preprocessed documents using BERT for {source}...")
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True)

    # حفظ النتائج
    joblib.dump(doc_ids, os.path.join(MODELS_DIR, f"bert_{source}_doc_ids.joblib"))
    joblib.dump(vectors, os.path.join(MODELS_DIR, f"bert_{source}_vectors.joblib"))

    print(f"✅ Saved BERT embeddings for {source}.")

conn.close()
