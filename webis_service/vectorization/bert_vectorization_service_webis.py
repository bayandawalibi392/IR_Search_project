import os
import sqlite3
import joblib
from sentence_transformers import SentenceTransformer
from TextPreprocessor import TextPreprocessor  # لا زال ممكن استخدامه إن أحببت لتنظيف بسيط، لكنه ليس ضروريًا

# إعدادات
DB_PATH = 'ir_project.db'
OUTPUT_DIR = 'models'
GROUPS = ['webis']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# نموذج BERT
bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # أو أي نموذج آخر من huggingface

def build_bert_for_group(group_name):
    cursor.execute("SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (group_name,))
    rows = cursor.fetchall()

    doc_ids = []
    texts = []

    for doc_id, content_str in rows:
        if content_str:
            doc_ids.append(doc_id)
            texts.append(content_str.strip())

    print(f"✅ Loaded {len(texts)} documents from group '{group_name}'")

    # تمثيل BERT
    embeddings = bert_model.encode(texts, batch_size=64, show_progress_bar=True)

    # التخزين
    joblib.dump(embeddings, f"{OUTPUT_DIR}/bert_vectors_{group_name}.joblib")
    joblib.dump(doc_ids, f"{OUTPUT_DIR}/doc_ids_bert_{group_name}.joblib")
    print(f"💾 Saved BERT embeddings for group '{group_name}'")

# تنفيذ
for group in GROUPS:
    build_bert_for_group(group)

print("✅ BERT vectorization completed for all documents per group.")
