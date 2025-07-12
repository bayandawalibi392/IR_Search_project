import os
import joblib
import sqlite3
import numpy as np
import pandas as pd
import time
import json
import faiss
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer
from TextPreprocessor import TextPreprocessor

# --- الإعدادات ---
SOURCE = "quora"
TOP_N = 10
MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "indexes"
MODEL_DIR = "models"
DB_PATH = "ir_project.db"

# --- تحميل FAISS Index ---
index_path = os.path.join(INDEX_DIR, f"faiss_index_{SOURCE}_bert.index")
print(f"📥 تحميل FAISS index من: {index_path}")
index = faiss.read_index(index_path)

# --- تحميل معرّفات الوثائق ---
doc_ids = joblib.load(os.path.join(MODEL_DIR, f"bert_{SOURCE}_doc_ids.joblib"))

# --- تحميل النموذج والمعالجة ---
model = SentenceTransformer(MODEL_NAME)
preprocessor = TextPreprocessor()

# --- ربط قاعدة البيانات ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
queries = cursor.fetchall()

cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
qrels_raw = cursor.fetchall()

qrels = {}
for qid, doc_id in qrels_raw:
    qrels.setdefault(qid, set()).add(doc_id)

# --- التقييم ---
precisions, recalls, average_precisions, reciprocal_ranks = [], [], [], []

print(f"\n⚙️ بدء التقييم على {len(queries)} استعلام...\n")
start_time = time.perf_counter()

for qid, qtext in tqdm(queries):
    if qid not in qrels:
        continue

    relevant_docs = qrels[qid]
    tokens = preprocessor.preprocess(qtext)
    if not tokens:
        continue

    query_vec = model.encode([" ".join(tokens)]).astype("float32")

    # --- البحث باستخدام FAISS ---
    D, I = index.search(query_vec, TOP_N)
    top_docs = [doc_ids[i] for i in I[0] if i < len(doc_ids)]
    scores = D[0]

    # Precision@10
    hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
    precisions.append(sum(hits) / TOP_N)
    recalls.append(sum(hits) / len(relevant_docs))

    # MAP
    y_true = [1 if doc in relevant_docs else 0 for doc in top_docs]
    y_scores = [1.0 / (i + 1) for i in range(len(y_true))]  # أو استخدم scores مباشرة إن أردت
    try:
        ap = average_precision_score(y_true, y_scores)
    except:
        ap = 0.0
    average_precisions.append(ap)

    # MRR
    for rank, doc in enumerate(top_docs, 1):
        if doc in relevant_docs:
            reciprocal_ranks.append(1 / rank)
            break
    else:
        reciprocal_ranks.append(0.0)

end_time = time.perf_counter()
elapsed = round(end_time - start_time, 2)

# --- النتائج ---
results = {
    "Precision@10": round(np.mean(precisions), 4),
    "Recall": round(np.mean(recalls), 4),
    "MAP": round(np.mean(average_precisions), 4),
    "MRR": round(np.mean(reciprocal_ranks), 4),
    "Execution Time (seconds)": elapsed,
    "Queries Evaluated": len(precisions)
}

# --- عرض النتائج ---
results_df = pd.DataFrame([results])
print("\n📊 تقييم النظام باستخدام BERT + FAISS:")
print(results_df)

# --- حفظ النتائج ---
results_df.to_csv(f"faiss_bert_evaluation_{SOURCE}.csv", index=False)
with open(f"faiss_bert_evaluation_{SOURCE}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

conn.close()
