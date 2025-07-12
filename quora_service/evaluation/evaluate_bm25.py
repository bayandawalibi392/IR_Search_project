# import os
# import joblib
# import sqlite3
# from tqdm import tqdm
# from rank_bm25 import BM25Okapi
# from text_preprocessing_service import TextPreprocessingService
# from sklearn.metrics import average_precision_score
# import numpy as np

# # --- إعدادات ---
# SOURCE = "quora"  # أو "quora"
# TOP_N = 10
# MODELS_DIR = "models"
# INDEX_DIR = "indexes"
# DB_PATH = "ir_project.db"

# # --- تحميل البيانات ---
# print("📥 تحميل البيانات...")
# bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
# inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

# doc_ids = bm25_data['doc_ids']
# tokenized_corpus = bm25_data['tokenized_texts']
# k1 = bm25_data['k1']
# b = bm25_data['b']
# bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

# doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

# # --- الاتصال بقاعدة البيانات ---
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # --- تحميل الاستعلامات و qrels ---
# cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
# queries = cursor.fetchall()

# cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
# qrels_raw = cursor.fetchall()

# # بناء قاموس الاستعلامات ذات الصلة
# qrels = {}
# for qid, doc_id in qrels_raw:
#     qrels.setdefault(qid, set()).add(doc_id)

# # --- تهيئة ---
# preprocessor = TextPreprocessingService()
# precisions = []
# recalls = []
# average_precisions = []
# reciprocal_ranks = []

# print(f"\n⚙️ بدء التقييم على {len(queries)} استعلام...\n")

# for qid, qtext in tqdm(queries):
#     if qid not in qrels:
#         continue

#     relevant_docs = qrels[qid]
#     tokens = preprocessor.preprocess(qtext, return_as_string=False)

#     # البحث في الفهرس
#     candidate_doc_ids = set()
#     for token in tokens:
#         if token in inverted_index:
#             candidate_doc_ids.update(inverted_index[token])

#     candidate_indices = [doc_id_to_idx[doc] for doc in candidate_doc_ids if doc in doc_id_to_idx]
#     if not candidate_indices:
#         continue

#     scores = bm25.get_scores(tokens)
#     scored = [(i, scores[i]) for i in candidate_indices]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     top_docs = [doc_ids[i] for i, _ in scored[:TOP_N]]

#     # حساب Precision@10
#     hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
#     precision_at_10 = sum(hits) / TOP_N
#     recall = sum(hits) / len(relevant_docs)
#     precisions.append(precision_at_10)
#     recalls.append(recall)

#     # MAP
#     y_true = [1 if doc_ids[i] in relevant_docs else 0 for i in range(len(doc_ids))]
#     y_scores = scores
#     try:
#         ap = average_precision_score(y_true, y_scores)
#     except:
#         ap = 0.0
#     average_precisions.append(ap)

#     # MRR
#     for rank, doc in enumerate(top_docs, 1):
#         if doc in relevant_docs:
#             reciprocal_ranks.append(1 / rank)
#             break
#     else:
#         reciprocal_ranks.append(0.0)

# # --- عرض النتائج ---
# print("\n📊 تقييم النظام:")
# print(f"Precision@10: {np.mean(precisions):.4f}")
# print(f"Recall:        {np.mean(recalls):.4f}")
# print(f"MAP:           {np.mean(average_precisions):.4f}")
# print(f"MRR:           {np.mean(reciprocal_ranks):.4f}")

# conn.close()
import os
import joblib
import sqlite3
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from text_preprocessing_service import TextPreprocessingService
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import json
import time

# --- إعدادات ---
SOURCE = "quora"
TOP_N = 10
MODELS_DIR = "models"
INDEX_DIR = "indexes"
DB_PATH = "ir_project.db"

# --- تحميل البيانات ---
print("📥 تحميل البيانات...")
bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

doc_ids = bm25_data['doc_ids']
tokenized_corpus = bm25_data['tokenized_texts']
k1 = bm25_data['k1']
b = bm25_data['b']
bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

# --- الاتصال بقاعدة البيانات ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# --- تحميل الاستعلامات و qrels ---
cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
queries = cursor.fetchall()

cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
qrels_raw = cursor.fetchall()

qrels = {}
for qid, doc_id in qrels_raw:
    qrels.setdefault(qid, set()).add(doc_id)

# --- تهيئة ---
preprocessor = TextPreprocessingService()
precisions, recalls, average_precisions, reciprocal_ranks = [], [], [], []

print(f"\n⚙️ بدء التقييم على {len(queries)} استعلام...\n")
start_time = time.perf_counter()

for qid, qtext in tqdm(queries):
    if qid not in qrels:
        continue

    relevant_docs = qrels[qid]
    tokens = preprocessor.preprocess(qtext, return_as_string=False)

    candidate_doc_ids = set()
    for token in tokens:
        if token in inverted_index:
            candidate_doc_ids.update(inverted_index[token])

    candidate_indices = [doc_id_to_idx[doc] for doc in candidate_doc_ids if doc in doc_id_to_idx]
    if not candidate_indices:
        continue

    scores = bm25.get_scores(tokens)
    scored = [(i, scores[i]) for i in candidate_indices]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc_ids[i] for i, _ in scored[:TOP_N]]

    # Precision@10 و Recall
    hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
    precisions.append(sum(hits) / TOP_N)
    recalls.append(sum(hits) / len(relevant_docs))

    # MAP
    y_true = [1 if doc_ids[i] in relevant_docs else 0 for i in range(len(doc_ids))]
    y_scores = scores
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
elapsed_time = round(end_time - start_time, 2)

# --- عرض النتائج ---
results = {
    "Precision@10": round(np.mean(precisions), 4),
    "Recall": round(np.mean(recalls), 4),
    "MAP": round(np.mean(average_precisions), 4),
    "MRR": round(np.mean(reciprocal_ranks), 4),
    "Execution Time (seconds)": elapsed_time,
    "Queries Evaluated": len(precisions)
}

pd.DataFrame([results])

# حفظ النتائج
results_df.to_csv("bm25_evaluation_results.csv", index=False)
with open("bm25_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# إغلاق الاتصال
conn.close()
