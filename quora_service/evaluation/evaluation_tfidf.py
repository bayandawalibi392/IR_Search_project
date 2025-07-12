import joblib
import os
import sqlite3
import numpy as np
import pandas as pd
import time
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing_service import TextPreprocessingService
import json

# إعدادات
INDEX_DIR = 'indexes'
MODELS_DIR = 'models'
SOURCE = "quora"
TOP_N = 10

# تحميل الفهرس المعكوس
inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

# تحميل نموذج TF-IDF
tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))
doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

# المعالجة النصية
preprocessor = TextPreprocessingService()

# الاتصال بقاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

# تحميل الاستعلامات
cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
queries = cursor.fetchall()

# تحميل الوثائق ذات الصلة (qrels)
cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
relevance_data = cursor.fetchall()

# تحويل qrels إلى dict: {query_id: set(doc_ids)}
relevant_docs = {}
for qid, did in relevance_data:
    relevant_docs.setdefault(qid, set()).add(did)

# مقاييس التقييم
total_precisions = []
total_recalls = []
average_precisions = []
reciprocal_ranks = []

start_time = time.perf_counter()

for query_id, query_text in queries:
    tokens = preprocessor.preprocess(query_text, return_as_string=False)

    candidate_doc_ids = set()
    for term in tokens:
        if term in inverted_index:
            candidate_doc_ids.update(inverted_index[term])

    candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]
    if not candidate_indices:
        continue

    query_vec = tfidf_vectorizer.transform([" ".join(tokens)])
    sims = cosine_similarity(query_vec, doc_matrix[candidate_indices])[0]
    top_indices = np.argsort(sims)[::-1][:TOP_N]
    retrieved_docs = [doc_ids[candidate_indices[i]] for i in top_indices]

    # وثائق الاستعلام ذات الصلة
    rel_docs = relevant_docs.get(query_id, set())
    if not rel_docs:
        continue

    # حساب Precision@10
    retrieved_relevant = [doc_id for doc_id in retrieved_docs if doc_id in rel_docs]
    precision_at_10 = len(retrieved_relevant) / TOP_N
    recall = len(retrieved_relevant) / len(rel_docs)

    # MRR
    rr = 0
    for rank, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in rel_docs:
            rr = 1 / rank
            break

    # Average Precision
    num_rel = 0
    sum_precisions = 0
    for rank, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in rel_docs:
            num_rel += 1
            sum_precisions += num_rel / rank
    ap = sum_precisions / len(rel_docs)

    # تحديث القيم الإجمالية
    total_precisions.append(precision_at_10)
    total_recalls.append(recall)
    reciprocal_ranks.append(rr)
    average_precisions.append(ap)

end_time = time.perf_counter()
elapsed = end_time - start_time

# حساب المتوسطات
MAP = np.mean(average_precisions)
MRR = np.mean(reciprocal_ranks)
mean_precision = np.mean(total_precisions)
mean_recall = np.mean(total_recalls)

# إعداد النتائج في dict
results = {
    "Precision@10": round(mean_precision, 4),
    "Recall": round(mean_recall, 4),
    "MAP": round(MAP, 4),
    "MRR": round(MRR, 4),
    "Execution Time (seconds)": round(elapsed, 2),
    "Queries Evaluated": len(total_precisions)
}

# طباعة النتائج باستخدام DataFrame
results_df = pd.DataFrame([results])
print("\n📊 تقييم النظام:")
print(results_df)

# حفظ النتائج كـ JSON
with open("tfidf_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# حفظ النتائج كـ CSV
results_df.to_csv("tfidf_evaluation_results.csv", index=False)

conn.close()
