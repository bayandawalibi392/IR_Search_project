import sqlite3
import joblib
import numpy as np
import time
from TextPreprocessor import TextPreprocessor
import warnings

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
GROUP = 'webis'
TOP_K = 10

# تحميل الموارد
print("📦 تحميل نموذج BM25 والفهرس المعكوس...")
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
bm25 = joblib.load(f"{MODEL_DIR}/bm25_model_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bm25_{GROUP}.joblib")

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# تحميل الاستعلامات والعلاقات
cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (GROUP,))
queries = cursor.fetchall()
query_dict = {q_id: text for q_id, text in queries}

cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (GROUP,))
qrel_rows = cursor.fetchall()
qrels = {}
for q_id, doc_id in qrel_rows:
    qrels.setdefault(q_id, set()).add(doc_id)

# أداة المعالجة
pre = TextPreprocessor()

# التقييم
map_scores = []
mrr_scores = []
recall_scores = []
precision_scores = []

start_time = time.perf_counter()

for query_id, query_text in query_dict.items():
    query_tokens = pre.tokenize(pre.clean_text(query_text))
    if not query_tokens:
        continue

    # استخراج المرشحين من الفهرس المعكوس
    candidate_indices = set()
    for token in query_tokens:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])

    if not candidate_indices:
        continue

    # حساب الدرجات باستخدام BM25
    all_scores = bm25.get_scores(query_tokens)
    candidate_indices = sorted(candidate_indices)
    candidate_scores = [(idx, all_scores[idx]) for idx in candidate_indices]
    top_indices_with_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:TOP_K]

    retrieved = [doc_ids[idx] for idx, _ in top_indices_with_scores]
    scores_only = [score for _, score in top_indices_with_scores]

    relevant_docs = qrels.get(query_id, set())
    y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]

    # MAP
    if any(y_true):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.metrics import average_precision_score
            map_scores.append(average_precision_score(y_true, scores_only))
    else:
        map_scores.append(0)

    # MRR
    for rank, rel in enumerate(y_true, 1):
        if rel:
            mrr_scores.append(1 / rank)
            break
    else:
        mrr_scores.append(0)

    # Recall@K
    recall = sum(y_true) / len(relevant_docs) if relevant_docs else 0
    recall_scores.append(recall)

    # Precision@K
    precision = sum(y_true) / len(y_true) if y_true else 0
    precision_scores.append(precision)

end_time = time.perf_counter()
elapsed = end_time - start_time

# عرض النتائج
print("\n✅ التقييم مكتمل باستخدام BM25!")
print(f"📊 MAP: {np.mean(map_scores):.4f}")
print(f"📊 MRR: {np.mean(mrr_scores):.4f}")
print(f"📊 Recall@{TOP_K}: {np.mean(recall_scores):.4f}")
print(f"📊 Precision@{TOP_K}: {np.mean(precision_scores):.4f}")
print(f"🕒 زمن التنفيذ الكلي: {elapsed:.2f} ثانية")
