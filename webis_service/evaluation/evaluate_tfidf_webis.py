# import sqlite3
# import joblib
# import numpy as np
# import time
# from sklearn.metrics import average_precision_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import pairwise
# from TextPreprocessor import TextPreprocessor  # تأكد أن هذا يحمل نفس نوع المعالجة الذي استخدمته
# import warnings

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل النماذج والفهارس
# print("📦 تحميل النماذج والبيانات...")
# inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
# tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
# tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# # الاتصال بقاعدة البيانات
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # تحميل الاستعلامات والعلاقات
# cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (GROUP,))
# queries = cursor.fetchall()
# query_dict = {q_id: text for q_id, text in queries}

# cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (GROUP,))
# qrel_rows = cursor.fetchall()
# qrels = {}
# for q_id, doc_id in qrel_rows:
#     qrels.setdefault(q_id, set()).add(doc_id)

# # المعالجة
# pre = TextPreprocessor()

# # التقييم
# map_scores = []
# mrr_scores = []
# recall_scores = []
# precision_scores = []

# start_time = time.perf_counter()

# for query_id, query_text in query_dict.items():
#     content = pre.preprocess(query_text, use_stemming=True, use_lemmatization=False)
#     if not content:
#         continue

#     candidate_indices = set()
#     for token in content:
#         if token in inverted_index:
#             candidate_indices.update(inverted_index[token])
#     if not candidate_indices:
#         continue

#     candidate_indices = sorted(candidate_indices)
#     candidate_vectors = tfidf_matrix[candidate_indices]
#     query_str = pre.clean_text(' '.join(content))  # نفس دالة preprocessor التي استُخدمت في التدريب
#     query_vec = tfidf_vectorizer.transform([query_str])

#     scores = pairwise.cosine_similarity(query_vec, candidate_vectors)[0]

#     top_indices_local = np.argsort(scores)[-TOP_K:][::-1]
#     top_indices_global = [candidate_indices[i] for i in top_indices_local]
#     retrieved = [doc_ids[i] for i in top_indices_global]

#     relevant_docs = qrels.get(query_id, set())
#     y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]
#     y_scores = scores[top_indices_local]

#     # MAP
#     if any(y_true):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             map_scores.append(average_precision_score(y_true, y_scores))
#     else:
#         map_scores.append(0)

#     # MRR
#     for rank, rel in enumerate(y_true, 1):
#         if rel:
#             mrr_scores.append(1 / rank)
#             break
#     else:
#         mrr_scores.append(0)

#     # Recall@K
#     recall = sum(y_true) / len(relevant_docs) if relevant_docs else 0
#     recall_scores.append(recall)

#     # Precision@K
#     precision = sum(y_true) / len(y_true) if y_true else 0
#     precision_scores.append(precision)

# end_time = time.perf_counter()
# elapsed = end_time - start_time

# # عرض النتائج
# print("\n✅ التقييم مكتمل!")
# print(f"📊 MAP (60% التركيز): {np.mean(map_scores):.4f}")
# print(f"📊 MRR: {np.mean(mrr_scores):.4f}")
# print(f"📊 Recall@{TOP_K}: {np.mean(recall_scores):.4f}")
# print(f"📊 Precision@{TOP_K}: {np.mean(precision_scores):.4f}")
# print(f"🕒 زمن التنفيذ الكلي: {elapsed:.2f} ثانية")
import sqlite3
import joblib
import numpy as np
import time
import warnings
import json
import pandas as pd
from sklearn.metrics import average_precision_score, pairwise
from TextPreprocessor import TextPreprocessor  # تأكد أن المسار صحيح داخل Jupyter
from pprint import pprint

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
GROUP = 'webis'
TOP_K = 10

# تحميل النماذج والفهارس
print("📦 تحميل النماذج والبيانات...")
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

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

# المعالجة
pre = TextPreprocessor()

# التقييم
map_scores = []
mrr_scores = []
recall_scores = []
precision_scores = []

start_time = time.perf_counter()

for query_id, query_text in query_dict.items():
    content = pre.preprocess(query_text, use_stemming=True, use_lemmatization=False)
    if not content:
        continue

    candidate_indices = set()
    for token in content:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])
    if not candidate_indices:
        continue

    candidate_indices = sorted(candidate_indices)
    candidate_vectors = tfidf_matrix[candidate_indices]
    query_str = pre.clean_text(' '.join(content))
    query_vec = tfidf_vectorizer.transform([query_str])
    scores = pairwise.cosine_similarity(query_vec, candidate_vectors)[0]

    top_indices_local = np.argsort(scores)[-TOP_K:][::-1]
    top_indices_global = [candidate_indices[i] for i in top_indices_local]
    retrieved = [doc_ids[i] for i in top_indices_global]

    relevant_docs = qrels.get(query_id, set())
    y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]
    y_scores = scores[top_indices_local]

    # MAP
    if any(y_true):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            map_scores.append(average_precision_score(y_true, y_scores))
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

# إعداد النتائج كقاموس
results = {
    "MAP": round(np.mean(map_scores), 4),
    "MRR": round(np.mean(mrr_scores), 4),
    f"Recall@{TOP_K}": round(np.mean(recall_scores), 4),
    f"Precision@{TOP_K}": round(np.mean(precision_scores), 4),
    "Execution Time (s)": round(elapsed, 2),
    "Queries Evaluated": len(map_scores)
}

# طباعة النتائج
print("✅ التقييم مكتمل!")
pprint(results)

# حفظ النتائج كـ JSON
with open("tfidf_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# حفظ النتائج كـ CSV
pd.DataFrame([results]).to_csv("tfidf_evaluation_results.csv", index=False)
