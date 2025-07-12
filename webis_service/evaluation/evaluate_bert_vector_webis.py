# import sqlite3
# import joblib
# import numpy as np
# import time
# import faiss
# from sklearn.metrics import average_precision_score
# from sentence_transformers import SentenceTransformer
# from TextPreprocessor import TextPreprocessor
# import warnings

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# FAISS_INDEX_PATH = f"indexes/faiss_index_webis_bert.index"
# DOC_IDS_PATH = f"{MODEL_DIR}/doc_ids_bert_webis.joblib"
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد
# print("📦 تحميل فهرس FAISS والتمثيلات...")
# faiss_index = faiss.read_index(FAISS_INDEX_PATH)
# doc_ids = joblib.load(DOC_IDS_PATH)
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')

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
# map_scores, mrr_scores, recall_scores, precision_scores = [], [], [], []

# start_time = time.perf_counter()

# for query_id, query_text in query_dict.items():
#     content = pre.preprocess(query_text)
#     if not content:
#         continue

#     query_vec = bert_model.encode([' '.join(content)]).astype('float32')
#     D, I = faiss_index.search(query_vec, TOP_K)  # D: distances, I: indices

#     retrieved = [doc_ids[i] for i in I[0]]
#     scores = 1 - D[0]  # تحويل L2 distance إلى score (كلما قلت المسافة زاد التشابه)

#     relevant_docs = qrels.get(query_id, set())
#     y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]

#     # MAP
#     if any(y_true):
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             map_scores.append(average_precision_score(y_true, scores))
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
# print("\n✅ التقييم مكتمل باستخدام FAISS + BERT!")
# print(f"📊 MAP: {np.mean(map_scores):.4f}")
# print(f"📊 MRR: {np.mean(mrr_scores):.4f}")
# print(f"📊 Recall@{TOP_K}: {np.mean(recall_scores):.4f}")
# print(f"📊 Precision@{TOP_K}: {np.mean(precision_scores):.4f}")
# print(f"🕒 زمن التنفيذ الكلي: {elapsed:.2f} ثانية")

import sqlite3
import joblib
import numpy as np
import time
import faiss
import pandas as pd
import json
import warnings
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer
from TextPreprocessor import TextPreprocessor
from pprint import pprint

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
FAISS_INDEX_PATH = f"indexes/faiss_index_webis_bert.index"
DOC_IDS_PATH = f"{MODEL_DIR}/doc_ids_bert_webis.joblib"
GROUP = 'webis'
TOP_K = 10

# تحميل الموارد
print("📦 تحميل فهرس FAISS والتمثيلات...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
doc_ids = joblib.load(DOC_IDS_PATH)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

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
map_scores, mrr_scores, recall_scores, precision_scores = [], [], [], []

start_time = time.perf_counter()

for query_id, query_text in query_dict.items():
    content = pre.preprocess(query_text)
    if not content:
        continue

    query_vec = bert_model.encode([' '.join(content)]).astype('float32')
    D, I = faiss_index.search(query_vec, TOP_K)  # D: distances, I: indices

    retrieved = [doc_ids[i] for i in I[0]]
    scores = 1 - D[0]  # تحويل L2 distance إلى score

    relevant_docs = qrels.get(query_id, set())
    y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]

    # MAP
    if any(y_true):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            map_scores.append(average_precision_score(y_true, scores))
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

# إعداد النتائج
results = {
    "MAP": round(np.mean(map_scores), 4),
    "MRR": round(np.mean(mrr_scores), 4),
    f"Recall@{TOP_K}": round(np.mean(recall_scores), 4),
    f"Precision@{TOP_K}": round(np.mean(precision_scores), 4),
    "Execution Time (s)": round(elapsed, 2),
    "Queries Evaluated": len(map_scores)
}

# طباعة النتائج
print("\n✅ التقييم مكتمل باستخدام FAISS + BERT!")
pprint(results)

# عرض النتائج في جدول داخل Jupyter
df_results = pd.DataFrame([results])
display(df_results)

# حفظ النتائج
df_results.to_csv("faiss_bert_evaluation_results.csv", index=False)
with open("faiss_bert_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
