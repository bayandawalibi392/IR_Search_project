# import sqlite3
# import joblib
# import numpy as np
# import time
# from sklearn.metrics import average_precision_score
# from sklearn.metrics.pairwise import cosine_similarity
# from TextPreprocessor import TextPreprocessor
# from sentence_transformers import SentenceTransformer
# import warnings

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# GROUP = 'webis'  # فقط مجموعة واحدة
# TOP_K = 10
# ALPHA = 0.7  # نسبة دمج TF-IDF و BERT

# print(f"📦 تحميل النماذج والبيانات لمجموعة: {GROUP}...")

# # تحميل النماذج
# tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
# tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
# tfidf_doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# bert_vectors = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
# bert_doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")

# # إعادة ترتيب bert_vectors ليتطابق مع ترتيب tfidf_doc_ids
# bert_id_to_vec = dict(zip(bert_doc_ids, bert_vectors))
# bert_vectors_aligned = np.array([bert_id_to_vec[doc_id] for doc_id in tfidf_doc_ids])

# # تحميل الفهرس العكسي
# inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")

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

# # التحضير
# pre = TextPreprocessor()
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# # التقييم
# map_scores, mrr_scores, recall_scores, precision_scores = [], [], [], []

# start_time = time.perf_counter()

# for query_id, query_text in query_dict.items():
#     tokens = pre.preprocess(query_text, use_stemming=True, use_lemmatization=False)
#     if not tokens:
#         continue

#     query_str = pre.clean_text(' '.join(tokens))
#     query_vec_tfidf = tfidf_vectorizer.transform([query_str])
#     query_vec_bert = bert_model.encode([' '.join(tokens)]).reshape(1, -1)

#     # استخراج المرشحين من الفهرس العكسي
#     candidate_indices = set()
#     for token in tokens:
#         if token in inverted_index:
#             candidate_indices.update(inverted_index[token])
#     if not candidate_indices:
#         continue

#     candidate_indices = sorted(candidate_indices)
#     candidate_tfidf = tfidf_matrix[candidate_indices]
#     candidate_bert = bert_vectors_aligned[candidate_indices]
#     candidate_doc_ids = [tfidf_doc_ids[i] for i in candidate_indices]

#     # حساب التشابه ودمج النتائج
#     sim_tfidf = cosine_similarity(query_vec_tfidf, candidate_tfidf)[0]
#     sim_bert = cosine_similarity(query_vec_bert, candidate_bert)[0]
#     scores = ALPHA * sim_tfidf + (1 - ALPHA) * sim_bert

#     top_indices = np.argsort(scores)[-TOP_K:][::-1]
#     retrieved = [candidate_doc_ids[i] for i in top_indices]
#     y_scores = scores[top_indices]

#     relevant_docs = qrels.get(query_id, set())
#     y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]

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
# print(f"📊 MAP: {np.mean(map_scores):.4f}")
# print(f"📊 MRR: {np.mean(mrr_scores):.4f}")
# print(f"📊 Recall@{TOP_K}: {np.mean(recall_scores):.4f}")
# print(f"📊 Precision@{TOP_K}: {np.mean(precision_scores):.4f}")
# print(f"🕒 زمن التنفيذ الكلي: {elapsed:.2f} ثانية")
import sqlite3
import joblib
import numpy as np
import time
import warnings
import pandas as pd
import json
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from TextPreprocessor import TextPreprocessor
from pprint import pprint

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
GROUP = 'webis'
TOP_K = 10
ALPHA = 0.7  # النسبة بين TF-IDF وBERT

print(f"📦 تحميل النماذج والبيانات لمجموعة: {GROUP}...")

# تحميل النماذج
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
tfidf_doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

bert_vectors = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
bert_doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")

# إعادة ترتيب bert_vectors بحسب ترتيب tfidf_doc_ids
bert_id_to_vec = dict(zip(bert_doc_ids, bert_vectors))
bert_vectors_aligned = np.array([bert_id_to_vec[doc_id] for doc_id in tfidf_doc_ids])

# تحميل الفهرس العكسي
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")

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

# التحضير
pre = TextPreprocessor()
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# التقييم
map_scores, mrr_scores, recall_scores, precision_scores = [], [], [], []

start_time = time.perf_counter()

for query_id, query_text in query_dict.items():
    tokens = pre.preprocess(query_text, use_stemming=True, use_lemmatization=False)
    if not tokens:
        continue

    query_str = pre.clean_text(' '.join(tokens))
    query_vec_tfidf = tfidf_vectorizer.transform([query_str])
    query_vec_bert = bert_model.encode([' '.join(tokens)]).reshape(1, -1)

    # استخراج المرشحين
    candidate_indices = set()
    for token in tokens:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])
    if not candidate_indices:
        continue

    candidate_indices = sorted(candidate_indices)
    candidate_tfidf = tfidf_matrix[candidate_indices]
    candidate_bert = bert_vectors_aligned[candidate_indices]
    candidate_doc_ids = [tfidf_doc_ids[i] for i in candidate_indices]

    # حساب التشابه
    sim_tfidf = cosine_similarity(query_vec_tfidf, candidate_tfidf)[0]
    sim_bert = cosine_similarity(query_vec_bert, candidate_bert)[0]
    scores = ALPHA * sim_tfidf + (1 - ALPHA) * sim_bert

    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    retrieved = [candidate_doc_ids[i] for i in top_indices]
    y_scores = scores[top_indices]

    relevant_docs = qrels.get(query_id, set())
    y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]

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

elapsed = time.perf_counter() - start_time

# النتائج النهائية
results = {
    "MAP": round(np.mean(map_scores), 4),
    "MRR": round(np.mean(mrr_scores), 4),
    f"Recall@{TOP_K}": round(np.mean(recall_scores), 4),
    f"Precision@{TOP_K}": round(np.mean(precision_scores), 4),
    "Execution Time (s)": round(elapsed, 2),
    "Queries Evaluated": len(map_scores)
}

# طباعة وجدول
print("\n✅ التقييم مكتمل باستخدام Hybrid (TF-IDF + BERT)!")
pprint(results)

df_results = pd.DataFrame([results])
display(df_results)  # لعرض الجدول داخل Jupyter

# حفظ النتائج
df_results.to_csv("hybrid_evaluation_results.csv", index=False)
with open("hybrid_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)
