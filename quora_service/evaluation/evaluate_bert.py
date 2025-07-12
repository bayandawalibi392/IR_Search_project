# import os
# import joblib
# import sqlite3
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import average_precision_score
# from sklearn.metrics.pairwise import cosine_similarity
# from text_preprocessing_service import TextPreprocessingService
# from transformers import AutoTokenizer, AutoModel
# import torch

# # إعدادات
# SOURCE = "quora"  # أو "quora"
# TOP_N = 10
# MODELS_DIR = "models"
# INDEX_DIR = "indexes"
# DB_PATH = "ir_project.db"

# # تحميل الفهرس
# inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

# # تحميل تمثيلات BERT
# doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
# doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))
# doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

# # تحميل نموذج BERT
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device).eval()

# # دالة استخراج التمثيل
# def embed(text):
#     tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#     with torch.no_grad():
#         output = model(**tokens)
#     return output.last_hidden_state.mean(dim=1).cpu().numpy()

# # ربط قاعدة البيانات
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # تحميل الاستعلامات والـ qrels
# cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
# queries = cursor.fetchall()
# cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
# qrels_raw = cursor.fetchall()

# qrels = {}
# for qid, doc_id in qrels_raw:
#     qrels.setdefault(qid, set()).add(doc_id)

# # تقييم
# preprocessor = TextPreprocessingService()
# precisions, recalls, average_precisions, reciprocal_ranks = [], [], [], []

# print(f"\n⚙️ بدء التقييم على {len(queries)} استعلام...\n")

# for qid, qtext in tqdm(queries):
#     if qid not in qrels:
#         continue

#     relevant_docs = qrels[qid]
#     tokens = preprocessor.preprocess(qtext, return_as_string=False)

#     candidate_doc_ids = set()
#     for token in tokens:
#         if token in inverted_index:
#             candidate_doc_ids.update(inverted_index[token])

#     candidate_indices = [doc_id_to_idx[doc] for doc in candidate_doc_ids if doc in doc_id_to_idx]
#     if not candidate_indices:
#         continue

#     query_vec = embed(" ".join(tokens))
#     candidate_vectors = doc_vectors[candidate_indices]
#     sims = cosine_similarity(query_vec, candidate_vectors)[0]

#     ranked = sorted(zip(candidate_indices, sims), key=lambda x: x[1], reverse=True)
#     top_docs = [doc_ids[i] for i, _ in ranked[:TOP_N]]

#     # Precision@10
#     hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
#     precisions.append(sum(hits) / TOP_N)
#     recalls.append(sum(hits) / len(relevant_docs))

#     # MAP
#     y_true = [1 if doc_ids[i] in relevant_docs else 0 for i in candidate_indices]
#     y_scores = sims
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

# # طباعة النتائج
# print("\n📊 تقييم النظام باستخدام BERT:")
# print(f"Precision@10: {np.mean(precisions):.4f}")
# print(f"Recall:        {np.mean(recalls):.4f}")
# print(f"MAP:           {np.mean(average_precisions):.4f}")
# print(f"MRR:           {np.mean(reciprocal_ranks):.4f}")

# conn.close()


import os
import joblib
import sqlite3
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing_service import TextPreprocessingService
from transformers import AutoTokenizer, AutoModel
import torch
import json

# إعدادات
SOURCE = "quora"
TOP_N = 10
MODELS_DIR = "models"
INDEX_DIR = "indexes"
DB_PATH = "ir_project.db"

# تحميل الفهرس
inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

# تحميل تمثيلات BERT
doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))
doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

# تحميل نموذج BERT
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# دالة استخراج التمثيل
def embed(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).cpu().numpy()

# ربط قاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# تحميل الاستعلامات والـ qrels
cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
queries = cursor.fetchall()
cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
qrels_raw = cursor.fetchall()

qrels = {}
for qid, doc_id in qrels_raw:
    qrels.setdefault(qid, set()).add(doc_id)

# تقييم
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

    query_vec = embed(" ".join(tokens))
    candidate_vectors = doc_vectors[candidate_indices]
    sims = cosine_similarity(query_vec, candidate_vectors)[0]

    ranked = sorted(zip(candidate_indices, sims), key=lambda x: x[1], reverse=True)
    top_docs = [doc_ids[i] for i, _ in ranked[:TOP_N]]

    # Precision@10
    hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
    precisions.append(sum(hits) / TOP_N)
    recalls.append(sum(hits) / len(relevant_docs))

    # MAP
    y_true = [1 if doc_ids[i] in relevant_docs else 0 for i in candidate_indices]
    y_scores = sims
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

# حساب المتوسطات
results = {
    "Precision@10": round(np.mean(precisions), 4),
    "Recall": round(np.mean(recalls), 4),
    "MAP": round(np.mean(average_precisions), 4),
    "MRR": round(np.mean(reciprocal_ranks), 4),
    "Execution Time (seconds)": elapsed,
    "Queries Evaluated": len(precisions)
}

# عرض النتائج باستخدام DataFrame
results_df = pd.DataFrame([results])
print("\n📊 تقييم النظام باستخدام BERT:")
print(results_df)

# حفظ النتائج كـ JSON
with open("bert_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# حفظ النتائج كـ CSV
results_df.to_csv("bert_evaluation_results.csv", index=False)

# إغلاق الاتصال
conn.close()
