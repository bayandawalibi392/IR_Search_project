# import os
# import sqlite3
# import joblib
# import numpy as np
# from sklearn.metrics import average_precision_score
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModel
# import torch
# from tqdm import tqdm
# from text_preprocessing_service import TextPreprocessingService

# # --- إعدادات ---
# SOURCE = "quora"  # أو "quora"
# MODELS_DIR = "models"
# INDEX_DIR = "indexes"
# TOP_N = 10
# ALPHA = 0.6  # وزن TF-IDF مقابل BERT

# # تحميل BERT
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()

# def embed_text(text):
#     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#     with torch.no_grad():
#         output = model(**encoded_input)
#     return output.last_hidden_state.mean(dim=1).cpu().numpy()

# # تحميل الفهرس
# inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

# # تحميل التمثيلات
# tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
# tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
# tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

# bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
# bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

# tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
# bert_id_to_idx = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

# # معالجة
# preprocessor = TextPreprocessingService()

# # الاتصال بقاعدة البيانات
# conn = sqlite3.connect("ir_project.db")
# cursor = conn.cursor()

# cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
# queries = cursor.fetchall()

# cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
# qrels_raw = cursor.fetchall()

# # qrels كقاموس
# qrels = {}
# for qid, doc_id in qrels_raw:
#     qrels.setdefault(qid, set()).add(doc_id)

# # --- تقييم ---
# precisions, recalls, average_precisions, reciprocal_ranks = [], [], [], []

# print(f"\n⚙️ بدء التقييم على {len(queries)} استعلام...\n")

# for qid, query_text in tqdm(queries):
#     if qid not in qrels:
#         continue

#     relevant_docs = qrels[qid]
#     tokens = preprocessor.preprocess(query_text, return_as_string=False)
#     cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

#     # مرشحون من الفهرس
#     candidate_doc_ids = set()
#     for token in tokens:
#         if token in inverted_index:
#             candidate_doc_ids.update(inverted_index[token])

#     common_doc_ids = list(candidate_doc_ids.intersection(tfidf_doc_ids).intersection(bert_doc_ids))
#     if not common_doc_ids:
#         continue

#     tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
#     bert_indices = [bert_id_to_idx[doc_id] for doc_id in common_doc_ids]

#     # التمثيلات
#     tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
#     sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix[tfidf_indices])[0]

#     bert_query_vec = embed_text(cleaned_query)
#     sims_bert = cosine_similarity(bert_query_vec, bert_vectors[bert_indices])[0]

#     # دمج الدرجات
#     final_sims = ALPHA * sims_tfidf + (1 - ALPHA) * sims_bert
#     ranked = [(doc_id, final_sims[i]) for i, doc_id in enumerate(common_doc_ids)]
#     ranked.sort(key=lambda x: x[1], reverse=True)
#     top_docs = [doc_id for doc_id, _ in ranked[:TOP_N]]

#     # حساب المقاييس
#     hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
#     precisions.append(sum(hits) / TOP_N)
#     recalls.append(sum(hits) / len(relevant_docs))

#     # MAP
#     y_true = [1 if doc in relevant_docs else 0 for doc in common_doc_ids]
#     y_scores = final_sims
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

# # --- النتائج ---
# print("\n📊 تقييم النظام باستخدام التمثيل الهجين مع الفهرس:")
# print(f"Precision@10: {np.mean(precisions):.4f}")
# print(f"Recall:        {np.mean(recalls):.4f}")
# print(f"MAP:           {np.mean(average_precisions):.4f}")
# print(f"MRR:           {np.mean(reciprocal_ranks):.4f}")

# conn.close()
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import time
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import json
from text_preprocessing_service import TextPreprocessingService

# --- إعدادات ---
SOURCE = "quora"
MODELS_DIR = "models"
INDEX_DIR = "indexes"
TOP_N = 10
ALPHA = 0.6  # وزن TF-IDF مقابل BERT

# تحميل BERT
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def embed_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**encoded_input)
    return output.last_hidden_state.mean(dim=1).cpu().numpy()

# تحميل الفهارس والنماذج
inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))
tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))
bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))
tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
bert_id_to_idx = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

# معالجة
preprocessor = TextPreprocessingService()

# الاتصال بقاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()
cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
queries = cursor.fetchall()
cursor.execute("SELECT query_id, doc_id FROM qrels WHERE source = ?", (SOURCE,))
qrels_raw = cursor.fetchall()

# qrels كقاموس
qrels = {}
for qid, doc_id in qrels_raw:
    qrels.setdefault(qid, set()).add(doc_id)

# --- تقييم ---
precisions, recalls, average_precisions, reciprocal_ranks = [], [], [], []
start_time = time.perf_counter()

print(f"\n⚙️ بدء التقييم على {len(queries)} استعلام...\n")

for qid, query_text in tqdm(queries):
    if qid not in qrels:
        continue

    relevant_docs = qrels[qid]
    tokens = preprocessor.preprocess(query_text, return_as_string=False)
    cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

    candidate_doc_ids = set()
    for token in tokens:
        if token in inverted_index:
            candidate_doc_ids.update(inverted_index[token])

    common_doc_ids = list(candidate_doc_ids.intersection(tfidf_doc_ids).intersection(bert_doc_ids))
    if not common_doc_ids:
        continue

    tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
    bert_indices = [bert_id_to_idx[doc_id] for doc_id in common_doc_ids]

    tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
    sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix[tfidf_indices])[0]

    bert_query_vec = embed_text(cleaned_query)
    sims_bert = cosine_similarity(bert_query_vec, bert_vectors[bert_indices])[0]

    final_sims = ALPHA * sims_tfidf + (1 - ALPHA) * sims_bert
    ranked = [(doc_id, final_sims[i]) for i, doc_id in enumerate(common_doc_ids)]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc_id for doc_id, _ in ranked[:TOP_N]]

    hits = [1 if doc in relevant_docs else 0 for doc in top_docs]
    precisions.append(sum(hits) / TOP_N)
    recalls.append(sum(hits) / len(relevant_docs))

    y_true = [1 if doc in relevant_docs else 0 for doc in common_doc_ids]
    y_scores = final_sims
    try:
        ap = average_precision_score(y_true, y_scores)
    except:
        ap = 0.0
    average_precisions.append(ap)

    for rank, doc in enumerate(top_docs, 1):
        if doc in relevant_docs:
            reciprocal_ranks.append(1 / rank)
            break
    else:
        reciprocal_ranks.append(0.0)

end_time = time.perf_counter()
elapsed_time = round(end_time - start_time, 2)

# --- النتائج ---
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
results_df.to_csv("hybrid_evaluation_results.csv", index=False)
with open("hybrid_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

# إغلاق الاتصال
conn.close()
