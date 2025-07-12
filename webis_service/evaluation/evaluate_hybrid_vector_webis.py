import sqlite3
import joblib
import numpy as np
import time
import faiss
import warnings
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from TextPreprocessor import TextPreprocessor
from sentence_transformers import SentenceTransformer
import pandas as pd

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
GROUP = 'webis'
TOP_K = 10
ALPHA = 0.7  # معامل الدمج بين TF-IDF و BERT

# تقليل استهلاك المعالج
torch.set_num_threads(4)

print("📦 تحميل النماذج والبيانات...")

# تحميل البيانات
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
tfidf_doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

bert_doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")
bert_id_to_index = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

faiss_index = faiss.read_index(f"{INDEX_DIR}/faiss_index_{GROUP}_bert.index")
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

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
bert_model = SentenceTransformer('models/bert_model_webis', device='cpu')

map_scores, mrr_scores, recall_scores, precision_scores = [], [], [], []
start_time = time.perf_counter()

for query_id, query_text in query_dict.items():
    tokens = pre.preprocess(query_text, use_stemming=True, use_lemmatization=False)
    if not tokens:
        continue

    query_str = pre.clean_text(' '.join(tokens))
    query_vec_tfidf = tfidf_vectorizer.transform([query_str])
    query_vec_bert = bert_model.encode([' '.join(tokens)]).astype('float32')

    # مرشحي TF-IDF
    candidate_indices = set()
    for token in tokens:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])
    if not candidate_indices:
        continue

    candidate_indices = sorted(candidate_indices)
    candidate_tfidf = tfidf_matrix[candidate_indices]
    candidate_doc_ids = [tfidf_doc_ids[i] for i in candidate_indices]

    # تشابه TF-IDF
    sim_tfidf = cosine_similarity(query_vec_tfidf, candidate_tfidf)[0]

    # استرجاع ترتيب FAISS لجميع الوثائق
    _, faiss_indices = faiss_index.search(query_vec_bert.reshape(1, -1), len(bert_doc_ids))
    faiss_ranks = faiss_indices[0]
    bert_idx_to_rank = {idx: rank for rank, idx in enumerate(faiss_ranks)}

    # تشابه BERT كـ (1 - ترتيب الوثيقة / العدد الكلي)
    sim_bert = []
    for doc_id in candidate_doc_ids:
        bert_idx = bert_id_to_index.get(doc_id)
        if bert_idx is not None:
            rank = bert_idx_to_rank.get(bert_idx, len(bert_doc_ids))
            sim = 1 - (rank / len(bert_doc_ids))
        else:
            sim = 0
        sim_bert.append(sim)
    sim_bert = np.array(sim_bert)

    # دمج التمثيلين
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

end_time = time.perf_counter()
elapsed = round(end_time - start_time, 2)

# عرض النتائج
print("\n✅ التقييم المكتمل باستخدام Hybrid (TF-IDF + FAISS-BERT)\n")
df = pd.DataFrame({
    "Metric": ["MAP", "MRR", f"Recall@{TOP_K}", f"Precision@{TOP_K}", "Execution Time (s)"],
    "Score": [
        round(np.mean(map_scores), 4),
        round(np.mean(mrr_scores), 4),
        round(np.mean(recall_scores), 4),
        round(np.mean(precision_scores), 4),
        elapsed
    ]
})
print(df.to_string(index=False))

# حفظ النتيجة