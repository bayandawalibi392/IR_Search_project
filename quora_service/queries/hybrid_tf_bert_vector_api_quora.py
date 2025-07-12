from flask import Flask, request, jsonify
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import sqlite3
import textwrap
import time
import requests
import os

# إعداد Flask
app = Flask(__name__)

# إعدادات
MODEL_NAME = 'all-MiniLM-L6-v2'
GROUPS = ['quora']
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
TOP_K = 10
ALPHA = 0.2
DB_PATH = 'ir_project.db'
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"

# تحميل نموذج BERT
print("🔄 تحميل نموذج BERT...")
bert_model = SentenceTransformer(MODEL_NAME)

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تحميل الموارد لكل مجموعة
vector_stores = {}
for group in GROUPS:
    print(f"📦 تحميل الموارد للمجموعة: {group}")
    vector_stores[group] = {
        'bert_embeddings': joblib.load(f"{MODEL_DIR}/bert_{group}_vectors.joblib"),
        'doc_ids': joblib.load(f"{MODEL_DIR}/bert_{group}_doc_ids.joblib"),
        'tfidf': joblib.load(f"{MODEL_DIR}/tfidf_{group}_vectorizer.joblib"),
        'tfidf_matrix': joblib.load(f"{MODEL_DIR}/tfidf_{group}_matrix.joblib"),
        'inverted_index': joblib.load(f"{INDEX_DIR}/inverted_index_{group}.joblib"),
        'faiss_index': faiss.read_index(f"{INDEX_DIR}/faiss_index_{group}_bert.index")
    }

@app.route('/search-hybrid-indexed', methods=['POST'])
def hybrid_indexed_search():
    data = request.get_json()
    query_text = data.get('query', '').strip()

    if not query_text:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    # إرسال الاستعلام إلى خدمة المعالجة النصية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query_text,
            "use_stemming": True,
            "use_lemmatization": False
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية"}), 500

        tokens = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    start_time = time.perf_counter()

    query_cleaned = ' '.join(tokens)
    query_vec_bert = bert_model.encode([query_cleaned])[0].astype('float32').reshape(1, -1)

    final_results = []

    for group in GROUPS:
        store = vector_stores[group]
        doc_ids = store['doc_ids']
        doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}  # ✅ تحسين السرعة

        tfidf_vectorizer = store['tfidf']
        tfidf_matrix = store['tfidf_matrix']
        inverted_index = store['inverted_index']
        faiss_index = store['faiss_index']

        # استرجاع المؤشرات من الـ inverted index
        candidate_indices = set()
        for token in tokens:
            if token in inverted_index:
                for doc_id in inverted_index[token]:
                    index = doc_id_to_index.get(doc_id)
                    if index is not None:
                        candidate_indices.add(index)

        if not candidate_indices:
            continue

        candidate_indices = sorted(candidate_indices)
        tfidf_candidates = tfidf_matrix[candidate_indices]
        scores_tfidf = cosine_similarity(tfidf_vectorizer.transform([query_cleaned]), tfidf_candidates)[0]

        # استعلام FAISS على كامل البيانات
        D, I = faiss_index.search(query_vec_bert, len(doc_ids))
        faiss_doc_indices = I[0]
        faiss_scores = 1 - D[0]  # تحويل المسافات إلى تشابه

        # دمج الدرجات
        hybrid_scores = []
        for local_rank, idx in enumerate(candidate_indices):
            doc_id = doc_ids[idx]
            faiss_idx = doc_id_to_index.get(doc_id)
            if faiss_idx is None:
                continue
            score_bert = faiss_scores[faiss_idx]
            score_tfidf = scores_tfidf[local_rank]
            hybrid_score = ALPHA * score_tfidf + (1 - ALPHA) * score_bert
            hybrid_scores.append((doc_id, hybrid_score))

        # أفضل النتائج
        top_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:TOP_K]

        for rank, (doc_id, score) in enumerate(top_results, 1):
            cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, group))
            row = cursor.fetchone()
            content = row[0] if row else "(لم يتم العثور على النص)"
            final_results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(score), 4),
                "content": textwrap.shorten(content, width=300),
                "group": group
            })

    elapsed_time = time.perf_counter() - start_time
    return jsonify({
        "query": query_text,
        "execution_time": round(elapsed_time, 4),
        "results": final_results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5019)
