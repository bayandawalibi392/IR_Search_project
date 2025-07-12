# from flask import Flask, request, jsonify
# import sqlite3
# import joblib
# import numpy as np
# import time
# import textwrap
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import sys
# import os

# # إضافة المسار الجذري حتى نتمكن من استيراد TextPreprocessor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from TextPreprocessor import TextPreprocessor


# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد
# print("📦 تحميل الفهرس المعكوس والتمثيلات BERT...")
# inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
# doc_embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')
# doc_embeddings = np.array(doc_embeddings)

# # اتصال قاعدة البيانات
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # Flask app
# app = Flask(__name__)
# pre = TextPreprocessor()

# @app.route('/search-bert-inv', methods=['POST'])
# def search_bert_with_inverted_index():
#     data = request.get_json()
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

#     start_time = time.perf_counter()

#     tokens = pre.preprocess(query)
#     if not tokens:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

#     # استرجاع الوثائق المرشحة
#     candidate_indices = set()
#     for token in tokens:
#         if token in inverted_index:
#             candidate_indices.update(inverted_index[token])

#     if not candidate_indices:
#         return jsonify({"error": "❌ لا توجد وثائق مطابقة لهذا الاستعلام في الفهرس"}), 404

#     candidate_indices = sorted(candidate_indices)
#     candidate_vectors = np.array([doc_embeddings[i] for i in candidate_indices])

#     # تمثيل الاستعلام
#     query_embedding = bert_model.encode([query])[0].reshape(1, -1)

#     # حساب التشابه
#     scores = cosine_similarity(query_embedding, candidate_vectors)[0]
#     top_indices_local = np.argsort(scores)[-TOP_K:][::-1]
#     top_indices_global = [candidate_indices[i] for i in top_indices_local]

#     elapsed_time = time.perf_counter() - start_time

#     results = []
#     for rank, idx in enumerate(top_indices_global, 1):
#         doc_id = doc_ids[idx]
#         score = scores[top_indices_local[rank - 1]]

#         cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
#         row = cursor.fetchone()
#         content = row[0] if row else "(لم يتم العثور على النص)"

#         results.append({
#             "rank": rank,
#             "doc_id": doc_id,
#             "score": round(float(score), 4),
#             "content": textwrap.shorten(content, width=300)
#         })

#     return jsonify({
#         "execution_time": round(elapsed_time, 4),
#         "results": results
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
# bert_inverted_index_api.py


# bert_inverted_index_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, joblib, numpy as np, time, textwrap, requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
GROUP = 'webis'
TOP_K = 10
PREPROCESS_API_URL = "http://localhost:5050/preprocess"

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
doc_embeddings = np.array(doc_embeddings)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/query/bert-inv', methods=['POST'])
def search_bert_inverted():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "use_stemming": True,
            "use_lemmatization": False
        })
        tokens = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    candidate_indices = set()
    for token in tokens:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])

    if not candidate_indices:
        return jsonify({"error": "❌ لا توجد وثائق مطابقة في الفهرس"}), 404
    start = time.perf_counter()
    candidate_indices = sorted(candidate_indices)
    candidate_vectors = np.array([doc_embeddings[i] for i in candidate_indices])

    query_embedding = bert_model.encode([query])[0].reshape(1, -1)
    scores = cosine_similarity(query_embedding, candidate_vectors)[0]
    top_local = np.argsort(scores)[-TOP_K:][::-1]
    top_global = [candidate_indices[i] for i in top_local]
    elapsed = time.perf_counter() - start

    results = []
    for rank, idx in enumerate(top_global, 1):
        doc_id = doc_ids[idx]
        score = scores[top_local[rank - 1]]
        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
        row = cursor.fetchone()
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": textwrap.shorten(row[0] if row else "", width=300)
        })

    return jsonify({"execution_time": round(float(elapsed), 4), "results": results})

if __name__ == '__main__':
    app.run(debug=True, port=5006)
