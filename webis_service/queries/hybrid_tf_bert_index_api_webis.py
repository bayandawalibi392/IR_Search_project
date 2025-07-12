# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import sqlite3
# import textwrap
# import time
# import sys
# import os
# import textwrap

# # إعداد المسارات
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from TextPreprocessor import TextPreprocessor

# # إعدادات
# MODEL_NAME = 'all-MiniLM-L6-v2'
# GROUPS = ['webis']
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# TOP_K = 10
# ALPHA = 0.2
# DB_PATH = 'ir_project.db'

# # تهيئة Flask
# app = Flask(__name__)

# # تحميل نموذج BERT والمعالجة
# print("🔄 تحميل نموذج BERT...")
# bert_model = SentenceTransformer(MODEL_NAME)
# pre = TextPreprocessor()
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # تحميل الموارد لكل مجموعة
# vector_stores = {}
# for group in GROUPS:
#     print(f"📦 تحميل الموارد للمجموعة: {group}")
#     vector_stores[group] = {
#         'bert_embeddings': joblib.load(f"{MODEL_DIR}/bert_vectors_{group}.joblib"),
#         'doc_ids': joblib.load(f"{MODEL_DIR}/doc_ids_bert_{group}.joblib"),
#         'tfidf': joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{group}.joblib"),
#         'tfidf_matrix': joblib.load(f"{MODEL_DIR}/tfidf_vectors_{group}.joblib"),
#         'inverted_index': joblib.load(f"{INDEX_DIR}/inverted_index1_{group}.joblib")
#     }

# @app.route('/search-hybrid-indexed', methods=['POST'])
# def hybrid_indexed_search():
#     data = request.get_json()
#     query_text = data.get('query', '').strip()

#     if not query_text:
#         return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

#     start_time = time.perf_counter()
#     tokens = pre.preprocess(query_text)
#     if not tokens:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

#     query_cleaned = ' '.join(tokens)
#     query_vec_bert = bert_model.encode([query_cleaned])[0].reshape(1, -1)

#     final_results = []

#     for group in GROUPS:
#         store = vector_stores[group]
#         doc_ids = store['doc_ids']
#         tfidf_vectorizer = store['tfidf']
#         tfidf_matrix = store['tfidf_matrix']
#         bert_embeddings = store['bert_embeddings']
#         inverted_index = store['inverted_index']

#         # استخدام الفهرس المعكوس للترشيح
#         candidate_indices = set()
#         for token in tokens:
#             if token in inverted_index:
#                 candidate_indices.update(inverted_index[token])
#         if not candidate_indices:
#             continue

#         candidate_indices = sorted(candidate_indices)
#         tfidf_candidates = tfidf_matrix[candidate_indices]
#         scores_tfidf = cosine_similarity(tfidf_vectorizer.transform([query_cleaned]), tfidf_candidates)[0]

#         scores_bert = cosine_similarity(query_vec_bert, bert_embeddings)[0]

#         # الدمج الهجين
#         hybrid_scores = ALPHA * scores_tfidf + (1 - ALPHA) * scores_bert[candidate_indices]
#         top_k_idx = np.argsort(hybrid_scores)[-TOP_K:][::-1]

#         for rank, local_idx in enumerate(top_k_idx, 1):
#             global_idx = candidate_indices[local_idx]
#             doc_id = doc_ids[global_idx]
#             score = hybrid_scores[local_idx]

#             cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, group))
#             row = cursor.fetchone()
#             content = row[0] if row else "(لم يتم العثور على النص)"

#             final_results.append({
#                 "rank": rank,
#                 "doc_id": doc_id,
#                 "score": round(float(score), 4),
#                 "content": textwrap.shorten(content, width=300),
#                 "group": group
#             })

#     elapsed_time = time.perf_counter() - start_time
#     return jsonify({
#         "query": query_text,
#         "execution_time": round(elapsed_time, 4),
#         "results": final_results
#     })

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import textwrap
import time
import requests
import os
import sys
from flask_cors import CORS

# إعداد Flask
app = Flask(__name__)
CORS(app)
# إعدادات
MODEL_NAME = 'all-MiniLM-L6-v2'
GROUPS = ['webis']
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
TOP_K = 10
ALPHA = 0.2
DB_PATH = 'ir_project.db'
PREPROCESS_API_URL = "http://localhost:5050/preprocess"  

# تحميل نموذج BERT
print("🔄 تحميل نموذج BERT...")
bert_model = SentenceTransformer(MODEL_NAME)

# تحميل قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تحميل الموارد لكل مجموعة
vector_stores = {}
for group in GROUPS:
    print(f"📦 تحميل الموارد للمجموعة: {group}")
    vector_stores[group] = {
        'bert_embeddings': joblib.load(f"{MODEL_DIR}/bert_vectors_{group}.joblib"),
        'doc_ids': joblib.load(f"{MODEL_DIR}/doc_ids_bert_{group}.joblib"),
        'tfidf': joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{group}.joblib"),
        'tfidf_matrix': joblib.load(f"{MODEL_DIR}/tfidf_vectors_{group}.joblib"),
        'inverted_index': joblib.load(f"{INDEX_DIR}/inverted_index1_{group}.joblib")
    }

@app.route('/search-hybrid-indexed', methods=['POST'])
def hybrid_indexed_search():
    data = request.get_json()
    query_text = data.get('query', '').strip()

    if not query_text:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    # ✅ إرسال الاستعلام إلى خدمة TextPreprocessor
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
    query_vec_bert = bert_model.encode([query_cleaned])[0].reshape(1, -1)

    final_results = []

    for group in GROUPS:
        store = vector_stores[group]
        doc_ids = store['doc_ids']
        tfidf_vectorizer = store['tfidf']
        tfidf_matrix = store['tfidf_matrix']
        bert_embeddings = store['bert_embeddings']
        inverted_index = store['inverted_index']

        candidate_indices = set()
        for token in tokens:
            if token in inverted_index:
                candidate_indices.update(inverted_index[token])

        if not candidate_indices:
            continue

        candidate_indices = sorted(candidate_indices)
        tfidf_candidates = tfidf_matrix[candidate_indices]
        scores_tfidf = cosine_similarity(tfidf_vectorizer.transform([query_cleaned]), tfidf_candidates)[0]

        scores_bert = cosine_similarity(query_vec_bert, bert_embeddings)[0]

        hybrid_scores = ALPHA * scores_tfidf + (1 - ALPHA) * scores_bert[candidate_indices]
        top_k_idx = np.argsort(hybrid_scores)[-TOP_K:][::-1]

        for rank, local_idx in enumerate(top_k_idx, 1):
            global_idx = candidate_indices[local_idx]
            doc_id = doc_ids[global_idx]
            score = hybrid_scores[local_idx]

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
        "execution_time": round(float(elapsed_time), 4),
        "results": final_results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5005)
