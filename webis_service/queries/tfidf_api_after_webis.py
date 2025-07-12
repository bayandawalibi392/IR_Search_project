# from flask import Flask, request, jsonify
# import sqlite3
# import joblib
# import numpy as np
# import time
# from sklearn.metrics.pairwise import cosine_similarity
# import sys
# import os
# import textwrap

# # إضافة المسار الجذري حتى نتمكن من استيراد TextPreprocessor
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from TextPreprocessor import TextPreprocessor

# # إعدادات عامة
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد الثابتة
# print("📦 تحميل الموارد...")
# inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
# tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
# tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# # المعالجة والنموذج
# pre = TextPreprocessor()
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # إعداد تطبيق Flask
# app = Flask(__name__)

# @app.route('/search-inverted', methods=['POST'])
# def search_inverted():
#     data = request.get_json()
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({"error": "الاستعلام فارغ"}), 400

#     start_time = time.perf_counter()
#     content = pre.preprocess(query)

#     if not content:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة."}), 400

#     # العثور على الوثائق المطابقة للفهرس المعكوس
#     candidate_indices = set()
#     for token in content:
#         if token in inverted_index:
#             candidate_indices.update(inverted_index[token])

#     if not candidate_indices:
#         return jsonify({"message": "❌ لا توجد وثائق مطابقة لهذا الاستعلام في الفهرس."}), 200

#     candidate_indices = sorted(candidate_indices)
#     candidate_vectors = tfidf_matrix[candidate_indices]
#     query_vec = tfidf_vectorizer.transform([' '.join(content)])

#     # حساب التشابه
#     scores = cosine_similarity(query_vec, candidate_vectors)[0]
#     top_indices_local = np.argsort(scores)[-TOP_K:][::-1]
#     top_indices_global = [candidate_indices[i] for i in top_indices_local]

#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time

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
#             "score": round(score, 4),
#             "content": textwrap.shorten(content, width=300)
#         })

#     return jsonify({
#         "execution_time": round(elapsed_time, 4),
#         "results": results
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
import sqlite3
import joblib
import numpy as np
import time
import requests
import textwrap
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
GROUP = 'webis'
TOP_K = 10
PREPROCESS_API_URL = "http://localhost:5050/preprocess"  # 🔁 الربط مع خدمة TextPreprocessor

# تحميل الموارد
print("📦 تحميل الموارد...")
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تطبيق Flask
app = Flask(__name__)
CORS(app)
@app.route('/search-inverted', methods=['POST'])
def search_inverted():
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "الاستعلام فارغ"}), 400

    # 🔁 طلب المعالجة النصية من الخدمة الخارجية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "use_stemming": True,
            "use_lemmatization": False
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية"}), 500
        content = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not content:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    start_time = time.perf_counter()

    # العثور على الوثائق المطابقة
    candidate_indices = set()
    for token in content:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])

    if not candidate_indices:
        return jsonify({"message": "❌ لا توجد وثائق مطابقة لهذا الاستعلام"}), 200

    candidate_indices = sorted(candidate_indices)
    candidate_vectors = tfidf_matrix[candidate_indices]
    query_vec = tfidf_vectorizer.transform([' '.join(content)])

    scores = cosine_similarity(query_vec, candidate_vectors)[0]
    top_indices_local = np.argsort(scores)[-TOP_K:][::-1]
    top_indices_global = [candidate_indices[i] for i in top_indices_local]

    elapsed_time = time.perf_counter() - start_time

    results = []
    for rank, idx in enumerate(top_indices_global, 1):
        doc_id = doc_ids[idx]
        score = scores[top_indices_local[rank - 1]]
        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
        row = cursor.fetchone()
        doc_content = row[0] if row else "(لم يتم العثور على النص)"
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(score, 4),
            "content": textwrap.shorten(doc_content, width=300)
        })

    return jsonify({
        "execution_time": round(float(elapsed_time), 4),
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002)
