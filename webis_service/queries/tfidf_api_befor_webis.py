# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import sqlite3
# import joblib
# import numpy as np
# import time
# from sklearn.metrics.pairwise import cosine_similarity
# import sys
# import os
# import textwrap

# # إعداد المسارات
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from TextPreprocessor import TextPreprocessor

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد عند تشغيل السيرفر
# print("📦 تحميل الموديلات...")
# tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
# tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# # إعداد المعالجة والنماذج
# pre = TextPreprocessor()
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # إعداد Flask
# app = Flask(__name__)

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({"error": "الاستعلام فارغ"}), 400

#     # معالجة الاستعلام
#     content = pre.preprocess(query)
#     if not content:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة."}), 400

#     query_text = ' '.join(content)
#     query_vec = tfidf_vectorizer.transform([query_text])

#     # حساب التشابه
#     start_time = time.perf_counter()
#     scores = cosine_similarity(query_vec, tfidf_matrix)[0]
#     top_indices = np.argsort(scores)[-TOP_K:][::-1]
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time

#     results = []
#     for rank, idx in enumerate(top_indices, 1):
#         doc_id = doc_ids[idx]
#         score = scores[idx]
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
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import sqlite3
# import joblib
# import numpy as np
# import time
# from sklearn.metrics.pairwise import cosine_similarity
# import sys
# import os
# import textwrap

# # إعداد المسارات
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from TextPreprocessor import TextPreprocessor

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد عند تشغيل السيرفر
# print("📦 تحميل الموديلات...")
# tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
# tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# # إعداد المعالجة والنماذج
# pre = TextPreprocessor()
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # إعداد Flask مع دعم CORS
# app = Flask(__name__)
# CORS(app)  # هذه السطر يفعّل دعم CORS لجميع الأصول

# @app.route('/search', methods=['POST'])
# def search():
#     data = request.get_json()
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({"error": "الاستعلام فارغ"}), 400

#     # معالجة الاستعلام
#     content = pre.preprocess(query)
#     if not content:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة."}), 400

#     query_text = ' '.join(content)
#     query_vec = tfidf_vectorizer.transform([query_text])

#     # حساب التشابه
#     start_time = time.perf_counter()
#     scores = cosine_similarity(query_vec, tfidf_matrix)[0]
#     top_indices = np.argsort(scores)[-TOP_K:][::-1]
#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time

#     results = []
#     for rank, idx in enumerate(top_indices, 1):
#         doc_id = doc_ids[idx]
#         score = scores[idx]
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




# tfidf_api_befor_webis.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import numpy as np
import time
import requests
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# إعداد Flask
app = Flask(__name__)
CORS(app)

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
GROUP = 'webis'
TOP_K = 10
PREPROCESS_API_URL = "http://localhost:5050/preprocess"

# تحميل الموارد
print("📦 تحميل الموديلات...")
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/query/tfidf', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    # 🔄 استدعاء خدمة TextPreprocessor
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "use_stemming": True,
            "use_lemmatization": False
        })
        print("🔁 Response code:", response.status_code)
        print("📄 Response text:", response.text)
        if response.status_code != 200:
            return jsonify({"error": "⚠️ خدمة المعالجة فشلت"}), 500
        
        tokens = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"⚠️ تعذر الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    query_text = ' '.join(tokens)
    query_vec = tfidf_vectorizer.transform([query_text])

    start_time = time.perf_counter()
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    elapsed_time = time.perf_counter() - start_time

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = doc_ids[idx]
        score = scores[idx]

        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
        row = cursor.fetchone()
        content = row[0] if row else "(لم يتم العثور على النص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(score, 4),
            # "content": textwrap.shorten(content, width=300)
            "content": content

        })

    return jsonify({
        "execution_time": round(float(elapsed_time), 4),
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
