# from flask import Flask, request, jsonify
# import sqlite3
# import joblib
# import numpy as np
# import time
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import textwrap

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد
# print("📦 تحميل التمثيلات BERT...")
# bert_model = SentenceTransformer('all-MiniLM-L6-v2')  # نفس الموديل المستخدم سابقًا
# doc_embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")
# doc_embeddings = np.array(doc_embeddings)

# # اتصال قاعدة البيانات
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # Flask app
# app = Flask(__name__)

# @app.route('/search-bert', methods=['POST'])
# def search_bert():
#     data = request.get_json()
#     query = data.get('query', '').strip()

#     if not query:
#         return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

#     start_time = time.perf_counter()

#     # تمثيل الاستعلام
#     query_embedding = bert_model.encode([query])[0].reshape(1, -1)

#     # حساب التشابه
#     scores = cosine_similarity(query_embedding, doc_embeddings)[0]
#     top_indices = np.argsort(scores)[-TOP_K:][::-1]

#     elapsed_time = time.perf_counter() - start_time

#     results = []
#     for rank, idx in enumerate(top_indices, 1):
#         doc_id = doc_ids[idx]
#         score = scores[idx]

#         cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
#         row = cursor.fetchone()
#         content = row[0] if row else "(لم يتم العثور على النص)"

#         results.append({
#            "rank": rank,
#            "doc_id": doc_id,
#            "score": round(float(score), 4),  # ✅ هذا هو التعديل
#            "content": textwrap.shorten(content, width=300)
#     })


#     return jsonify({
#         "execution_time": round(elapsed_time, 4),
#         "results": results
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
# bert_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, joblib, numpy as np, time, textwrap
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
GROUP = 'webis'
TOP_K = 10

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")
doc_embeddings = np.array(doc_embeddings)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/query/bert', methods=['POST'])
def search_bert():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    query_embedding = bert_model.encode([query])[0].reshape(1, -1)
    start = time.perf_counter()
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    elapsed = time.perf_counter() - start

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = doc_ids[idx]
        score = scores[idx]
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
    app.run(debug=True, port=5001)
