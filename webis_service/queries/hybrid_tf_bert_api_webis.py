# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from TextPreprocessor import TextPreprocessor
# from sklearn.metrics.pairwise import cosine_similarity
# import sqlite3
# import textwrap
# import time

# # إعدادات
# MODEL_NAME = 'all-MiniLM-L6-v2'
# GROUPS = ['webis']
# MODEL_DIR = 'models'
# TOP_K = 5
# ALPHA = 0.5
# DB_PATH = 'ir_project.db'

# # تحميل النموذج
# print("🔄 تحميل نموذج BERT...")
# bert_model = SentenceTransformer(MODEL_NAME)

# # تحميل المعالجة
# pre = TextPreprocessor()

# # اتصال قاعدة البيانات
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # تحميل التمثيلات
# vector_stores = {}
# for group in GROUPS:
#     print(f"📦 تحميل التمثيلات للمجموعة: {group}")
#     bert_embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{group}.joblib")
#     doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{group}.joblib")
#     tfidf = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{group}.joblib")
#     tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{group}.joblib")

#     vector_stores[group] = {
#         'bert_embeddings': np.array(bert_embeddings),
#         'doc_ids': doc_ids,
#         'tfidf': tfidf,
#         'tfidf_matrix': tfidf_matrix
#     }

# # إنشاء تطبيق Flask
# app = Flask(__name__)

# @app.route('/search-hybrid-parallel', methods=['POST'])
# def search_hybrid_parallel():
#     data = request.get_json()
#     query = data.get('query', '').strip()
#     top_k = int(data.get('top_k', TOP_K))

#     if not query:
#         return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

#     start_time = time.perf_counter()

#     # المعالجة
#     tokens = pre.preprocess(query)
#     if not tokens:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

#     query_text = ' '.join(tokens)
#     query_vec_bert = bert_model.encode([query_text])[0].reshape(1, -1)

#     results_all_groups = []

#     for group in GROUPS:
#         doc_ids = vector_stores[group]['doc_ids']
#         bert_embeddings = vector_stores[group]['bert_embeddings']
#         tfidf_vectorizer = vector_stores[group]['tfidf']
#         tfidf_matrix = vector_stores[group]['tfidf_matrix']

#         query_vec_tfidf = tfidf_vectorizer.transform([query_text])
#         scores_bert = cosine_similarity(query_vec_bert, bert_embeddings)[0]
#         scores_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix)[0]

#         hybrid_scores = ALPHA * scores_tfidf + (1 - ALPHA) * scores_bert
#         top_k_idx = np.argsort(hybrid_scores)[-top_k:][::-1]

#         group_results = []
#         for rank, idx in enumerate(top_k_idx, 1):
#             doc_id = doc_ids[idx]
#             score = hybrid_scores[idx]

#             cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, group))
#             row = cursor.fetchone()
#             content = row[0] if row else "(لم يتم العثور على النص)"

#             group_results.append({
#                 "rank": rank,
#                 "doc_id": doc_id,
#                 "score": round(float(score), 4),
#                 "content": textwrap.shorten(content, width=300)
#             })

#         results_all_groups.append({
#             "group": group,
#             "results": group_results
#         })

#     elapsed_time = time.perf_counter() - start_time

#     return jsonify({
#         "execution_time": round(elapsed_time, 4),
#         "results": results_all_groups
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
from flask_cors import CORS
# إعدادات
MODEL_NAME = 'all-MiniLM-L6-v2'
GROUPS = ['webis']
MODEL_DIR = 'models'
TOP_K = 10
ALPHA = 0.5
DB_PATH = 'ir_project.db'
PREPROCESS_API_URL = "http://localhost:5050/preprocess"  # 🔁 خدمة المعالجة

# تحميل نموذج BERT
print("🔄 تحميل نموذج BERT...")
bert_model = SentenceTransformer(MODEL_NAME)

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تحميل التمثيلات
vector_stores = {}
for group in GROUPS:
    print(f"📦 تحميل التمثيلات للمجموعة: {group}")
    bert_embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{group}.joblib")
    doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{group}.joblib")
    tfidf = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{group}.joblib")
    tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{group}.joblib")

    vector_stores[group] = {
        'bert_embeddings': np.array(bert_embeddings),
        'doc_ids': doc_ids,
        'tfidf': tfidf,
        'tfidf_matrix': tfidf_matrix
    }

# إعداد تطبيق Flask
app = Flask(__name__)
CORS(app)
@app.route('/search-hybrid-parallel', methods=['POST'])
def search_hybrid_parallel():
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = int(data.get('top_k', TOP_K))

    if not query:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    # 🔁 إرسال الاستعلام إلى خدمة المعالجة النصية الخارجية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
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

    query_text = ' '.join(tokens)
    query_vec_bert = bert_model.encode([query_text])[0].reshape(1, -1)

    results_all_groups = []

    for group in GROUPS:
        doc_ids = vector_stores[group]['doc_ids']
        bert_embeddings = vector_stores[group]['bert_embeddings']
        tfidf_vectorizer = vector_stores[group]['tfidf']
        tfidf_matrix = vector_stores[group]['tfidf_matrix']

        query_vec_tfidf = tfidf_vectorizer.transform([query_text])
        scores_bert = cosine_similarity(query_vec_bert, bert_embeddings)[0]
        scores_tfidf = cosine_similarity(query_vec_tfidf, tfidf_matrix)[0]

        hybrid_scores = ALPHA * scores_tfidf + (1 - ALPHA) * scores_bert
        top_k_idx = np.argsort(hybrid_scores)[-top_k:][::-1]

        group_results = []
        for rank, idx in enumerate(top_k_idx, 1):
            doc_id = doc_ids[idx]
            score = hybrid_scores[idx]

            cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, group))
            row = cursor.fetchone()
            content = row[0] if row else "(لم يتم العثور على النص)"

            group_results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(score), 4),
                "content": textwrap.shorten(content, width=300)
            })

        results_all_groups.append({
            "group": group,
            "results": group_results
        })

    elapsed_time = time.perf_counter() - start_time

    return jsonify({
        "execution_time": round(float(elapsed_time), 4),
        "results": results_all_groups
    })

if __name__ == '__main__':
    app.run(debug=True, port=5003)
