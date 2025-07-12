# from flask import Flask, request, jsonify
# import os
# import sqlite3
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.tokenize import word_tokenize
# from rank_bm25 import BM25Okapi
# import requests

# # إعدادات
# MODELS_DIR = "models"
# TOP_N = 10
# ALPHA = 0.5  # وزن TF-IDF مقابل BM25
# PREPROCESS_API_URL = "http://localhost:5050/preprocess"  # رابط خدمة المعالجة النصية

# # المصدر الوحيد الحالي
# GROUPS = ['webis']

# # إنشاء تطبيق Flask
# app = Flask(__name__)

# # قاعدة البيانات
# conn = sqlite3.connect("ir_project.db", check_same_thread=False)
# cursor = conn.cursor()

# # تحميل الموارد
# resources = {}
# for group in GROUPS:
#     try:
#         tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_vectorizer_{group}.joblib"))
#         tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"doc_ids_{group}.joblib"))
#         tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_vectors_{group}.joblib"))

#         bm25 = joblib.load(os.path.join(MODELS_DIR, f"bm25_model_{group}.joblib"))
#         bm25_doc_ids = bm25['doc_ids']
#         tokenized_docs = bm25['tokenized_texts']
#         # bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

#         resources[group] = {
#             "tfidf_vectorizer": tfidf_vectorizer,
#             "tfidf_doc_ids": tfidf_doc_ids,
#             "tfidf_matrix": tfidf_matrix,
#             "bm25": bm25,
#             "bm25_doc_ids": bm25_doc_ids
#         }
#         print(f"✅ تم تحميل الموارد للمصدر: {group}")
#     except Exception as e:
#         print(f"⚠️ فشل تحميل الموارد للمصدر {group}: {e}")

# @app.route("/search-hybrid", methods=["POST"])
# def search_hybrid():
#     data = request.get_json()
#     query = data.get("query", "").strip()
#     source = "webis"  # 👈 ثبّتنا المصدر هنا بدل أن يكون ديناميكياً

#     if not query:
#         return jsonify({"error": "⚠️ يجب إرسال حقل 'query' في الطلب"}), 400

#     if source not in resources:
#         return jsonify({"error": f"⚠️ المصدر غير مدعوم. المصادر المتاحة: {GROUPS}"}), 400

#     # 🔁 المعالجة النصية
#     try:
#         response = requests.post(PREPROCESS_API_URL, json={
#             "text": query,
#             "return_as_string": True
#         })
#         if response.status_code != 200:
#             return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية الخارجية"}), 500
#         cleaned_query = response.json().get("clean_text", "")
#     except Exception as e:
#         return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

#     if not cleaned_query.strip():
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

#     # استرجاع الموارد
#     tfidf_vectorizer = resources[source]["tfidf_vectorizer"]
#     tfidf_doc_ids = resources[source]["tfidf_doc_ids"]
#     tfidf_matrix = resources[source]["tfidf_matrix"]
#     bm25 = resources[source]["bm25"]
#     bm25_doc_ids = resources[source]["bm25_doc_ids"]

#     # 🔎 تمثيل الاستعلام
#     tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
#     sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]

#     tokenized_query = word_tokenize(cleaned_query)
#     sims_bm25 = bm25.get_scores(tokenized_query)

#     # دمج النتائج
#     tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
#     bm25_id_to_idx = {doc_id: i for i, doc_id in enumerate(bm25_doc_ids)}
#     common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bm25_doc_ids)))

#     tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
#     bm25_indices = [bm25_id_to_idx[doc_id] for doc_id in common_doc_ids]

#     aligned_sims_tfidf = sims_tfidf[tfidf_indices]
#     aligned_sims_bm25 = np.array([sims_bm25[i] for i in bm25_indices])

#     final_sims = ALPHA * aligned_sims_tfidf + (1 - ALPHA) * aligned_sims_bm25
#     start = time.perf_counter()
#     top_indices = np.argsort(final_sims)[::-1][:TOP_N]
#     elapsed = time.perf_counter() - start
#     results = []
#     for rank, idx in enumerate(top_indices, 1):
#         doc_id = common_doc_ids[idx]
#         score = final_sims[idx]

#         cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, "webis"))
#         row = cursor.fetchone()
#         content = row[0] if row else "(لا يوجد نص)"

#         results.append({
#             "rank": rank,
#             "doc_id": doc_id,
#             "hybrid_similarity": round(float(score), 4),
#             "content": content[:500]
#         })
    
#     return jsonify({
#         "query": query,
#         "cleaned_query": cleaned_query,
#         "source": source,
#         "top_n": TOP_N,
#         "execution_time": round(float(elapsed), 4),
#         "results": results
#     })

# if __name__ == "__main__":
#     app.run(port=5021, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, sqlite3, joblib, numpy as np, time, textwrap
from sklearn.metrics.pairwise import cosine_similarity

# إعداد Flask
app = Flask(__name__)
CORS(app)

# إعدادات
PREPROCESS_API = "http://localhost:5050/preprocess"
GROUP = 'webis'
TOP_K = 10
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'

# تحميل الموارد
print("📦 تحميل النماذج والفهارس...")
# TF-IDF
inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
tfidf_vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{GROUP}.joblib")
tfidf_matrix = joblib.load(f"{MODEL_DIR}/tfidf_vectors_{GROUP}.joblib")
doc_ids_tfidf = joblib.load(f"{MODEL_DIR}/doc_ids_{GROUP}.joblib")

# BM25
bm25 = joblib.load(f"{MODEL_DIR}/bm25_model_{GROUP}.joblib")
doc_ids_bm25 = joblib.load(f"{MODEL_DIR}/doc_ids_bm25_{GROUP}.joblib")

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/search-hybrid-parallel', methods=['POST'])
def hybrid_parallel():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    # 🔁 إرسال الاستعلام لخدمة المعالجة النصية فقط
    try:
        pre_resp = requests.post(PREPROCESS_API, json={
            "text": query,
            "use_stemming": True,
            "use_lemmatization": False
        })
        if pre_resp.status_code != 200:
            return jsonify({"error": "⚠️ فشل المعالجة النصية"}), 500
        tokens = pre_resp.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"❌ خطأ أثناء الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    start = time.perf_counter()

    # 🔹 حساب نتائج TF-IDF (مقيد بالفهارس المعكوسة)
    candidate_indices = set()
    for token in tokens:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])

    tfidf_scores = {}
    if candidate_indices:
        candidate_indices = sorted(candidate_indices)
        candidate_vectors = tfidf_matrix[candidate_indices]
        query_vec = tfidf_vectorizer.transform([' '.join(tokens)])
        similarities = cosine_similarity(query_vec, candidate_vectors)[0]
        for i, idx in enumerate(candidate_indices):
            tfidf_scores[idx] = similarities[i]

    # 🔹 حساب نتائج BM25 (على كل الوثائق)
    bm25_all_scores = bm25.get_scores(tokens)
    bm25_scores = {i: bm25_all_scores[i] for i in range(len(bm25_all_scores))}

    # 🔹 دمج النتائج باستخدام CombSUM
    combined_scores = {}
    for idx, score in tfidf_scores.items():
        combined_scores[idx] = combined_scores.get(idx, 0) + score
    for idx, score in bm25_scores.items():
        combined_scores[idx] = combined_scores.get(idx, 0) + score

    # 🔹 ترتيب النتائج
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

    results = []
    for rank, (idx, score) in enumerate(ranked, 1):
        # نحاول تحديد المصدر الصحيح للـ doc_id
        doc_id = None
        if idx < len(doc_ids_tfidf):
            doc_id = doc_ids_tfidf[idx]
        elif idx < len(doc_ids_bm25):
            doc_id = doc_ids_bm25[idx]

        if not doc_id:
            continue

        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
        row = cursor.fetchone()
        content = row[0] if row else "(لا يوجد نص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": textwrap.shorten(content, width=300)
        })

    elapsed = time.perf_counter() - start
    return jsonify({
        "execution_time": round(float(elapsed), 4),
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5021)
