
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import sqlite3, joblib, numpy as np, time, textwrap, requests
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# import faiss

# app = Flask(__name__)
# CORS(app)

# # إعدادات
# MODEL_NAME = 'all-MiniLM-L6-v2'
# GROUPS = ['webis']
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# TOP_K = 10
# ALPHA = 0.2
# DB_PATH = 'ir_project.db'
# PREPROCESS_API_URL = "http://localhost:5050/preprocess"

# # تحميل نموذج BERT
# bert_model = SentenceTransformer(MODEL_NAME)

# # SQLite
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # تحميل الموارد
# vector_stores = {}
# for group in GROUPS:
#     doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{group}.joblib")
#     id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}  # ✅ لتحسين الأداء

#     vector_stores[group] = {
#         'doc_ids': doc_ids,
#         'id_to_idx': id_to_idx,
#         'bert_embeddings': joblib.load(f"{MODEL_DIR}/bert_vectors_{group}.joblib"),
#         'tfidf': joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{group}.joblib"),
#         'tfidf_matrix': joblib.load(f"{MODEL_DIR}/tfidf_vectors_{group}.joblib"),
#         'inverted_index': joblib.load(f"{INDEX_DIR}/inverted_index1_{group}.joblib"),
#         'faiss_index': faiss.read_index(f"{INDEX_DIR}/faiss_index_{group}_bert.index")
#     }

# @app.route('/search-hybrid-indexed', methods=['POST'])
# def hybrid_indexed_search():
#     data = request.get_json()
#     query_text = data.get('query', '').strip()
#     if not query_text:
#         return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

#     try:
#         response = requests.post(PREPROCESS_API_URL, json={
#             "text": query_text,
#             "use_stemming": True,
#             "use_lemmatization": False
#         })
#         if response.status_code != 200:
#             return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية"}), 500
#         tokens = response.json().get("tokens", [])
#     except Exception as e:
#         return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

#     if not tokens:
#         return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

#     start_time = time.perf_counter()
#     query_cleaned = ' '.join(tokens)
#     query_vec_bert = bert_model.encode([query_cleaned])[0].astype('float32').reshape(1, -1)

#     final_results = []

#     for group in GROUPS:
#         store = vector_stores[group]
#         doc_ids = store['doc_ids']
#         id_to_idx = store['id_to_idx']
#         tfidf_vectorizer = store['tfidf']
#         tfidf_matrix = store['tfidf_matrix']
#         inverted_index = store['inverted_index']
#         faiss_index = store['faiss_index']

#         # ✅ استخراج المرشحين من الفهرس المعكوس
#         candidate_indices = set()
#         for token in tokens:
#             candidate_indices.update(inverted_index.get(token, []))
#         if not candidate_indices:
#             continue

#         candidate_indices = sorted(candidate_indices)
#         tfidf_candidates = tfidf_matrix[candidate_indices]
#         scores_tfidf = cosine_similarity(tfidf_vectorizer.transform([query_cleaned]), tfidf_candidates)[0]

#         # ✅ FAISS: استرجاع جميع الوثائق
#         D, I = faiss_index.search(query_vec_bert, len(doc_ids))
#         faiss_scores = 1 - D[0]  # درجة التشابه

#         # ✅ بناء dict {doc_id: score} لـ FAISS
#         faiss_score_dict = {doc_ids[i]: float(faiss_scores[i]) for i in range(len(doc_ids))}

#         hybrid_scores = []
#         for local_rank, idx in enumerate(candidate_indices):
#             doc_id = doc_ids[idx]
#             score_bert = faiss_score_dict.get(doc_id)
#             if score_bert is None:
#                 continue
#             score_tfidf = float(scores_tfidf[local_rank])
#             hybrid_score = ALPHA * score_tfidf + (1 - ALPHA) * score_bert
#             hybrid_scores.append((doc_id, hybrid_score))

#         top_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:TOP_K]

#         # ✅ جلب المحتوى دفعة واحدة
#         doc_id_list = [doc_id for doc_id, _ in top_results]
#         placeholders = ','.join(['?'] * len(doc_id_list))
#         cursor.execute(f"SELECT doc_id, content FROM documents WHERE doc_id IN ({placeholders}) AND source = ?", (*doc_id_list, group))
#         rows = dict(cursor.fetchall())

#         for rank, (doc_id, score) in enumerate(top_results, 1):
#             content = rows.get(doc_id, "(لم يتم العثور على النص)")
#             final_results.append({
#                 "rank": rank,
#                 "doc_id": doc_id,
#                 "score": round(score, 4),
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
#     app.run(debug=True, port=5004)


from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, joblib, numpy as np, time, textwrap, requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss

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
MAX_CANDIDATES = 1000  # 👈 الحد الأقصى لعدد الوثائق المعالجة لكل استعلام

# تحميل نموذج BERT
bert_model = SentenceTransformer(MODEL_NAME)

# SQLite
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تحميل الموارد
vector_stores = {}
for group in GROUPS:
    doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{group}.joblib")
    id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}  # لتحسين الأداء

    vector_stores[group] = {
        'doc_ids': doc_ids,
        'id_to_idx': id_to_idx,
        'bert_embeddings': joblib.load(f"{MODEL_DIR}/bert_vectors_{group}.joblib"),
        'tfidf': joblib.load(f"{MODEL_DIR}/tfidf_vectorizer_{group}.joblib"),
        'tfidf_matrix': joblib.load(f"{MODEL_DIR}/tfidf_vectors_{group}.joblib"),
        'inverted_index': joblib.load(f"{INDEX_DIR}/inverted_index1_{group}.joblib"),
        'faiss_index': faiss.read_index(f"{INDEX_DIR}/faiss_index_{group}_bert.index")
    }

@app.route('/search-hybrid-indexed', methods=['POST'])
def hybrid_indexed_search():
    data = request.get_json()
    query_text = data.get('query', '').strip()
    if not query_text:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

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
        id_to_idx = store['id_to_idx']
        tfidf_vectorizer = store['tfidf']
        tfidf_matrix = store['tfidf_matrix']
        inverted_index = store['inverted_index']
        faiss_index = store['faiss_index']

        # ✅ استخراج المرشحين من الفهرس المعكوس
        candidate_indices = set()
        for token in tokens:
            candidate_indices.update(inverted_index.get(token, []))
        if not candidate_indices:
            continue

        candidate_indices = sorted(candidate_indices)[:MAX_CANDIDATES]  # 👈 تحديد عدد المرشحين

        tfidf_candidates = tfidf_matrix[candidate_indices]
        scores_tfidf = cosine_similarity(tfidf_vectorizer.transform([query_cleaned]), tfidf_candidates)[0]

        # ✅ FAISS: استرجاع جميع الوثائق
        D, I = faiss_index.search(query_vec_bert, len(doc_ids))
        faiss_scores = 1 - D[0]  # تحويل المسافة إلى درجة تشابه

        faiss_score_dict = {doc_ids[i]: float(faiss_scores[i]) for i in range(len(doc_ids))}

        hybrid_scores = []
        for local_rank, idx in enumerate(candidate_indices):
            doc_id = doc_ids[idx]
            score_bert = faiss_score_dict.get(doc_id)
            if score_bert is None:
                continue
            score_tfidf = float(scores_tfidf[local_rank])
            hybrid_score = ALPHA * score_tfidf + (1 - ALPHA) * score_bert
            hybrid_scores.append((doc_id, hybrid_score))

        top_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:TOP_K]

        # ✅ جلب المحتوى دفعة واحدة
        doc_id_list = [doc_id for doc_id, _ in top_results]
        placeholders = ','.join(['?'] * len(doc_id_list))
        cursor.execute(f"SELECT doc_id, content FROM documents WHERE doc_id IN ({placeholders}) AND source = ?", (*doc_id_list, group))
        rows = dict(cursor.fetchall())

        for rank, (doc_id, score) in enumerate(top_results, 1):
            content = rows.get(doc_id, "(لم يتم العثور على النص)")
            final_results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(score, 4),
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
    app.run(debug=True, port=5004)
