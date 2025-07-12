import threading
import time
from flask import Flask, request, jsonify
import os
import sqlite3
import joblib
import numpy as np
import requests
import faiss
import textwrap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel
import torch
from flask_cors import CORS
# إعدادات عامة
MODELS_DIR = "models"
DB_PATH = "ir_project.db"
INDEX_DIR="indexes"
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"
TOP_N = 10
ALPHA = 0.5
SOURCE = "quora"

app = Flask(__name__)
CORS(app)

# إعداد BERT
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

resources = {}
for source in [SOURCE]:
    try:
        resources[source] = {
            "inverted_index": joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{source}.joblib")),
            "tfidf_vectorizer": joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_vectorizer.joblib")),
            "tfidf_doc_ids": joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_doc_ids.joblib")),
            "tfidf_matrix": joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_matrix.joblib")),
            "bert_doc_ids": joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_doc_ids.joblib")),
            "bert_vectors": joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_vectors.joblib")),
            "bm25_data": joblib.load(os.path.join(MODELS_DIR, f"bm25_{source}_model.joblib")),
        }
    except Exception as e:
        print(f"⚠️ فشل تحميل بيانات {source}: {e}")

def embed_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy()

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

def run_tfidf_api():
    app = Flask("tfidf_api")
    CORS(app)
    vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
    doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
    doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

    @app.route('/search-tfidf', methods=['POST'])
    def search_tfidf():
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query missing"}), 400
        start_time = time.perf_counter()
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": True})
        cleaned_query = response.json().get("clean_text", "")
        query_vec = vectorizer.transform([cleaned_query])
        sims = cosine_similarity(query_vec, doc_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:TOP_N]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = doc_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims[idx]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})

    app.run(port=5010)

def run_bert_api():
    app = Flask("bert_api")
    CORS(app)
    doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
    doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

    @app.route('/search-bert', methods=['POST'])
    def search_bert():
        data = request.get_json()
        query = data.get("query", "").strip()
        start_time = time.perf_counter()
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": True})
        cleaned_query = response.json().get("clean_text", "")
        query_vec = embed_text(cleaned_query)
        sims = cosine_similarity(query_vec, doc_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:TOP_N]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = doc_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims[idx]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})

    app.run(port=5011)

def run_bm25_api():
    app = Flask("bm25_api")
    CORS(app)
    bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
    doc_ids = bm25_data['doc_ids']
    tokenized_docs = bm25_data['tokenized_texts']
    bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

    @app.route('/search-bm25', methods=['POST'])
    def search_bm25():
        data = request.get_json()
        query = data.get("query", "").strip()
        start_time = time.perf_counter()
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": False})
        tokens = response.json().get("tokens", [])
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:TOP_N]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = doc_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(scores[idx]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})

    app.run(port=5017)


def run_hybrid_parallel_api():
    app = Flask("hybrid_parallel_api")
    CORS(app)
    # تحميل الموارد مرة واحدة
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
    tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
    tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))
    bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
    bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

    # تأكيد تطابق الوثائق
    common_ids = list(set(tfidf_doc_ids) & set(bert_doc_ids))
    tfidf_id_to_index = {doc_id: idx for idx, doc_id in enumerate(tfidf_doc_ids)}
    bert_id_to_index = {doc_id: idx for idx, doc_id in enumerate(bert_doc_ids)}
    tfidf_idx = [tfidf_id_to_index[i] for i in common_ids]
    bert_idx = [bert_id_to_index[i] for i in common_ids]

    @app.route('/search-hybrid-parallel', methods=['POST'])
    def search_hybrid_parallel():
        data = request.get_json()
        query = data.get("query", "").strip()
        alpha = float(data.get("alpha", 0.5))

        if not query:
            return jsonify({"error": "query is missing"}), 400
        start_time = time.perf_counter()
        # تنظيف الاستعلام
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": True})
        cleaned_query = response.json().get("clean_text", "")

        # تمثيل TF-IDF و BERT
        tfidf_vec = tfidf_vectorizer.transform([cleaned_query])
        sims_tfidf = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
        bert_vec = embed_text(cleaned_query)
        sims_bert = cosine_similarity(bert_vec, bert_vectors)[0]
        # استخدم فقط الوثائق المشتركة بين الطريقتين
        sims_combined = alpha * sims_tfidf[tfidf_idx] + (1 - alpha) * sims_bert[bert_idx]

        # ترتيب وتهيئة النتائج
        top_indices = np.argsort(sims_combined)[::-1][:TOP_N]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = common_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims_combined[idx]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})

    app.run(port=5012)



def run_hybrid_tf_bm25_api():
    app = Flask("hybrid_tf_bm25_api")
    CORS(app)
    # تحميل الموارد مرة واحدة
    tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
    tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
    tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

    bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
    bm25_doc_ids = bm25_data['doc_ids']
    tokenized_docs = bm25_data['tokenized_texts']
    bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

    # إعداد الخرائط لمرة واحدة لتسريع الفهرسة
    tfidf_id_to_index = {doc_id: idx for idx, doc_id in enumerate(tfidf_doc_ids)}
    bm25_id_to_index = {doc_id: idx for idx, doc_id in enumerate(bm25_doc_ids)}

    # استخراج الوثائق المشتركة وفهارسها في كلا التمثيلين
    common_ids = list(set(tfidf_doc_ids) & set(bm25_doc_ids))
    tfidf_idx = [tfidf_id_to_index[doc_id] for doc_id in common_ids]
    bm25_idx = [bm25_id_to_index[doc_id] for doc_id in common_ids]

    @app.route('/search-hybrid', methods=['POST'])
    def search_hybrid():
        data = request.get_json()
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query missing"}), 400
        start_time = time.perf_counter()
        # تنظيف الاستعلام
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": True})
        cleaned_query = response.json().get("clean_text", "")
        if not cleaned_query:
            return jsonify({"error": "empty query after preprocessing"}), 400

        # حساب التشابه
        tfidf_vec = tfidf_vectorizer.transform([cleaned_query])
        sims_tfidf = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
        sims_bm25 = bm25.get_scores(word_tokenize(cleaned_query))
        # دمج النتائج
        sims_combined = ALPHA * sims_tfidf[tfidf_idx] + (1 - ALPHA) * np.array([sims_bm25[i] for i in bm25_idx])

        # استخراج أفضل النتائج
        top_indices = np.argsort(sims_combined)[::-1][:TOP_N]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = common_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims_combined[idx]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})

    app.run(port=5018)


def create_app_tfidf_inverted():
    app = Flask("tfidf_inverted")
    CORS(app)
    @app.route('/search-tfidf-inverted', methods=['POST'])
    def tfidf_search():
        data = request.get_json()
        query = data.get("query", "").strip()
        source = data.get("source", "quora")

        if not query or source not in resources:
            return jsonify({"error": "invalid request"}), 400
        start_time = time.perf_counter()
        res = resources[source]
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": False})
        tokens = response.json().get("tokens", [])
        if not tokens:
            return jsonify({"results": [], "message": "no tokens"})

        inv = res['inverted_index']
        candidates = set()
        for tok in tokens:
            if tok in inv:
                candidates.update(inv[tok])
        tfidf_idx = res["tfidf_doc_ids"]
        id_map = {doc_id: idx for idx, doc_id in enumerate(tfidf_idx)}
        indices = [id_map[i] for i in candidates if i in id_map]
        vec = res["tfidf_vectorizer"].transform([" ".join(tokens)])
        sims = cosine_similarity(vec, res["tfidf_matrix"][indices])[0]
        top = np.argsort(sims)[::-1][:TOP_N]
        results = []
        for rank, i in enumerate(top, 1):
            doc_id = tfidf_idx[indices[i]]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims[i]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time    
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})
    app.run(port=5013)

def create_app_bm25_inverted():
    app = Flask("bm25_inverted")
    CORS(app)
    @app.route('/search-bm25-inverted', methods=['POST'])
    def bm25_search():
        data = request.get_json()
        query = data.get("query", "").strip()
        source = data.get("source", "quora")

        if not query or source not in resources:
            return jsonify({"error": "invalid request"}), 400


        start_time = time.perf_counter()
        res = resources[source]
        tokens = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": False}).json().get("tokens", [])
        inv = res['inverted_index']
        doc_ids = res["bm25_data"]["doc_ids"]
        id_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        bm25 = BM25Okapi(res["bm25_data"]["tokenized_texts"], k1=res["bm25_data"]["k1"], b=res["bm25_data"]["b"])

        candidates = set()
        for tok in tokens:
            if tok in inv:
                candidates.update(inv[tok])
        indices = [id_map[i] for i in candidates if i in id_map]
        scores = bm25.get_scores(tokens)
        ranked = sorted([(i, scores[i]) for i in indices], key=lambda x: x[1], reverse=True)[:TOP_N]
        results = []
        for rank, (idx, score) in enumerate(ranked, 1):
            doc_id = doc_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(score), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})
    app.run(port=5016)

def create_app_bert_inverted():
    app = Flask("bert_inverted")
    CORS(app)
    @app.route('/search-bert-inverted', methods=['POST'])
    def bert_search():
        data = request.get_json()
        query = data.get("query", "").strip()
        source = data.get("source", "quora")

        if not query or source not in resources:
            return jsonify({"error": "invalid request"}), 400


        start_time = time.perf_counter()
        res = resources[source]
        tokens = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": False}).json().get("tokens", [])
        inv = res['inverted_index']
        doc_ids = res["bert_doc_ids"]
        vecs = res["bert_vectors"]
        id_map = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        candidates = set()
        for tok in tokens:
            if tok in inv:
                candidates.update(inv[tok])
        indices = [id_map[i] for i in candidates if i in id_map]

        query_vec = embed_text(" ".join(tokens))
        sims = cosine_similarity(query_vec, vecs[indices])[0]
        top = np.argsort(sims)[::-1][:TOP_N]
        results = []
        for rank, i in enumerate(top, 1):
            doc_id = doc_ids[indices[i]]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims[i]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})
    app.run(port=5014)

def create_app_hybrid_inverted():
    app = Flask("hybrid_inverted")
    CORS(app)
    @app.route('/search-hybrid-inverted', methods=['POST'])
    def hybrid_search():
        data = request.get_json()
        query = data.get("query", "").strip()
        source = data.get("source", "quora")

        if not query or source not in resources:
            return jsonify({"error": "invalid request"}), 400
        start_time = time.perf_counter()
        res = resources[source]
        tokens = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": False}).json().get("tokens", [])
        inv = res["inverted_index"]
        tfidf_ids = res["tfidf_doc_ids"]
        bert_ids = res["bert_doc_ids"]
        common_ids = set(tfidf_ids) & set(bert_ids)
        candidates = set()
        for tok in tokens:
            if tok in inv:
                candidates.update(inv[tok])
        filtered_ids = list(candidates & common_ids)
        if not filtered_ids:
            return jsonify({"results": []})

        tfidf_map = {doc_id: i for i, doc_id in enumerate(tfidf_ids)}
        bert_map = {doc_id: i for i, doc_id in enumerate(bert_ids)}
        tfidf_vec = res["tfidf_vectorizer"].transform([" ".join(tokens)])
        tfidf_sims = cosine_similarity(tfidf_vec, res["tfidf_matrix"][[tfidf_map[i] for i in filtered_ids]])[0]
        bert_vec = embed_text(" ".join(tokens))
        bert_sims = cosine_similarity(bert_vec, res["bert_vectors"][[bert_map[i] for i in filtered_ids]])[0]
        sims = ALPHA * tfidf_sims + (1 - ALPHA) * bert_sims
        top = np.argsort(sims)[::-1][:TOP_N]
        results = []
        for rank, idx in enumerate(top, 1):
            doc_id = filtered_ids[idx]
            cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
            row = cursor.fetchone()
            results.append({
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(sims[idx]), 4),
                "content": row[0][:300] if row else ""
            })
        elapsed_time = time.perf_counter() - start_time
        return jsonify({"execution_time": round(elapsed_time, 4),"results": results})
    app.run(port=5015)

def run_hybrid_indexed_faiss_api():
    app = Flask("hybrid_indexed_faiss")
    CORS(app)
    # تحميل نموذج BERT
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")

    # تحميل الموارد لكل مجموعة
    vector_stores = {}
    for source in [SOURCE]:
        print(f"📦 تحميل الموارد للمجموعة: {source}")
        vector_stores[source] = {
            'bert_embeddings': joblib.load(f"{MODELS_DIR}/bert_{source}_vectors.joblib"),
            'doc_ids': joblib.load(f"{MODELS_DIR}/bert_{source}_doc_ids.joblib"),
            'tfidf': joblib.load(f"{MODELS_DIR}/tfidf_{source}_vectorizer.joblib"),
            'tfidf_matrix': joblib.load(f"{MODELS_DIR}/tfidf_{source}_matrix.joblib"),
            'inverted_index': joblib.load(f"{INDEX_DIR}/inverted_index_{source}.joblib"),
            'faiss_index': faiss.read_index(f"{INDEX_DIR}/faiss_index_{source}_bert.index")
        }

    @app.route('/search-hybrid-indexed', methods=['POST'])
    def hybrid_indexed_search():
        data = request.get_json()
        query_text = data.get('query', '').strip()

        if not query_text:
            return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400
        start_time = time.perf_counter()
        # معالجة الاستعلام
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

        query_cleaned = ' '.join(tokens)
        query_vec_bert = bert_model.encode([query_cleaned])[0].astype('float32').reshape(1, -1)

        final_results = []

        for source in [SOURCE]:
            store = vector_stores[source]
            doc_ids = store['doc_ids']
            doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
            tfidf_vectorizer = store['tfidf']
            tfidf_matrix = store['tfidf_matrix']
            inverted_index = store['inverted_index']
            faiss_index = store['faiss_index']

            # استخراج مؤشرات المرشحين من inverted index
            candidate_indices = set()
            for token in tokens:
                if token in inverted_index:
                    for doc_id in inverted_index[token]:
                        idx = doc_id_to_index.get(doc_id)
                        if idx is not None:
                            candidate_indices.add(idx)

            if not candidate_indices:
                continue

            candidate_indices = sorted(candidate_indices)
            tfidf_candidates = tfidf_matrix[candidate_indices]
            scores_tfidf = cosine_similarity(tfidf_vectorizer.transform([query_cleaned]), tfidf_candidates)[0]

            # استعلام faiss
            D, I = faiss_index.search(query_vec_bert, len(doc_ids))
            faiss_scores = 1 - D[0]

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

            top_results = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:TOP_N]

            for rank, (doc_id, score) in enumerate(top_results, 1):
                cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
                row = cursor.fetchone()
                content = row[0] if row else "(لم يتم العثور على النص)"
                final_results.append({
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": round(float(score), 4),
                    "content": textwrap.shorten(content, width=300),
                    "source": source
                })

        elapsed_time = time.perf_counter() - start_time
        return jsonify({
            "query": query_text,
            "execution_time": round(elapsed_time, 4),
            "results": final_results
        })

    app.run(port=5019)

# ---------------- Run all APIs ----------------
if __name__ == "__main__":
    print("🚀 بدء تشغيل جميع APIs في سكربت واحد ...")
    threads = [
        threading.Thread(target=run_tfidf_api),
        threading.Thread(target=run_bert_api),
        threading.Thread(target=run_bm25_api),
        threading.Thread(target=run_hybrid_parallel_api),
        threading.Thread(target=run_hybrid_tf_bm25_api),
        threading.Thread(target=create_app_tfidf_inverted),
        threading.Thread(target=create_app_bm25_inverted),
        threading.Thread(target=create_app_bert_inverted),
        threading.Thread(target=create_app_hybrid_inverted),
        threading.Thread(target=run_hybrid_indexed_faiss_api),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()