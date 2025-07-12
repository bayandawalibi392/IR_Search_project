# import joblib
# import os
# import sqlite3
# import numpy as np
# import time
# from sklearn.metrics.pairwise import cosine_similarity
# from text_preprocessing_service import TextPreprocessingService
# # إعدادات
# INDEX_DIR = 'indexes'
# MODELS_DIR = 'models'
# TOP_N = 10
# SOURCE = "quora"  # أو "quora"

# # تحميل الفهرس المعكوس
# index_path = os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib")
# print(f"📂 تحميل الفهرس المعكوس من: {index_path}")
# inverted_index = joblib.load(index_path)

# # تحميل نموذج TF-IDF
# tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
# doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
# doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

# # خدمة المعالجة النصية
# preprocessor = TextPreprocessingServiceWebis()

# # إنشاء قاموس doc_id إلى index
# doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

# def search_tfidf(query):
#     tokens = preprocessor.preprocess(query, return_as_string=False)
#     print(f"✅ الاستعلام المعالج (tokens): {tokens}")

#     # الوثائق المرشحة من الفهرس
#     candidate_doc_ids = set()
#     for term in tokens:
#         if term in inverted_index:
#             candidate_doc_ids.update(inverted_index[term])

#     if not candidate_doc_ids:
#         print("❌ لم يتم العثور على وثائق.")
#         return

#     print(f"✅ عدد الوثائق المرشحة من الفهرس: {len(candidate_doc_ids)}")
#     candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]

#     if not candidate_indices:
#         print("❌ لا توجد تمثيلات متوفرة لهذه الوثائق.")
#         return

#     # حساب الزمن
#     start_time = time.time()

#     query_vec = tfidf_vectorizer.transform([" ".join(tokens)])
#     sub_doc_matrix = doc_matrix[candidate_indices]
#     sims = cosine_similarity(query_vec, sub_doc_matrix)[0]

#     top_indices = np.argsort(sims)[::-1][:TOP_N]

#     end_time = time.time()
#     execution_time = end_time - start_time

#     print(f"\n📄 أعلى {TOP_N} نتائج:\n")
#     for rank, i in enumerate(top_indices, 1):
#         doc_idx = candidate_indices[i]
#         print(f"{rank}. doc_id: {doc_ids[doc_idx]}, similarity: {sims[i]:.4f}")

#     print(f"\n⏱️ زمن تنفيذ الاستعلام: {execution_time:.4f} ثانية")

# # البحث
# while True:
#     query = input("\n🔍 [TF-IDF] أدخل استعلامك (أو 'exit'): ").strip()
#     if query.lower() == 'exit':
#         break
#     search_tfidf(query)
from flask import Flask, request, jsonify
import joblib
import os
import sqlite3
import numpy as np
import time
import requests
from sklearn.metrics.pairwise import cosine_similarity

# إعداد Flask
app = Flask(__name__)

# إعدادات
INDEX_DIR = 'indexes'
MODELS_DIR = 'models'
TOP_N = 10
DB_PATH = "ir_project.db"
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"
AVAILABLE_SOURCES = ["quora"]

# تحميل قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تحميل الموارد حسب المصدر
resources = {}
for source in AVAILABLE_SOURCES:
    try:
        print(f"🔄 تحميل الموارد لـ {source}...")
        inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{source}.joblib"))
        tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_vectorizer.joblib"))
        doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_doc_ids.joblib"))
        doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_matrix.joblib"))
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        resources[source] = {
            "inverted_index": inverted_index,
            "vectorizer": tfidf_vectorizer,
            "doc_ids": doc_ids,
            "doc_matrix": doc_matrix,
            "doc_id_to_idx": doc_id_to_idx
        }
    except Exception as e:
        print(f"⚠️ خطأ أثناء تحميل بيانات المصدر {source}: {e}")

@app.route('/search-tfidf-inverted', methods=['POST'])
def tfidf_inverted_search():
    data = request.get_json()
    query = data.get("query", "").strip()
    source = data.get("source", "quora").strip().lower()

    if not query:
        return jsonify({"error": "⚠️ يجب إرسال الاستعلام في الحقل 'query'"}), 400

    if source not in resources:
        return jsonify({"error": f"⚠️ المصدر غير مدعوم. المصادر المتاحة: {AVAILABLE_SOURCES}"}), 400

    res = resources[source]

    # المعالجة النصية عبر الخدمة الخارجية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "return_as_string": False
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية"}), 500

        tokens = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    inverted_index = res['inverted_index']
    tfidf_vectorizer = res['vectorizer']
    doc_ids = res['doc_ids']
    doc_matrix = res['doc_matrix']
    doc_id_to_idx = res['doc_id_to_idx']

    # استخراج الوثائق المرشحة
    candidate_doc_ids = set()
    for term in tokens:
        if term in inverted_index:
            candidate_doc_ids.update(inverted_index[term])

    if not candidate_doc_ids:
        return jsonify({"results": [], "message": "❌ لم يتم العثور على وثائق."})

    candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]

    if not candidate_indices:
        return jsonify({"results": [], "message": "❌ لا توجد تمثيلات متاحة لهذه الوثائق."})

    # حساب التشابه
    start_time = time.time()
    query_vec = tfidf_vectorizer.transform([" ".join(tokens)])
    sub_doc_matrix = doc_matrix[candidate_indices]
    sims = cosine_similarity(query_vec, sub_doc_matrix)[0]
    top_indices = np.argsort(sims)[::-1][:TOP_N]
    end_time = time.time()

    results = []
    for rank, i in enumerate(top_indices, 1):
        doc_idx = candidate_indices[i]
        doc_id = doc_ids[doc_idx]
        score = sims[i]

        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
        row = cursor.fetchone()
        content = row[0] if row else "(لم يتم العثور على النص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": content[:300] + ("..." if len(content) > 300 else "")
        })

    return jsonify({
        "query": query,
        "tokens": tokens,
        "source": source,
        "execution_time": round(end_time - start_time, 4),
        "results": results
    })


if __name__ == '__main__':
    app.run(debug=True, port=5013)
