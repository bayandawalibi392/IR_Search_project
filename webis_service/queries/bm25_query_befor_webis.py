# import sqlite3
# import joblib
# import numpy as np
# import time
# from TextPreprocessor import TextPreprocessor
# import textwrap

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد
# print("📦 تحميل نموذج BM25 والبيانات...")
# bm25 = joblib.load(f"{MODEL_DIR}/bm25_model_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bm25_{GROUP}.joblib")

# # الاتصال بقاعدة البيانات
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # المعالجة
# pre = TextPreprocessor()

# # استعلام المستخدم
# while True:
#     query = input("\n🔍 أدخل استعلامك (أو 'exit' للخروج): ")
#     if query.lower() == 'exit':
#         break

#     start_time = time.perf_counter()  # ⏱️ بدء تتبع الزمن

#     # معالجة الاستعلام
#     tokens = pre.tokenize(pre.clean_text(query))
#     if not tokens:
#         print("⚠️ الاستعلام فارغ بعد المعالجة.")
#         continue

#     # حساب تشابه BM25
#     scores = bm25.get_scores(tokens)
#     top_indices = np.argsort(scores)[-TOP_K:][::-1]

#     end_time = time.perf_counter()  # ⏱️ نهاية تتبع الزمن
#     elapsed_time = end_time - start_time

#     print(f"\n🕒 زمن التنفيذ: {elapsed_time:.4f} ثانية")
#     print(f"\n📄 أعلى {TOP_K} نتائج باستخدام BM25:")

#     for rank, idx in enumerate(top_indices, 1):
#         doc_id = doc_ids[idx]
#         score = scores[idx]

#         cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
#         row = cursor.fetchone()
#         content = row[0] if row else "(لم يتم العثور على النص)"

#         print(f"#{rank} | 🔑 ID: {doc_id} | 🔢 Score: {score:.4f}")
#         print(textwrap.shorten(content, width=300))
#         print()


# bm25_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, joblib, numpy as np, time, textwrap, requests

app = Flask(__name__)
CORS(app)

DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
GROUP = 'webis'
TOP_K = 10
PREPROCESS_API_URL = "http://localhost:5050/preprocess"

bm25 = joblib.load(f"{MODEL_DIR}/bm25_model_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bm25_{GROUP}.joblib")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/query/bm25', methods=['POST'])
def search_bm25():
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

    start = time.perf_counter()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    elapsed = time.perf_counter() - start

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
            "score": round(float(score), 4),
            "content": textwrap.shorten(content, width=300)
        })

    return jsonify({"execution_time": round(float(elapsed), 4), "results": results})

if __name__ == '__main__':
    app.run(debug=True, port=5007)

