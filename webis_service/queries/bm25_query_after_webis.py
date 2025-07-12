# import sqlite3
# import joblib
# import numpy as np
# import time
# from TextPreprocessor import TextPreprocessor
# import textwrap

# # إعدادات
# DB_PATH = 'ir_project.db'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# GROUP = 'webis'
# TOP_K = 10

# # تحميل الموارد
# print("📦 تحميل الفهرس المعكوس ونموذج BM25...")
# inverted_index = joblib.load(f"{INDEX_DIR}/inverted_index1_{GROUP}.joblib")
# bm25 = joblib.load(f"{MODEL_DIR}/bm25_model_{GROUP}.joblib")
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bm25_{GROUP}.joblib")
# tokenized_documents = joblib.load(f"{MODEL_DIR}/bm25_tokenized_docs_{GROUP}.joblib")

# # الاتصال بقاعدة البيانات
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # تحميل الاستعلامات
# cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (GROUP,))
# queries = cursor.fetchall()

# if not queries:
#     print("⚠️ لا توجد استعلامات في قاعدة البيانات.")
#     exit()

# # عرض قائمة استعلامات للاختيار منها
# print("\n📋 اختر استعلامًا من القائمة:")
# for i, (qid, qtext) in enumerate(queries[:10]):
#     print(f"{i+1}. {qtext[:80]}...")

# choice = input("\n🔢 أدخل رقم الاستعلام الذي تريد تنفيذه (أو 'exit' للخروج): ")
# if choice.lower() == 'exit':
#     exit()
# try:
#     index = int(choice) - 1
#     if index < 0 or index >= len(queries):
#         raise IndexError
#     query = queries[index][1]
# except (ValueError, IndexError):
#     print("⚠️ اختيار غير صالح.")
#     exit()

# # المعالجة
# pre = TextPreprocessor()

# # تنفيذ الاستعلام
# start_time = time.perf_counter()

# query_tokens = pre.tokenize(pre.clean_text(query))
# if not query_tokens:
#     print("⚠️ الاستعلام فارغ بعد المعالجة.")
#     exit()

# # استخراج الوثائق المطابقة فقط من الفهرس المعكوس
# candidate_indices = set()
# for token in query_tokens:
#     if token in inverted_index:
#         candidate_indices.update(inverted_index[token])

# if not candidate_indices:
#     print("❌ لا توجد وثائق مطابقة لهذا الاستعلام في الفهرس.")
#     exit()

# # حساب درجات BM25 فقط على الوثائق المطابقة
# scores = bm25.get_scores(query_tokens)
# candidate_indices = sorted(candidate_indices)

# # تصفية الدرجات لتشمل فقط الوثائق المطابقة
# candidate_scores = [(idx, scores[idx]) for idx in candidate_indices]
# top_indices_with_scores = sorted(candidate_scores, key=lambda x: x[1], reverse=True)[:TOP_K]

# end_time = time.perf_counter()
# elapsed_time = end_time - start_time

# print(f"\n🕒 زمن التنفيذ: {elapsed_time:.4f} ثانية")
# print(f"\n📄 أعلى {TOP_K} نتائج باستخدام BM25 + الفهرس المعكوس:")

# for rank, (idx, score) in enumerate(top_indices_with_scores, 1):
#     doc_id = doc_ids[idx]

#     cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
#     row = cursor.fetchone()
#     content = row[0] if row else "(لم يتم العثور على النص)"

#     print(f"#{rank} | 🔑 ID: {doc_id} | 🔢 Score: {score:.4f}")
#     print(textwrap.shorten(content, width=300))
#     print()
# bm25_inv_api.py


from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3, joblib, numpy as np, time, textwrap, requests

app = Flask(__name__)
CORS(app)

# إعدادات
DB_PATH = 'ir_project.db'
MODEL_DIR = 'models'
GROUP = 'webis'
TOP_K = 10
PREPROCESS_API_URL = "http://localhost:5050/preprocess"

# تحميل الموارد
bm25 = joblib.load(f"{MODEL_DIR}/bm25_model_{GROUP}.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bm25_{GROUP}.joblib")  # قائمة doc_ids

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/query/bm25-all', methods=['POST'])
def search_bm25_all():
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
        return jsonify({"error": f"⚠️ فشل في خدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    start = time.perf_counter()
    
    scores = bm25.get_scores(tokens)
    scores = scores.astype(float)  # لتفادي float32 في JSON

    top_indices = np.argsort(scores)[-TOP_K:][::-1]
    top_doc_ids = [doc_ids[i] for i in top_indices]

    placeholders = ','.join(['?'] * len(top_doc_ids))
    cursor.execute(f"""
        SELECT doc_id, content FROM documents 
        WHERE doc_id IN ({placeholders}) AND source = ?
    """, (*top_doc_ids, GROUP))
    rows = dict(cursor.fetchall())

    elapsed = time.perf_counter() - start

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = doc_ids[idx]
        score = round(float(scores[idx]), 4)
        content = textwrap.shorten(rows.get(doc_id, "(لم يتم العثور على النص)"), width=300)

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": score,
            "content": content
        })

    return jsonify({
        "execution_time": round(float(elapsed), 4),
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5008)
