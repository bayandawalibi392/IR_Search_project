# # faiss_search_quora.py

# import faiss
# import joblib
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from TextPreprocessor import TextPreprocessor
# import sqlite3
# import textwrap
# import time

# # إعدادات
# GROUP = 'quora'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# DB_PATH = 'ir_project.db'
# TOP_K = 10
# MODEL_NAME = 'all-MiniLM-L6-v2'

# # تحميل فهرس FAISS المحفوظ
# print("📥 تحميل فهرس FAISS...")
# index = faiss.read_index(f"{INDEX_DIR}/faiss_index_{GROUP}_bert.index")
# print(f"✅ تم تحميل الفهرس بعدد: {index.ntotal} متجه")

# # تحميل معرفات الوثائق
# doc_ids = joblib.load(f"{MODEL_DIR}/bert_{GROUP}_doc_ids.joblib")

# # تحميل النموذج والمعالجة
# bert_model = SentenceTransformer(MODEL_NAME)
# pre = TextPreprocessor()

# # الاتصال بقاعدة البيانات
# conn = sqlite3.connect(DB_PATH)
# cursor = conn.cursor()

# # استعلام المستخدم
# while True:
#     query = input("\n🔍 أدخل استعلامك (أو 'exit' للخروج): ").strip()
#     if query.lower() == 'exit':
#         break

#     # معالجة الاستعلام
#     tokens = pre.preprocess(query)
#     if not tokens:
#         print("⚠️ الاستعلام فارغ بعد المعالجة.")
#         continue

#     query_text = ' '.join(tokens)
#     query_vec = bert_model.encode([query_text]).astype('float32')

#     # البحث في FAISS
#     start = time.perf_counter()
#     D, I = index.search(query_vec, TOP_K)
#     end = time.perf_counter()

#     print(f"\n⏱️ زمن التنفيذ: {end - start:.4f} ثانية")
#     print(f"\n📄 أعلى {TOP_K} نتائج باستخدام FAISS + BERT:")

#     for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
#         doc_id = doc_ids[idx]

#         cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
#         row = cursor.fetchone()
#         content = row[0] if row else "(لم يتم العثور على النص)"

#         print(f"#{rank} | 🔑 ID: {doc_id} | 🧮 Distance: {dist:.4f}")
#         print(textwrap.shorten(content, width=300))
#         print()
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import joblib
import numpy as np
import sqlite3
import requests
import textwrap
import time
from sentence_transformers import SentenceTransformer

# إعدادات
GROUP = 'quora'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
DB_PATH = 'ir_project.db'
TOP_K = 10
MODEL_NAME = 'all-MiniLM-L6-v2'
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"

# إعداد Flask
app = Flask(__name__)
CORS(app)

# تحميل الموارد
print("📥 تحميل فهرس FAISS...")
index = faiss.read_index(f"{INDEX_DIR}/faiss_index_{GROUP}_bert.index")
print(f"✅ تم تحميل الفهرس بعدد: {index.ntotal} متجه")

doc_ids = joblib.load(f"{MODEL_DIR}/bert_{GROUP}_doc_ids.joblib")
bert_model = SentenceTransformer(MODEL_NAME)

# قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/query/bert-faiss', methods=['POST'])
def bert_faiss_search():
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
        if response.status_code != 200:
            return jsonify({"error": "⚠️ خدمة المعالجة فشلت"}), 500

        tokens = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"⚠️ تعذر الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    query_text = ' '.join(tokens)
    query_vec = bert_model.encode([query_text]).astype('float32')

    # البحث باستخدام FAISS
    start_time = time.perf_counter()
    D, I = index.search(query_vec, TOP_K)
    elapsed_time = time.perf_counter() - start_time

    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        doc_id = doc_ids[idx]

        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, GROUP))
        row = cursor.fetchone()
        content = row[0] if row else "(لم يتم العثور على النص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "distance": round(float(dist), 4),
            "content": textwrap.shorten(content, width=300)
        })

    return jsonify({
        "execution_time": round(elapsed_time, 4),
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5009)
