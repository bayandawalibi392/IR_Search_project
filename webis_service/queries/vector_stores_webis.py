# faiss_search_webis.py

# import faiss
# import joblib
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from TextPreprocessor import TextPreprocessor
# import sqlite3
# import textwrap
# import time

# # إعدادات
# GROUP = 'webis'
# MODEL_DIR = 'models'
# INDEX_DIR = 'indexes'
# DB_PATH = 'ir_project.db'
# TOP_K = 10
# MODEL_NAME = 'all-MiniLM-L6-v2'

# # تحميل فهرس FAISS المحفوظ مسبقًا
# print("📥 تحميل فهرس FAISS...")
# index = faiss.read_index(f"{INDEX_DIR}/faiss_index_{GROUP}_bert.index")
# print(f"✅ تم تحميل الفهرس بعدد: {index.ntotal} متجه")

# # تحميل معرفات الوثائق
# doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")

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
import sqlite3, joblib, numpy as np, time, textwrap, requests
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)
CORS(app)

# الإعدادات
GROUP = 'webis'
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_DIR = 'models'
INDEX_DIR = 'indexes'
DB_PATH = 'ir_project.db'
TOP_K = 10
PREPROCESS_API_URL = "http://localhost:5050/preprocess"

# تحميل النموذج و الفهرس
print("📥 تحميل نموذج BERT والفهرس...")
bert_model = SentenceTransformer(MODEL_NAME)
doc_ids = joblib.load(f"{MODEL_DIR}/doc_ids_bert_{GROUP}.joblib")
faiss_index = faiss.read_index(f"{INDEX_DIR}/faiss_index_{GROUP}_bert.index")

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/search-faiss', methods=['POST'])
def search_faiss():
    data = request.get_json()
    query_text = data.get('query', '').strip()
    if not query_text:
        return jsonify({"error": "⚠️ الاستعلام فارغ"}), 400

    # معالجة الاستعلام عبر خدمة خارجية
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

    # تحويل الاستعلام إلى تمثيل BERT
    query_cleaned = ' '.join(tokens)
    query_vec = bert_model.encode([query_cleaned]).astype('float32')

    # البحث في FAISS
    start_time = time.perf_counter()
    D, I = faiss_index.search(query_vec, TOP_K)
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
            "score": round(float(1 - dist), 4),  # تحويل المسافة إلى تشابه
            "content": textwrap.shorten(content, width=300)
        })

    return jsonify({
        "query": query_text,
        "execution_time": round(float(elapsed_time), 4),
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5020)
