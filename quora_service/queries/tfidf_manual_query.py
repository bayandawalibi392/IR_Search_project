# import os
# import sqlite3
# import joblib
# import numpy as np
# import time
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from sklearn.metrics.pairwise import cosine_similarity
# from text_preprocessing_service import TextPreprocessingService

# # --- إعدادات ---
# MODELS_DIR = "models"
# SOURCE = "quora"  # أو "quora"
# TOP_N = 10

# # ربط قاعدة البيانات
# conn = sqlite3.connect("ir_project.db")
# cursor = conn.cursor()

# # تحميل الاستعلامات
# cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
# all_queries = cursor.fetchall()

# print(f"\n🔎 اختر استعلامًا من مجموعة {SOURCE.upper()}:")
# for i, (qid, qtext) in enumerate(all_queries[:10]):
#     print(f"{i+1}. {qtext}")

# index = int(input("\n📌 أدخل رقم الاستعلام (1-10): ")) - 1
# query_id, query_text = all_queries[index]

# print(f"\n🧠 تم اختيار الاستعلام: {query_text} (ID: {query_id})")

# # تحميل ملفات TF-IDF
# vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
# doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
# doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

# # تهيئة المعالجة
# preprocessor = TextPreprocessingService()
# cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

# # بدء توقيت التنفيذ
# start_time = time.time()

# query_vec = vectorizer.transform([cleaned_query])
# sims = cosine_similarity(query_vec, doc_matrix)[0]
# top_indices = np.argsort(sims)[::-1][:TOP_N]

# # نهاية توقيت التنفيذ
# end_time = time.time()
# execution_time = end_time - start_time

# # عرض النتائج
# print(f"\n📄 أعلى {TOP_N} نتائج لاستعلامك:\n")
# for rank, idx in enumerate(top_indices, 1):
#     doc_id = doc_ids[idx]
#     similarity = sims[idx]

#     cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
#     result = cursor.fetchone()
#     content = result[0] if result else "(لا يوجد نص)"

#     print(f"{rank}. doc_id: {doc_id}")
#     print(f"   similarity: {similarity:.4f}")
#     print(f"   content: {content[:300]}...")
#     print("-" * 80)

# print(f"\n⏱️ زمن تنفيذ الاستعلام: {execution_time:.4f} ثانية")
# conn.close()
from flask import Flask, request, jsonify
import os
import sqlite3
import joblib
import numpy as np
import time
import requests
from sklearn.metrics.pairwise import cosine_similarity

# إعداد Flask
app = Flask(__name__)

# إعدادات
MODELS_DIR = "models"
SOURCE = "quora"
TOP_N = 10
DB_PATH = "ir_project.db"
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"

# تحميل ملفات TF-IDF
vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/search-tfidf', methods=['POST'])
def search_tfidf():
    data = request.get_json()
    query_text = data.get("query", "").strip()

    if not query_text:
        return jsonify({"error": "⚠️ يجب إرسال حقل 'query'"}), 400

    # المعالجة النصية عبر API
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query_text,
            "return_as_string": True
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ خطأ في خدمة المعالجة النصية"}), 500

        cleaned_query = response.json().get("clean_text", "")
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    # تنفيذ البحث
    start_time = time.time()
    query_vec = vectorizer.transform([cleaned_query])
    sims = cosine_similarity(query_vec, doc_matrix)[0]
    top_indices = np.argsort(sims)[::-1][:TOP_N]
    end_time = time.time()

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = doc_ids[idx]
        similarity = sims[idx]

        cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        content = row[0] if row else "(لا يوجد نص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(similarity), 4),
            "content": content[:300] + ("..." if len(content) > 300 else "")
        })

    return jsonify({
        "query": query_text,
        "cleaned_query": cleaned_query,
        "results": results,
        "execution_time": round(end_time - start_time, 4)
    })

if __name__ == '__main__':
    app.run(port=5010, debug=True)
