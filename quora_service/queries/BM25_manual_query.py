# import os
# import sqlite3
# import joblib
# from nltk.tokenize import word_tokenize
# from rank_bm25 import BM25Okapi
# from text_preprocessing_service import TextPreprocessingService
# from tqdm import tqdm

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

# # تحميل نموذج BM25
# model_path = os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib")
# bm25_data = joblib.load(model_path)
# doc_ids = bm25_data['doc_ids']
# tokenized_docs = bm25_data['tokenized_texts']
# bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

# # تهيئة خدمة المعالجة النصية
# preprocessor = TextPreprocessingService()
# cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)
# tokenized_query = word_tokenize(cleaned_query)

# # حساب درجات BM25 للاستعلام
# scores = bm25.get_scores(tokenized_query)
# top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_N]

# print(f"\n📄 أعلى {TOP_N} نتائج لاستعلامك باستخدام BM25:\n")

# for rank, idx in enumerate(top_indices, 1):
#     doc_id = doc_ids[idx]
#     score = scores[idx]

#     # جلب محتوى الوثيقة من قاعدة البيانات
#     cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
#     result = cursor.fetchone()
#     content = result[0] if result else "(لا يوجد نص)"

#     print(f"{rank}. doc_id: {doc_id}")
#     print(f"   BM25 score: {score:.4f}")
#     print(f"   content: {content[:300]}...")  # عرض أول 300 حرف فقط
#     print("-" * 80)

# conn.close()

from flask import Flask, request, jsonify
import os
import sqlite3
import joblib
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import requests

# إعدادات
MODELS_DIR = "models"
SOURCE = "quora"  # ممكن تعدل المصدر حسب حاجتك
TOP_N = 10
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"  # رابط خدمة المعالجة النصية الخارجية

# إنشاء تطبيق Flask
app = Flask(__name__)

# فتح اتصال بقاعدة البيانات
conn = sqlite3.connect("ir_project.db", check_same_thread=False)
cursor = conn.cursor()

# تحميل نموذج BM25 مرة واحدة عند تشغيل السيرفر
model_path = os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib")
bm25_data = joblib.load(model_path)
doc_ids = bm25_data['doc_ids']
tokenized_docs = bm25_data['tokenized_texts']
bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

@app.route("/search-bm25", methods=["POST"])
def search_bm25():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "⚠️ يجب إرسال حقل 'query' في الطلب"}), 400

    # إرسال النص إلى خدمة المعالجة النصية الخارجية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "return_as_string": False  # نريد التوكنز كقائمة
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية الخارجية"}), 500
        tokens = response.json().get("tokens", [])
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    # حساب درجات BM25
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_N]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = doc_ids[idx]
        score = scores[idx]

        # جلب محتوى الوثيقة من قاعدة البيانات
        cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
        result = cursor.fetchone()
        content = result[0] if result else "(لا يوجد نص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": content[:500]  # أول 500 حرف
        })

    return jsonify({
        "query": query,
        "tokens": tokens,
        "top_n": TOP_N,
        "results": results
    })


if __name__ == "__main__":
    app.run(port=5017, debug=True)
