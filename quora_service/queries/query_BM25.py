# import joblib
# import os
# from text_preprocessing_service import TextPreprocessingService
# from rank_bm25 import BM25Okapi
# from nltk.tokenize import word_tokenize
# from tqdm import tqdm

# # إعدادات
# INDEX_DIR = 'indexes'
# MODELS_DIR = 'models'
# TOP_N = 10

# # اختيار المصدر
# SOURCE = "quora"  # أو "quora"

# # تحميل الفهرس المعكوس
# index_path = os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib")
# print(f"📂 تحميل الفهرس المعكوس من: {index_path}")
# inverted_index = joblib.load(index_path)

# # تحميل بيانات BM25
# bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
# doc_ids = bm25_data['doc_ids']
# tokenized_corpus = bm25_data['tokenized_texts']
# k1 = bm25_data['k1']
# b = bm25_data['b']

# # إعادة بناء نموذج BM25 (للسهولة في حساب الدرجات)
# bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

# # خدمة المعالجة النصية
# preprocessor = TextPreprocessingService()

# def search(query):
#     # 1. معالجة الاستعلام للحصول على tokens
#     processed_query_tokens = preprocessor.preprocess(query, return_as_string=False)

#     print(f"✅ الاستعلام المعالج (tokens): {processed_query_tokens}")

#     # 2. استخدم الفهرس المعكوس لتجميع الوثائق المحتملة
#     candidate_doc_ids = set()
#     for term in processed_query_tokens:
#         if term in inverted_index:
#             candidate_doc_ids.update(inverted_index[term])

#     if not candidate_doc_ids:
#         print("❌ لم يتم العثور على أي وثائق تحتوي كلمات الاستعلام.")
#         return

#     print(f"✅ عدد الوثائق المرشحة من الفهرس: {len(candidate_doc_ids)}")

#     # 3. بناء قائمة مؤشرات هذه الوثائق في الـ bm25_data (لكي نأخذ فقط هذه الوثائق للترتيب)
#     doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
#     candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]

#     if not candidate_indices:
#         print("❌ لا توجد تمثيلات لهذه الوثائق المرشحة.")
#         return

#     # 4. حساب درجة BM25 للاستعلام مع الوثائق المرشحة فقط
#     scores = bm25.get_scores(processed_query_tokens)
#     candidate_scores = [(idx, scores[idx]) for idx in candidate_indices]

#     # 5. ترتيب النتائج حسب الدرجة
#     candidate_scores.sort(key=lambda x: x[1], reverse=True)
#     top_results = candidate_scores[:TOP_N]

#     print(f"\n📄 أفضل {TOP_N} نتائج مرتبة:\n")
#     for rank, (idx, score) in enumerate(top_results, 1):
#         print(f"{rank}. doc_id: {doc_ids[idx]}, score: {score:.4f}")

# # تجربة البحث
# while True:
#     query = input("\n🔍 أدخل استعلامك (أو 'exit' للخروج): ").strip()
#     if query.lower() == 'exit':
#         break
#     search(query)
from flask import Flask, request, jsonify
import joblib
import os
from rank_bm25 import BM25Okapi
import requests
import sqlite3

# إعدادات
INDEX_DIR = 'indexes'
MODELS_DIR = 'models'
TOP_N = 10
AVAILABLE_SOURCES = ["quora"]
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"

# إعداد Flask
app = Flask(__name__)

DB_PATH = "ir_project.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
# تحميل الموارد لكل مصدر
resources = {}
for source in AVAILABLE_SOURCES:
    try:
        print(f"🔄 تحميل بيانات BM25 للمصدر: {source}")
        inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{source}.joblib"))
        bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{source}_model.joblib"))

        doc_ids = bm25_data["doc_ids"]
        tokenized_corpus = bm25_data["tokenized_texts"]
        k1 = bm25_data["k1"]
        b = bm25_data["b"]

        # بناء نموذج BM25
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

        resources[source] = {
            "inverted_index": inverted_index,
            "bm25": bm25,
            "doc_ids": doc_ids,
            "doc_id_to_idx": doc_id_to_idx
        }
    except Exception as e:
        print(f"⚠️ فشل تحميل بيانات {source}: {e}")

@app.route("/search-bm25-inverted", methods=["POST"])
def search_bm25():
    data = request.get_json()
    query = data.get("query", "").strip()
    source = data.get("source", "quora").strip().lower()

    if not query:
        return jsonify({"error": "⚠️ يجب إرسال الحقل 'query'"}), 400
    if source not in resources:
        return jsonify({"error": f"⚠️ المصدر غير مدعوم. المصادر المتاحة: {AVAILABLE_SOURCES}"}), 400

    res = resources[source]

    # إرسال إلى خدمة المعالجة النصية
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

    # استخراج الوثائق المرشحة من الفهرس
    inverted_index = res["inverted_index"]
    candidate_doc_ids = set()
    for token in tokens:
        if token in inverted_index:
            candidate_doc_ids.update(inverted_index[token])

    if not candidate_doc_ids:
        return jsonify({"results": [], "message": "❌ لا توجد وثائق مرشحة."})

    doc_id_to_idx = res["doc_id_to_idx"]
    candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]
    if not candidate_indices:
        return jsonify({"results": [], "message": "❌ لا توجد تمثيلات لهذه الوثائق."})

    
    # حساب درجات BM25
    bm25 = res["bm25"]
    scores = bm25.get_scores(tokens)
    candidate_scores = [(idx, scores[idx]) for idx in candidate_indices]
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = candidate_scores[:TOP_N]

    results = []
    for rank, (idx, score) in enumerate(top_results, 1):
      doc_id = res["doc_ids"][idx]

    # جلب المحتوى من قاعدة البيانات
      cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
      row = cursor.fetchone()
      content = row[0] if row else "(لا يوجد نص)"

      results.append({
        "rank": rank,
        "doc_id": doc_id,
        "score": round(float(score), 4),
        "content": content[:500]  # على سبيل المثال أول 500 حرف فقط
      })


    return jsonify({
        "query": query,
        "tokens": tokens,
        "source": source,
        "top_n": TOP_N,
        "results": results
    })

if __name__ == "__main__":
    app.run(port=5016, debug=True)
