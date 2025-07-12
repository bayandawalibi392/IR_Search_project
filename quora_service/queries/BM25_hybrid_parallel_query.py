# import os
# import sqlite3
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.tokenize import word_tokenize
# from rank_bm25 import BM25Okapi
# from text_preprocessing_service import TextPreprocessingService

# # --- إعدادات ---
# MODELS_DIR = "models"
# TOP_N = 10
# ALPHA = 0.5  # وزن TF-IDF مقابل BM25 (يمكنك التعديل)

# # ربط قاعدة البيانات (مرة واحدة)
# conn = sqlite3.connect("ir_project.db")
# cursor = conn.cursor()

# available_sources = ['webis', 'quora']

# def perform_search(SOURCE):
#     cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
#     all_queries = cursor.fetchall()
#     if not all_queries:
#         print(f"لا توجد استعلامات لمصدر {SOURCE}")
#         return

#     print(f"\n🔎 اختر استعلامًا من مجموعة {SOURCE.upper()}:")
#     for i, (qid, qtext) in enumerate(all_queries[:10]):
#         print(f"{i+1}. {qtext}")

#     try:
#         index = int(input("\n📌 أدخل رقم الاستعلام (1-10): ")) - 1
#         if index < 0 or index >= len(all_queries):
#             print("رقم غير صحيح، حاول مجدداً.")
#             return
#     except ValueError:
#         print("الرجاء إدخال رقم صحيح.")
#         return

#     query_id, query_text = all_queries[index]
#     print(f"\n🧠 تم اختيار الاستعلام: {query_text} (ID: {query_id})")

#     # خدمة المعالجة النصية
#     preprocessor = TextPreprocessingService()
#     cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

#     # --- تحميل تمثيلات TF-IDF ---
#     tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
#     tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
#     tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

#     tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
#     sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]

#     # --- تحميل نموذج BM25 ---
#     bm25_data = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
#     bm25_doc_ids = bm25_data['doc_ids']
#     tokenized_docs = bm25_data['tokenized_texts']
#     bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

#     # تمثيل الاستعلام لـ BM25
#     tokenized_query = word_tokenize(cleaned_query)
#     sims_bm25 = bm25.get_scores(tokenized_query)

#     # --- موائمة قوائم الوثائق ---
#     tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
#     bm25_id_to_idx = {doc_id: i for i, doc_id in enumerate(bm25_doc_ids)}

#     common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bm25_doc_ids)))

#     tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
#     bm25_indices = [bm25_id_to_idx[doc_id] for doc_id in common_doc_ids]

#     aligned_sims_tfidf = sims_tfidf[tfidf_indices]
#     aligned_sims_bm25 = np.array([sims_bm25[i] for i in bm25_indices])

#     # --- دمج التشابهات ---
#     final_sims = ALPHA * aligned_sims_tfidf + (1 - ALPHA) * aligned_sims_bm25

#     top_indices = np.argsort(final_sims)[::-1][:TOP_N]

#     # --- عرض النتائج ---
#     print(f"\n📄 أعلى {TOP_N} نتائج (تمثيل هجين متوازي TF-IDF + BM25، ALPHA = {ALPHA}):\n")
#     for rank, idx in enumerate(top_indices, 1):
#         doc_id = common_doc_ids[idx]
#         score = final_sims[idx]

#         cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
#         result = cursor.fetchone()
#         content = result[0] if result else "(لا يوجد نص)"

#         print(f"{rank}. doc_id: {doc_id}")
#         print(f"   hybrid_similarity: {score:.4f}")
#         print(f"   content: {content[:300]}...")
#         print("-" * 80)

# def main_loop():
#     print("اكتب Exit للخروج في أي وقت.\n")
#     while True:
#         print("Available sources:")
#         for i, src in enumerate(available_sources, 1):
#             print(f"{i}. {src}")
#         user_input = input("Select a source by number or type Exit to quit: ").strip()
#         if user_input.lower() == "exit":
#             print("خروج من البرنامج.")
#             break
#         try:
#             selected_idx = int(user_input) - 1
#             if selected_idx < 0 or selected_idx >= len(available_sources):
#                 print("رقم غير صحيح، حاول مجدداً.\n")
#                 continue
#         except ValueError:
#             print("الرجاء إدخال رقم صحيح أو Exit.\n")
#             continue

#         source = available_sources[selected_idx]
#         perform_search(source)
#         print("\n" + "="*50 + "\n")

# if __name__ == "__main__":
#     main_loop()
#     conn.close()

from flask import Flask, request, jsonify
import os
import sqlite3
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
import requests

# إعدادات
MODELS_DIR = "models"
TOP_N = 10
ALPHA = 0.5  # وزن TF-IDF مقابل BM25
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"  # رابط خدمة المعالجة النصية الخارجية

# قائمة المصادر المتاحة
available_sources = ['quora']

# إنشاء تطبيق Flask
app = Flask(__name__)

# فتح اتصال بقاعدة البيانات مع السماح بالتعامل من عدة ثريدات
conn = sqlite3.connect("ir_project.db", check_same_thread=False)
cursor = conn.cursor()

# تحميل نماذج وموارد لكل مصدر عند بدء التشغيل لتسريع الاستجابة
resources = {}
for SOURCE in available_sources:
    try:
        # تحميل TF-IDF
        tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
        tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
        tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

        # تحميل BM25
        bm25 = joblib.load(os.path.join(MODELS_DIR, f"bm25_{SOURCE}_model.joblib"))
        # bm25_doc_ids = bm25_data['doc_ids']
        bm25_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bm25_tokenized_docs_{group}.joblib"))

        tokenized_docs = bm25_data['tokenized_texts']
        bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

        resources[SOURCE] = {
            "tfidf_vectorizer": tfidf_vectorizer,
            "tfidf_doc_ids": tfidf_doc_ids,
            "tfidf_matrix": tfidf_matrix,
            "bm25": bm25,
            "bm25_doc_ids": bm25_doc_ids
        }
        print(f"✅ تم تحميل الموارد للمصدر: {SOURCE}")
    except Exception as e:
        print(f"⚠️ فشل تحميل الموارد للمصدر {SOURCE}: {e}")

@app.route("/search-hybrid", methods=["POST"])
def search_hybrid():
    data = request.get_json()
    query = data.get("query", "").strip()
    source = data.get("source", "quora").strip().lower()

    if not query:
        return jsonify({"error": "⚠️ يجب إرسال حقل 'query' في الطلب"}), 400
    if source not in resources:
        return jsonify({"error": f"⚠️ المصدر غير مدعوم. المصادر المتاحة: {available_sources}"}), 400

    # معالجة النص باستخدام الخدمة الخارجية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "return_as_string": True  # نص المعالجة كنص واحد
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ فشل في خدمة المعالجة النصية الخارجية"}), 500
        cleaned_query = response.json().get("clean_text", "")
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not cleaned_query.strip():
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    # استخرج الموارد الخاصة بالمصدر
    tfidf_vectorizer = resources[source]["tfidf_vectorizer"]
    tfidf_doc_ids = resources[source]["tfidf_doc_ids"]
    tfidf_matrix = resources[source]["tfidf_matrix"]
    bm25 = resources[source]["bm25"]
    bm25_doc_ids = resources[source]["bm25_doc_ids"]

    # تمثيل الاستعلام بـ TF-IDF
    tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
    sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]

    # تمثيل الاستعلام لـ BM25
    tokenized_query = word_tokenize(cleaned_query)
    sims_bm25 = bm25.get_scores(tokenized_query)

    # موائمة الوثائق المشتركة بين النموذجين
    tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
    bm25_id_to_idx = {doc_id: i for i, doc_id in enumerate(bm25_doc_ids)}

    common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bm25_doc_ids)))

    tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
    bm25_indices = [bm25_id_to_idx[doc_id] for doc_id in common_doc_ids]

    aligned_sims_tfidf = sims_tfidf[tfidf_indices]
    aligned_sims_bm25 = np.array([sims_bm25[i] for i in bm25_indices])

    # دمج التشابهات
    final_sims = ALPHA * aligned_sims_tfidf + (1 - ALPHA) * aligned_sims_bm25
    top_indices = np.argsort(final_sims)[::-1][:TOP_N]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = common_doc_ids[idx]
        score = final_sims[idx]

        cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
        result = cursor.fetchone()
        content = result[0] if result else "(لا يوجد نص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "hybrid_similarity": round(float(score), 4),
            "content": content[:500]
        })

    return jsonify({
        "query": query,
        "cleaned_query": cleaned_query,
        "source": source,
        "top_n": TOP_N,
        "results": results
    })


if __name__ == "__main__":
    app.run(port=5018, debug=True)
