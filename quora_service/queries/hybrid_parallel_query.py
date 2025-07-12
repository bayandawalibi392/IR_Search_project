# import os
# import sqlite3
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModel
# import torch
# from text_preprocessing_service import TextPreprocessingService

# # --- إعدادات ---
# MODELS_DIR = "models"
# TOP_N = 10
# ALPHA = 1  # وزن BERT مقابل TF-IDF

# # تحميل BERT model & tokenizer
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()

# def embed_text(text):
#     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     return model_output.last_hidden_state.mean(dim=1).cpu().numpy()

# # خدمة المعالجة النصية
# preprocessor = TextPreprocessingService()

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

#     cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

#     # --- تحميل تمثيلات TF-IDF ---
#     tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
#     tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
#     tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

#     tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
#     sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]

#     # --- تحميل تمثيلات BERT ---
#     bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
#     bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

#     bert_query_vec = embed_text(cleaned_query)
#     sims_bert = cosine_similarity(bert_query_vec, bert_vectors)[0]

#     # --- موائمة قوائم الوثائق ---
#     tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
#     bert_id_to_idx = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

#     common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bert_doc_ids)))

#     tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
#     bert_indices = [bert_id_to_idx[doc_id] for doc_id in common_doc_ids]

#     aligned_sims_tfidf = sims_tfidf[tfidf_indices]
#     aligned_sims_bert = sims_bert[bert_indices]

#     # --- دمج التشابهات ---
#     final_sims = ALPHA * aligned_sims_tfidf + (1 - ALPHA) * aligned_sims_bert

#     top_indices = np.argsort(final_sims)[::-1][:TOP_N]

#     # --- عرض النتائج ---
#     print(f"\n📄 أعلى {TOP_N} نتائج (تمثيل هجين متوازي، ALPHA = {ALPHA}):\n")
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
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import time

# إعداد Flask
app = Flask(__name__)

# إعدادات
MODELS_DIR = "models"
TOP_N = 10
DB_PATH = "ir_project.db"
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"
AVAILABLE_SOURCES = ['webis', 'quora']

# تحميل BERT model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# تحميل قاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

def embed_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy()

@app.route('/search-hybrid-parallel', methods=['POST'])
def hybrid_parallel_search():
    data = request.get_json()

    query = data.get("query", "").strip()
    source = data.get("source", "quora").strip().lower()
    alpha = float(data.get("alpha", 0.5))  # وزن TF-IDF

    if not query:
        return jsonify({"error": "⚠️ يجب إرسال استعلام في الحقل 'query'"}), 400

    if source not in AVAILABLE_SOURCES:
        return jsonify({"error": f"⚠️ المصدر غير مدعوم. المصادر المتاحة: {AVAILABLE_SOURCES}"}), 400

    # المعالجة النصية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query,
            "return_as_string": True
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ فشل المعالجة النصية"}), 500

        cleaned_query = response.json().get("clean_text", "")
    except Exception as e:
        return jsonify({"error": f"⚠️ مشكلة في الاتصال بخدمة المعالجة: {str(e)}"}), 500

    # تحميل تمثيلات TF-IDF
    try:
        tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_vectorizer.joblib"))
        tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_doc_ids.joblib"))
        tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_matrix.joblib"))

        tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
        sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]
    except Exception as e:
        return jsonify({"error": f"⚠️ خطأ في تحميل ملفات TF-IDF: {str(e)}"}), 500

    # تحميل تمثيلات BERT
    try:
        bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_doc_ids.joblib"))
        bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_vectors.joblib"))

        bert_query_vec = embed_text(cleaned_query)
        sims_bert = cosine_similarity(bert_query_vec, bert_vectors)[0]
    except Exception as e:
        return jsonify({"error": f"⚠️ خطأ في تحميل ملفات BERT: {str(e)}"}), 500

    # موائمة الوثائق المشتركة
    tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
    bert_id_to_idx = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}
    common_doc_ids = list(set(tfidf_doc_ids).intersection(set(bert_doc_ids)))

    tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
    bert_indices = [bert_id_to_idx[doc_id] for doc_id in common_doc_ids]

    aligned_sims_tfidf = sims_tfidf[tfidf_indices]
    aligned_sims_bert = sims_bert[bert_indices]

    # دمج النتائج
    final_sims = alpha * aligned_sims_tfidf + (1 - alpha) * aligned_sims_bert
    top_indices = np.argsort(final_sims)[::-1][:TOP_N]

    # عرض النتائج
    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = common_doc_ids[idx]
        score = final_sims[idx]

        cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        content = row[0] if row else "(لا يوجد نص)"

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": content[:300] + ("..." if len(content) > 300 else "")
        })

    return jsonify({
        "query": query,
        "cleaned_query": cleaned_query,
        "source": source,
        "alpha": alpha,
        "results": results
    })


if __name__ == '__main__':
    app.run(debug=True, port=5012)
