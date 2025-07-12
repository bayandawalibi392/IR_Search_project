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
# INDEX_DIR = "indexes"
# TOP_N = 10
# ALPHA = 0.6  # وزن TF-IDF مقابل BERT (0 = فقط BERT، 1 = فقط TF-IDF)

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
#     # تحميل الفهرس المعكوس
#     inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib"))

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

#     # --- المعالجة ---
#     cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)
#     tokens = preprocessor.preprocess(query_text, return_as_string=False)

#     # --- استخراج الوثائق المرشحة من الفهرس ---
#     candidate_doc_ids = set()
#     for token in tokens:
#         if token in inverted_index:
#             candidate_doc_ids.update(inverted_index[token])

#     if not candidate_doc_ids:
#         print("❌ لا توجد وثائق مرشحة من الفهرس.")
#         return

#     print(f"✅ عدد الوثائق المرشحة من الفهرس: {len(candidate_doc_ids)}")

#     # --- تحميل التمثيلات ---
#     tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
#     tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
#     tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))

#     bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
#     bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

#     tfidf_id_to_idx = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
#     bert_id_to_idx = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

#     # --- تقاطع الوثائق المرشحة بين النموذجين ---
#     common_doc_ids = list(candidate_doc_ids.intersection(tfidf_doc_ids).intersection(bert_doc_ids))
#     if not common_doc_ids:
#         print("❌ لا توجد وثائق مشتركة بين الفهرس وBERT/TF-IDF.")
#         return

#     tfidf_indices = [tfidf_id_to_idx[doc_id] for doc_id in common_doc_ids]
#     bert_indices = [bert_id_to_idx[doc_id] for doc_id in common_doc_ids]

#     # --- التمثيل ---
#     tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
#     sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix[tfidf_indices])[0]

#     bert_query_vec = embed_text(cleaned_query)
#     sims_bert = cosine_similarity(bert_query_vec, bert_vectors[bert_indices])[0]

#     # --- دمج ---
#     final_sims = ALPHA * sims_tfidf + (1 - ALPHA) * sims_bert
#     top_indices = np.argsort(final_sims)[::-1][:TOP_N]

#     print(f"\n📄 أعلى {TOP_N} نتائج (باستخدام الفهرس، ALPHA = {ALPHA}):\n")
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
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import requests

# إعداد Flask
app = Flask(__name__)

# إعدادات
MODELS_DIR = "models"
INDEX_DIR = "indexes"
DB_PATH = "ir_project.db"
TOP_N = 10
ALPHA = 0.6
AVAILABLE_SOURCES = ["quora"]
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"

# تحميل BERT model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

def embed_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# تحميل الموارد لكل مصدر
resources = {}
for source in AVAILABLE_SOURCES:
    try:
        resources[source] = {
            "inverted_index": joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{source}.joblib")),
            "tfidf_vectorizer": joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_vectorizer.joblib")),
            "tfidf_doc_ids": joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_doc_ids.joblib")),
            "tfidf_matrix": joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_matrix.joblib")),
            "bert_doc_ids": joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_doc_ids.joblib")),
            "bert_vectors": joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_vectors.joblib")),
        }
    except Exception as e:
        print(f"⚠️ فشل تحميل بيانات {source}: {e}")

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route("/search-hybrid-inverted", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    source = data.get("source", "quora").strip().lower()

    if not query:
        return jsonify({"error": "⚠️ يجب إدخال الاستعلام"}), 400
    if source not in resources:
        return jsonify({"error": f"⚠️ المصدر غير مدعوم. استخدم أحد: {AVAILABLE_SOURCES}"}), 400

    # المعالجة النصية عبر الخدمة الخارجية
    try:
        response = requests.post(PREPROCESS_API_URL, json={"text": query, "return_as_string": False})
        response.raise_for_status()
        tokens = response.json().get("tokens", [])
        cleaned_query = " ".join(tokens)
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة: {str(e)}"}), 500

    if not tokens:
        return jsonify({"error": "⚠️ الاستعلام فارغ بعد المعالجة"}), 400

    res = resources[source]
    inv_index = res["inverted_index"]
    tfidf_vectorizer = res["tfidf_vectorizer"]
    tfidf_doc_ids = res["tfidf_doc_ids"]
    tfidf_matrix = res["tfidf_matrix"]
    bert_doc_ids = res["bert_doc_ids"]
    bert_vectors = res["bert_vectors"]

    tfidf_idx_map = {doc_id: i for i, doc_id in enumerate(tfidf_doc_ids)}
    bert_idx_map = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

    # استخراج الوثائق المرشحة
    candidate_doc_ids = set()
    for token in tokens:
        if token in inv_index:
            candidate_doc_ids.update(inv_index[token])

    common_doc_ids = list(candidate_doc_ids & set(tfidf_doc_ids) & set(bert_doc_ids))
    if not common_doc_ids:
        return jsonify({"results": [], "message": "❌ لا توجد وثائق مرشحة"}), 200

    tfidf_indices = [tfidf_idx_map[doc_id] for doc_id in common_doc_ids]
    bert_indices = [bert_idx_map[doc_id] for doc_id in common_doc_ids]

    # التمثيل
    tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
    bert_query_vec = embed_text(cleaned_query)

    sims_tfidf = cosine_similarity(tfidf_query_vec, tfidf_matrix[tfidf_indices])[0]
    sims_bert = cosine_similarity(bert_query_vec, bert_vectors[bert_indices])[0]
    sims_final = ALPHA * sims_tfidf + (1 - ALPHA) * sims_bert

    top_indices = np.argsort(sims_final)[::-1][:TOP_N]
    results = []
    for rank, idx in enumerate(top_indices, 1):
        doc_id = common_doc_ids[idx]
        score = sims_final[idx]
        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
        row = cursor.fetchone()
        content = row[0] if row else "(لا يوجد نص)"
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": content[:300] + "..." if len(content) > 300 else content
        })

    return jsonify({
        "query": query,
        "tokens": tokens,
        "source": source,
        "alpha": ALPHA,
        "top_n": TOP_N,
        "results": results
    })

if __name__ == "__main__":
    app.run(port=5015, debug=True)
