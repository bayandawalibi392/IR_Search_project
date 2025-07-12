# import joblib
# import os
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from sklearn.metrics.pairwise import cosine_similarity
# from text_preprocessing_service import TextPreprocessingService

# # إعدادات
# INDEX_DIR = 'indexes'
# MODELS_DIR = 'models'
# TOP_N = 10
# SOURCE = "quora"  # أو "quora"

# # تحميل الفهرس
# index_path = os.path.join(INDEX_DIR, f"inverted_index_{SOURCE}.joblib")
# inverted_index = joblib.load(index_path)
# print(f"📂 تم تحميل الفهرس: {index_path}")

# # تحميل تمثيلات BERT
# doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
# doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

# doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

# # تحميل BERT Model
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device).eval()

# def embed(text):
#     tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#     with torch.no_grad():
#         output = model(**tokens)
#     return output.last_hidden_state.mean(dim=1).cpu().numpy()

# # خدمة المعالجة
# preprocessor = TextPreprocessingService()

# def search_bert(query):
#     tokens = preprocessor.preprocess(query, return_as_string=False)
#     print(f"✅ الاستعلام المعالج (tokens): {tokens}")

#     # الوثائق المرشحة
#     candidate_doc_ids = set()
#     for term in tokens:
#         if term in inverted_index:
#             candidate_doc_ids.update(inverted_index[term])

#     if not candidate_doc_ids:
#         print("❌ لا توجد وثائق مرشحة.")
#         return

#     candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]
#     if not candidate_indices:
#         print("❌ لا توجد تمثيلات لهذه الوثائق.")
#         return

#     query_vec = embed(" ".join(tokens))
#     candidate_vectors = doc_vectors[candidate_indices]
#     sims = cosine_similarity(query_vec, candidate_vectors)[0]

#     top_indices = np.argsort(sims)[::-1][:TOP_N]
#     print(f"\n📄 أفضل {TOP_N} نتائج:\n")
#     for rank, i in enumerate(top_indices, 1):
#         doc_idx = candidate_indices[i]
#         print(f"{rank}. doc_id: {doc_ids[doc_idx]}, similarity: {sims[i]:.4f}")

# # التفاعل
# while True:
#     query = input("\n🔍 [BERT] أدخل استعلامك (أو 'exit'): ").strip()
#     if query.lower() == 'exit':
#         break
#     search_bert(query)
from flask import Flask, request, jsonify
import joblib
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import sqlite3
import textwrap

# إعداد Flask
app = Flask(__name__)

# إعدادات
INDEX_DIR = 'indexes'
MODELS_DIR = 'models'
TOP_N = 10
AVAILABLE_SOURCES = ["quora"]
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"
DB_PATH = "ir_project.db"

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# تحميل BERT Model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

def embed(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).cpu().numpy()

# تحميل الموارد الخاصة بكل مجموعة
resources = {}
for source in AVAILABLE_SOURCES:
    try:
        print(f"🔄 تحميل بيانات المصدر: {source}")
        inverted_index = joblib.load(os.path.join(INDEX_DIR, f"inverted_index_{source}.joblib"))
        doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_doc_ids.joblib"))
        doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_vectors.joblib"))
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        resources[source] = {
            "inverted_index": inverted_index,
            "doc_ids": doc_ids,
            "doc_vectors": doc_vectors,
            "doc_id_to_idx": doc_id_to_idx
        }
    except Exception as e:
        print(f"⚠️ خطأ أثناء تحميل بيانات {source}: {e}")

@app.route('/search-bert-inverted', methods=['POST'])
def search_bert_inverted():
    data = request.get_json()
    query = data.get("query", "").strip()
    source = data.get("source", "quora").strip().lower()

    if not query:
        return jsonify({"error": "⚠️ يجب إرسال الاستعلام في الحقل 'query'"}), 400
    if source not in resources:
        return jsonify({"error": f"⚠️ المصدر غير مدعوم. المصادر المتاحة: {AVAILABLE_SOURCES}"}), 400

    res = resources[source]
    start_time = time.time()

    # إرسال إلى خدمة المعالجة الخارجية
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

    inverted_index = res["inverted_index"]
    doc_ids = res["doc_ids"]
    doc_vectors = res["doc_vectors"]
    doc_id_to_idx = res["doc_id_to_idx"]

    # استرجاع الوثائق من الفهرس المعكوس
    candidate_doc_ids = set()
    for term in tokens:
        if term in inverted_index:
            candidate_doc_ids.update(inverted_index[term])

    if not candidate_doc_ids:
        return jsonify({"results": [], "message": "❌ لا توجد وثائق مرشحة."})

    candidate_indices = [doc_id_to_idx[doc_id] for doc_id in candidate_doc_ids if doc_id in doc_id_to_idx]
    if not candidate_indices:
        return jsonify({"results": [], "message": "❌ لا توجد تمثيلات لهذه الوثائق."})

    # تمثيل الاستعلام
    query_vec = embed(" ".join(tokens))
    candidate_vectors = doc_vectors[candidate_indices]
    sims = cosine_similarity(query_vec, candidate_vectors)[0]

    top_indices = np.argsort(sims)[::-1][:TOP_N]
    results = []
    for rank, i in enumerate(top_indices, 1):
        doc_idx = candidate_indices[i]
        doc_id = doc_ids[doc_idx]
        score = sims[i]

        # ✅ جلب المحتوى من قاعدة البيانات
        cursor.execute("SELECT content FROM documents WHERE doc_id = ? AND source = ?", (doc_id, source))
        row = cursor.fetchone()
        content = row[0] if row else "(لا يوجد محتوى)"
        content_short = textwrap.shorten(content, width=300, placeholder="...")

        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "score": round(float(score), 4),
            "content": content_short
        })

    elapsed_time = round(time.time() - start_time, 4)

    return jsonify({
        "query": query,
        "tokens": tokens,
        "source": source,
        "execution_time": elapsed_time,
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5014)
