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
# SOURCE = "quora"  # أو "quora"
# TOP_N = 10

# # تحميل موديل BERT نفسه المستخدم سابقًا
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()

# def embed_text(text):
#     """تمثيل النص باستخدام BERT مع mean pooling"""
#     encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
#     with torch.no_grad():
#         model_output = model(**encoded_input)
#     embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
#     return embeddings

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

# # تحميل تمثيلات BERT للوثائق
# doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
# doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))  # numpy array

# # تهيئة خدمة المعالجة النصية
# preprocessor = TextPreprocessingService()
# cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

# # تمثيل الاستعلام بالـ BERT
# query_vec = embed_text(cleaned_query)  # شكل (1, embedding_dim)

# # حساب التشابه
# sims = cosine_similarity(query_vec, doc_vectors)[0]
# top_indices = np.argsort(sims)[::-1][:TOP_N]

# print(f"\n📄 أعلى {TOP_N} نتائج لاستعلامك:\n")

# for rank, idx in enumerate(top_indices, 1):
#     doc_id = doc_ids[idx]
#     similarity = sims[idx]

#     # جلب محتوى الوثيقة من قاعدة البيانات
#     cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
#     result = cursor.fetchone()
#     content = result[0] if result else "(لا يوجد نص)"

#     print(f"{rank}. doc_id: {doc_id}")
#     print(f"   similarity: {similarity:.4f}")
#     print(f"   content: {content[:300]}...")  # عرض أول 300 حرف فقط
#     print("-" * 80)

# conn.close()
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
SOURCE = "quora"
TOP_N = 10
DB_PATH = "ir_project.db"
PREPROCESS_API_URL = "http://127.0.0.1:5060/preprocess"  # خدمة المعالجة النصية

# تحميل نموذج BERT
print("🔄 تحميل نموذج BERT...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def embed_text(text):
    """تمثيل النص باستخدام BERT مع mean pooling"""
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# تحميل تمثيلات الوثائق
print("📦 تحميل بيانات الوثائق...")
doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

@app.route('/search-bert', methods=['POST'])
def search_bert():
    data = request.get_json()
    query_text = data.get("query", "").strip()

    if not query_text:
        return jsonify({"error": "⚠️ يجب إرسال استعلام ضمن الحقل 'query'"}), 400

    # إرسال الاستعلام إلى خدمة المعالجة النصية
    try:
        response = requests.post(PREPROCESS_API_URL, json={
            "text": query_text,
            "return_as_string": True
        })
        if response.status_code != 200:
            return jsonify({"error": "⚠️ خطأ في خدمة المعالجة النصية"}), 500

        cleaned_query = response.json().get("clean_text", "")
    except Exception as e:
        return jsonify({"error": f"⚠️ فشل الاتصال بخدمة المعالجة النصية: {str(e)}"}), 500

    # تمثيل الاستعلام باستخدام BERT
    start_time = time.time()
    query_vec = embed_text(cleaned_query)
    sims = cosine_similarity(query_vec, doc_vectors)[0]
    top_indices = np.argsort(sims)[::-1][:TOP_N]
    end_time = time.time()

    # تجهيز النتائج
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
    app.run(debug=True, port=5011)
