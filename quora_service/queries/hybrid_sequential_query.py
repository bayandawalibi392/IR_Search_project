import os
import sqlite3
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from text_preprocessing_service import TextPreprocessingService

# --- إعدادات ---
MODELS_DIR = "models"
SOURCE = "webis"  # أو "quora"
TOP_TFIDF = 100  # عدد النتائج الأولية من TF-IDF
TOP_FINAL = 10   # عدد النتائج النهائية بعد تصفية BERT

# تحميل BERT model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

def embed_text(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).cpu().numpy()

# ربط قاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

# تحميل الاستعلامات
cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (SOURCE,))
all_queries = cursor.fetchall()

print(f"\n🔎 اختر استعلامًا من مجموعة {SOURCE.upper()}:")
for i, (qid, qtext) in enumerate(all_queries[:10]):
    print(f"{i+1}. {qtext}")

index = int(input("\n📌 أدخل رقم الاستعلام (1-10): ")) - 1
query_id, query_text = all_queries[index]
print(f"\n🧠 تم اختيار الاستعلام: {query_text} (ID: {query_id})")

# خدمة المعالجة النصية
preprocessor = TextPreprocessingService()
cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)

# --- تمثيل الاستعلام بـ TF-IDF ---
tfidf_vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_vectorizer.joblib"))
tfidf_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_doc_ids.joblib"))
tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{SOURCE}_matrix.joblib"))
tfidf_query_vec = tfidf_vectorizer.transform([cleaned_query])
tfidf_sims = cosine_similarity(tfidf_query_vec, tfidf_matrix)[0]

# --- اختيار أعلى TOP_TFIDF وثيقة من TF-IDF ---
top_indices = np.argsort(tfidf_sims)[::-1][:TOP_TFIDF]
selected_doc_ids = [tfidf_doc_ids[i] for i in top_indices]

# --- تحميل تمثيلات BERT ---
bert_doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_doc_ids.joblib"))
bert_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{SOURCE}_vectors.joblib"))
bert_id_to_index = {doc_id: i for i, doc_id in enumerate(bert_doc_ids)}

# --- تمثيل الاستعلام باستخدام BERT ---
bert_query_vec = embed_text(cleaned_query)

# --- حساب التشابه بين الاستعلام وأعلى وثائق BERT فقط ---
bert_results = []
for doc_id in selected_doc_ids:
    if doc_id in bert_id_to_index:
        doc_vec = bert_vectors[bert_id_to_index[doc_id]]
        score = cosine_similarity(bert_query_vec, [doc_vec])[0][0]
        bert_results.append((doc_id, score))

# --- ترتيب النتائج النهائية ---
bert_results.sort(key=lambda x: x[1], reverse=True)
top_results = bert_results[:TOP_FINAL]

# --- عرض النتائج ---
print(f"\n📄 أعلى {TOP_FINAL} نتائج (تمثيل هجين تسلسلي):\n")
for rank, (doc_id, score) in enumerate(top_results, 1):
    cursor.execute("SELECT content FROM documents WHERE doc_id = ?", (doc_id,))
    result = cursor.fetchone()
    content = result[0] if result else "(لا يوجد نص)"

    print(f"{rank}. doc_id: {doc_id}")
    print(f"   similarity (BERT): {score:.4f}")
    print(f"   content: {content[:300]}...")
    print("-" * 80)

conn.close()
