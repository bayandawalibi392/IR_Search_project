import os
import sqlite3
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from text_preprocessing_service import TextPreprocessingService

# إعدادات
MODELS_DIR = "models"
SOURCES = [ "quora"]
TOP_N = 5

# تحميل موديل BERT نفسه المستخدم في التمثيل (all-MiniLM-L6-v2)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def embed_text(text):
    """احصل على embedding للنص باستخدام BERT"""
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # mean pooling
    embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# ربط قاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

# خدمة المعالجة النصية
preprocessor = TextPreprocessingService()

for source in SOURCES:
    print(f"\n🔍 Running BERT query matching for source: {source}")

    # تحميل الاستعلامات
    cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (source,))
    queries = cursor.fetchall()

    # تحميل بيانات BERT embeddings للوثائق
    doc_ids = joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_doc_ids.joblib"))
    doc_vectors = joblib.load(os.path.join(MODELS_DIR, f"bert_{source}_vectors.joblib"))  # numpy array (N_docs x embedding_dim)

    results = []  # لتخزين النتائج (query_id, doc_id, similarity)

    for query_id, query_text in tqdm(queries, desc=f"Matching queries - {source}"):
        # معالجة الاستعلام
        cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)
        # تمثيل الاستعلام باستخدام BERT
        query_vec = embed_text(cleaned_query)  # شكل (1, embedding_dim)

        # حساب التشابه (cosine similarity)
        sims = cosine_similarity(query_vec, doc_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:TOP_N]

        for rank, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            sim_score = sims[idx]
            results.append((query_id, doc_id, sim_score))

    # حفظ النتائج
    output_path = os.path.join(MODELS_DIR, f"bert_{source}_top{TOP_N}_results.tsv")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for qid, did, score in results:
            f.write(f"{qid}\t{did}\t{score:.4f}\n")

    print(f"✅ Saved top-{TOP_N} results for '{source}' to: {output_path}")

conn.close()
