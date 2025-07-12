import os
import sqlite3
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing_service import TextPreprocessingService
from tqdm import tqdm

# --- إعدادات ---
MODELS_DIR = "models"
SOURCES = ["quora"]
TOP_N = 5  # عدد النتائج لكل استعلام

# ربط قاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

# تهيئة خدمة المعالجة
preprocessor = TextPreprocessingService()

for source in SOURCES:
    print(f"\n🔍 Running query matching for source: {source}")

    # تحميل الاستعلامات الأصلية
    cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (source,))
    queries = cursor.fetchall()

    # تحميل ملفات TF-IDF الخاصة بالوثائق
    vectorizer = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_vectorizer.joblib"))
    doc_ids = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_doc_ids.joblib"))
    doc_matrix = joblib.load(os.path.join(MODELS_DIR, f"tfidf_{source}_matrix.joblib"))

    results = []  # (query_id, doc_id, similarity)

    for query_id, query_text in tqdm(queries, desc=f"Matching queries - {source}"):
        cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)
        query_vec = vectorizer.transform([cleaned_query])

        sims = cosine_similarity(query_vec, doc_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:TOP_N]

        for rank, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            sim_score = sims[idx]
            results.append((query_id, doc_id, sim_score))

    # حفظ النتائج في ملف أو قاعدة بيانات حسب الحاجة
    output_path = os.path.join(MODELS_DIR, f"tfidf_{source}_top{TOP_N}_results.tsv")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for qid, did, score in results:
            f.write(f"{qid}\t{did}\t{score:.4f}\n")

    print(f"✅ Saved top-{TOP_N} results for '{source}' to: {output_path}")

conn.close()
