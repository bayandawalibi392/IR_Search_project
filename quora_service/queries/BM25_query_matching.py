import os
import sqlite3
import joblib
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from text_preprocessing_service import TextPreprocessingService

# إعدادات
MODELS_DIR = "models"
SOURCES = ["webis", "quora"]
TOP_N = 5

# ربط قاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

# خدمة المعالجة النصية
preprocessor = TextPreprocessingService()

for source in SOURCES:
    print(f"\n🔍 تنفيذ استعلامات BM25 على المصدر: {source}")

    # تحميل الاستعلامات
    cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (source,))
    queries = cursor.fetchall()

    # تحميل نموذج BM25
    model_path = os.path.join(MODELS_DIR, f"bm25_{source}_model.joblib")
    bm25_data = joblib.load(model_path)
    doc_ids = bm25_data['doc_ids']
    tokenized_docs = bm25_data['tokenized_texts']
    bm25 = BM25Okapi(tokenized_docs, k1=bm25_data['k1'], b=bm25_data['b'])

    results = []

    for query_id, query_text in tqdm(queries, desc=f"Matching queries - {source}"):
        # معالجة الاستعلام
        cleaned_query = preprocessor.preprocess(query_text, return_as_string=True)
        tokenized_query = word_tokenize(cleaned_query)

        # حساب الدرجات
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_N]

        for rank, idx in enumerate(top_indices):
            doc_id = doc_ids[idx]
            score = scores[idx]
            results.append((query_id, doc_id, score))

    # حفظ النتائج
    output_path = os.path.join(MODELS_DIR, f"bm25_{source}_top{TOP_N}_results.tsv")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("query_id\tdoc_id\tscore\n")
        for qid, did, score in results:
            f.write(f"{qid}\t{did}\t{score:.4f}\n")

    print(f"✅ تم حفظ نتائج Top-{TOP_N} لـ '{source}' في: {output_path}")

conn.close()
