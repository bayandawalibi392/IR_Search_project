import os
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from tqdm import tqdm
from text_preprocessing_service import TextPreprocessingService

# مسار التخزين
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# مصادر المجموعات
SOURCES = ["webis", "quora"]

# ربط قاعدة البيانات
conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

# تهيئة خدمة المعالجة
preprocessor = TextPreprocessingService()

for source in SOURCES:
    print(f"\n📥 Loading original documents from source: {source}")
    cursor.execute("SELECT doc_id, content FROM documents WHERE source = ?", (source,))
    rows = cursor.fetchall()

    doc_ids = []
    clean_docs = []

    print(f"🧼 Preprocessing documents before TF-IDF for source: {source}")
    for doc_id, content in tqdm(rows, desc=f"{source} docs"):
        processed = preprocessor.preprocess(content, return_as_string=True)
        doc_ids.append(doc_id)
        clean_docs.append(processed)

    print(f"🔢 Creating TF-IDF Vectorizer for {source}...")
    vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(clean_docs)

    print(f"💾 Saving TF-IDF vectors for {source} in models/ ...")
    joblib.dump(doc_ids, os.path.join(MODELS_DIR, f"tfidf_{source}_doc_ids.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, f"tfidf_{source}_vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(MODELS_DIR, f"tfidf_{source}_matrix.joblib"))

print("\n✅ TF-IDF representation for all sources completed and saved in 'models/' folder.")
conn.close()
