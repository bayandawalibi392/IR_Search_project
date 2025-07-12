import os
import sqlite3
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from TextPreprocessor import TextPreprocessor  # لا مشكلة في استخدامه يدويًا هنا

# إعدادات
DB_PATH = 'ir_project.db'
OUTPUT_DIR = 'models'
GROUPS = ['webis']
os.makedirs(OUTPUT_DIR, exist_ok=True)

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# كائن المعالجة
pre = TextPreprocessor()

def build_tfidf_for_group(group_name):
    cursor.execute("SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (group_name,))
    rows = cursor.fetchall()

    doc_ids = []
    processed_texts = []

    for doc_id, content_str in rows:
        if content_str:
            doc_ids.append(doc_id)

            # ⚠️ معالجة المحتوى بالكامل يدوياً، بدون تمريره داخل الـ Vectorizer
            tokens = pre.preprocess(content_str, use_stemming=True, use_lemmatization=False)
            clean_text = ' '.join(tokens)
            processed_texts.append(clean_text)

    print(f"✅ Loaded and processed {len(processed_texts)} documents from group '{group_name}'")

    # ✅ إنشاء vectorizer بدون preprocessor/tokenizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    # ✅ التخزين بدون تضمين أي كائن مخصص
    joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer_{group_name}.joblib")
    joblib.dump(tfidf_matrix, f"{OUTPUT_DIR}/tfidf_vectors_{group_name}.joblib")
    joblib.dump(doc_ids, f"{OUTPUT_DIR}/doc_ids_{group_name}.joblib")
    print(f"💾 Saved TF-IDF model for group '{group_name}'")

# تنفيذ
for group in GROUPS:
    build_tfidf_for_group(group)

print("✅ TF-IDF vectorization completed and saved WITHOUT custom objects.")
