import os
import sqlite3
import joblib
from rank_bm25 import BM25Okapi
from TextPreprocessor import TextPreprocessor
from tqdm import tqdm  # ✅ إضافة tqdm

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

def build_bm25_for_group(group_name):
    cursor.execute("SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (group_name,))
    rows = cursor.fetchall()

    doc_ids = []
    tokenized_documents = []

    print(f"🔄 Processing {len(rows)} documents from group '{group_name}'...")

    # ✅ إضافة tqdm هنا لعرض التقدم
    for doc_id, content_str in tqdm(rows, desc="🔍 Tokenizing documents"):
        if content_str:
            doc_ids.append(doc_id)
            tokens = pre.tokenize(pre.clean_text(content_str))
            tokenized_documents.append(tokens)

    print(f"✅ Loaded {len(tokenized_documents)} preprocessed documents from group '{group_name}'")

    # إنشاء نموذج BM25
    bm25 = BM25Okapi(tokenized_documents)

    # التخزين
    joblib.dump(bm25, f"{OUTPUT_DIR}/bm25_model_{group_name}.joblib")
    joblib.dump(doc_ids, f"{OUTPUT_DIR}/doc_ids_bm25_{group_name}.joblib")
    joblib.dump(tokenized_documents, f"{OUTPUT_DIR}/bm25_tokenized_docs_{group_name}.joblib")

    print(f"💾 Saved BM25 model for group '{group_name}'")

# تنفيذ
for group in GROUPS:
    build_bm25_for_group(group)

print("✅ BM25 vectorization completed for all documents per group.")
