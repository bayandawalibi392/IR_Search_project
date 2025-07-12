import os
import sqlite3
import joblib
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from tqdm import tqdm

nltk.download('punkt')

MODELS_DIR = "models"
SOURCES = ['webis', 'quora']
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

conn = sqlite3.connect("ir_project.db")
cursor = conn.cursor()

for source in SOURCES:
    print(f"🔄 معالجة المصدر: {source}")

    cursor.execute("SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (source,))
    rows = cursor.fetchall()
    if not rows:
        print(f"⚠️ لا توجد مستندات لمصدر {source}، تخطي...")
        continue

    doc_ids, contents = zip(*rows)
    print(f"عدد المستندات: {len(contents)}")

    print(f"بدء تقسيم المحتوى إلى كلمات لـ {source}...")
    tokenized_corpus = [word_tokenize(doc) for doc in tqdm(contents, desc=f"Tokenizing {source} docs")]
    print(f"انتهى تقسيم المحتوى لـ {source}.")

    print(f"بدء بناء نموذج BM25 لـ {source}...")
    bm25 = BM25Okapi(tokenized_corpus, k1=DEFAULT_K1, b=DEFAULT_B)
    print(f"تم بناء نموذج BM25 لـ {source}.")

    data_to_save = {
        'doc_ids': doc_ids,
        'tokenized_texts': tokenized_corpus,
        'k1': DEFAULT_K1,
        'b': DEFAULT_B
    }

    save_path = os.path.join(MODELS_DIR, f"bm25_{source}_model.joblib")
    joblib.dump(data_to_save, save_path)
    print(f"✅ تم حفظ نموذج BM25 للمصدر {source} في {save_path}\n")

conn.close()
print("انتهى إنشاء وحفظ نماذج BM25 لكل المصادر.")
