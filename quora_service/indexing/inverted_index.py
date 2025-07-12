import os
import sqlite3
import joblib
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from tqdm import tqdm

# تحميل أدوات NLTK
nltk.download('punkt')

# إعداد المسارات
DB_PATH = "ir_project.db"
INDEX_DIR = "indexes"
SOURCES = ['quora']
os.makedirs(INDEX_DIR, exist_ok=True)

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

for source in SOURCES:
    print(f"\n📚 معالجة مجموعة البيانات: {source}")

    # جلب الوثائق المعالجة من قاعدة البيانات
    cursor.execute("SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (source,))
    rows = cursor.fetchall()
    if not rows:
        print(f"⚠️ لا توجد مستندات معالجة للمصدر {source}، تخطي...")
        continue

    inverted_index = defaultdict(set)  # term → set of doc_ids

    print(f"📥 بناء الفهرس المعكوس لـ {len(rows)} مستند...")
    for doc_id, content in tqdm(rows, desc=f"Indexing {source}"):
        if not content.strip():
            continue
        tokens = word_tokenize(content.lower())
        for token in set(tokens):  # نتجنب التكرار داخل نفس المستند
            inverted_index[token].add(doc_id)

    # تحويل القيم إلى قوائم
    inverted_index = {term: list(doc_ids) for term, doc_ids in inverted_index.items()}

    # حفظ الفهرس في مجلد indexes
    index_path = os.path.join(INDEX_DIR, f"inverted_index_{source}.joblib")
    joblib.dump(inverted_index, index_path)
    print(f"✅ تم حفظ الفهرس في: {index_path}")

conn.close()
print("\n🏁 انتهى بناء وحفظ الفهارس المعكوسة.")
