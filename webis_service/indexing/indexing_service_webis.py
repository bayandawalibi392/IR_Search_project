import sqlite3
import joblib
from collections import defaultdict
from tqdm import tqdm
import os

# الإعدادات
GROUP = 'webis'
DB_PATH = 'ir_project.db'
OUTPUT_DIR = 'indexes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# تحميل doc_ids بنفس ترتيب تمثيل TF-IDF
doc_ids = joblib.load(f"models/doc_ids_{GROUP}.joblib")

# جلب التوكنات المعالجة المرتبطة بالوثائق
print(f"📄 تحميل التوكنات لـ {len(doc_ids)} وثيقة من الجدول المعالج...")
cursor.execute(
    f"SELECT doc_id, content FROM preprocessed_documents WHERE source = ?", (GROUP,)
)
token_map = {row[0]: row[1] for row in cursor.fetchall()}

# بناء الفهرس المعكوس
print("🔄 بناء الفهرس المعكوس...")
inverted_index = defaultdict(set)

for idx, doc_id in tqdm(enumerate(doc_ids), total=len(doc_ids)):
    content_str = token_map.get(doc_id, "")
    content = content_str.split()  # لأننا خزناها كسلسلة نصية مفصولة بمسافات
    for token in set(content):  # نتجنب التكرار داخل نفس الوثيقة
        inverted_index[token].add(idx)

# حفظ الفهرس
joblib.dump(inverted_index, f"{OUTPUT_DIR}/inverted_index1_{GROUP}.joblib")
print(f"✅ تم حفظ الفهرس المعكوس في {OUTPUT_DIR}/inverted_index1_{GROUP}.joblib")
