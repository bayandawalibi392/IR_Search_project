import sqlite3
from TextPreprocessor import TextPreprocessor
from tqdm import tqdm

# إعدادات
DB_PATH = "ir_project.db"
GROUPS = ['webis']  # ← ركّز فقط على المجموعة الجديدة

# الاتصال بقاعدة البيانات
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# إنشاء جدول لحفظ النصوص المعالجة
cursor.execute("""
CREATE TABLE IF NOT EXISTS preprocessed_documents (
    doc_id TEXT PRIMARY KEY,
    content TEXT,
    source TEXT
)
""")
conn.commit()

# تحميل أداة المعالجة مع تفعيل Stemming فقط
pre = TextPreprocessor()

# تنفيذ المعالجة
for group in GROUPS:
    print(f"\n🔄 معالجة الوثائق للمجموعة: {group}")
    cursor.execute("SELECT doc_id, content FROM documents WHERE source = ?", (group,))
    docs = cursor.fetchall()

    for doc_id, content in tqdm(docs):
        try:
            content = pre.preprocess(content, use_stemming=True, use_lemmatization=False)  # ✅ Stemming فقط
            content_str = ' '.join(content)
            cursor.execute(
                "INSERT OR REPLACE INTO preprocessed_documents (doc_id, content, source) VALUES (?, ?, ?)",
                (doc_id, content_str, group)
            )
        except Exception as e:
            print(f"⚠️ خطأ في الوثيقة {doc_id}: {e}")

    conn.commit()

print("\n✅ تم الانتهاء من معالجة وحفظ جميع الوثائق.")
conn.close()
