import joblib
import numpy as np
import faiss
import os

# إعدادات
GROUP = 'quora'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# تحميل تمثيلات BERT (يفترض أنها محفوظة مسبقًا)
print("📦 تحميل تمثيلات BERT...")
bert_vectors = joblib.load(f"{MODEL_DIR}/bert_{GROUP}_vectors.joblib")
doc_ids = joblib.load(f"{MODEL_DIR}/bert_{GROUP}_doc_ids.joblib")  # ترتيب مطابق

# تحويل التمثيلات إلى float32 كما يتطلب FAISS
bert_vectors = np.array(bert_vectors).astype('float32')

# بناء الفهرس باستخدام FAISS
print("🧠 بناء فهرس FAISS...")
dimension = bert_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)  # فهرس باستخدام مسافة L2
index.add(bert_vectors)

# حفظ الفهرس
faiss.write_index(index, f"indexes/faiss_index_{GROUP}_bert.index")
print(f"✅ تم بناء وحفظ فهرس FAISS لعدد: {index.ntotal} وثيقة.")
