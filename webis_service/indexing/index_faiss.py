import joblib
import numpy as np
import faiss

# المسارات
GROUP = 'webis'
MODEL_DIR = 'models'

# تحميل تمثيلات BERT
print("📦 تحميل تمثيلات BERT...")
embeddings = joblib.load(f"{MODEL_DIR}/bert_vectors_{GROUP}.joblib")
embeddings = np.array(embeddings).astype('float32')  # FAISS يحتاج float32

# بناء الفهرس
print("🧠 بناء فهرس FAISS...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# حفظ الفهرس
faiss.write_index(index, f"indexes/faiss_index_{GROUP}_bert.index")

print(f"✅ تم بناء الفهرس بعدد: {index.ntotal} متجه")
