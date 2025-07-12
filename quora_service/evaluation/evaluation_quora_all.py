import pandas as pd
import json

# قائمة الملفات (غيّرها حسب ما لديك فعلياً)
evaluation_files = [
    ("TF-IDF", "tfidf_evaluation_results.json"),
    ("BERT", "bert_evaluation_results.json"),
    ("BM25", "bm25_evaluation_results.json"),
    ("Hybrid", "hybrid_evaluation_results.json")
]

# تحميل وتجميع النتائج
all_results = []

for name, filepath in evaluation_files:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["Representation"] = name
        all_results.append(data)

# تحويل إلى DataFrame
df = pd.DataFrame(all_results)

# ترتيب الأعمدة: نضع "Representation" أولاً
cols = ["Representation"] + [col for col in df.columns if col != "Representation"]
df = df[cols]

# عرض النتائج
display(df)

# حفظ الجدول النهائي
df.to_csv("all_representations_evaluation.csv", index=False)
df.to_json("all_representations_evaluation.json", orient="records", indent=4, force_ascii=False)
