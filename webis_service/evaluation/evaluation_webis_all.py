import pandas as pd

# تحميل كل ملف تقييم مع تحديد نوع التمثيل
tfidf = pd.read_json("tfidf_evaluation_results.json", typ='series').to_frame().T
tfidf["Representation"] = "TF-IDF"

bert = pd.read_json("bert_evaluation_results.json", typ='series').to_frame().T
bert["Representation"] = "BERT"

faiss = pd.read_json("faiss_evaluation_results.json", typ='series').to_frame().T
faiss["Representation"] = "FAISS-BERT"

hybrid = pd.read_json("hybrid_evaluation_results.json", typ='series').to_frame().T
hybrid["Representation"] = "Hybrid (TF-IDF + BERT)"

# دمج كل النتائج في جدول واحد
all_results = pd.concat([tfidf, bert, faiss, hybrid], ignore_index=True)

# إعادة ترتيب الأعمدة لجعل اسم التمثيل أول عمود
cols = ["Representation"] + [col for col in all_results.columns if col != "Representation"]
all_results = all_results[cols]

# عرض النتائج في جدول
display(all_results)

# حفظ الجدول في ملف
all_results.to_csv("all_evaluation_results.csv", index=False)
