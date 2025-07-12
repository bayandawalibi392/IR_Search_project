 مشروع نظام استرجاع المعلومات متعدد التمثيلات

نظام متكامل لاسترجاع المعلومات تم تطويره باستخدام عدة طرق تمثيل نصي (TF-IDF, BM25, BERT، وHybrid)، مع دعم فهرسة سريعة باستخدام FAISS لتسريع عمليات البحث الدلالي.

---

 مجموعات البيانات المستخدمة

تم استخدام مجموعتين من البيانات من [ir-datasets.com](https://ir-datasets.com):

1. **BEIR - Webis-Touche2020**  
   - أسئلة نقاشية معقدة تهدف لاختبار أنظمة استرجاع المعلومات في بيئة حقيقية.
   - تحتوي على استعلامات تتطلب تحليلًا دلاليًا عميقًا.

2. **BEIR - Quora Question Pairs (dev)**  
   - تحتوي على أسئلة متكررة من موقع Quora.
   - هدفها اختبار قدرة النظام على تمييز الأسئلة المتكررة/المتشابهة.

---
 الخدمات ضمن النظام

تم بناء كل محرك بحث كخدمة منفصلة (Microservice) باستخدام Flask:

 `/search-tfidf-inverted` | TF-IDF  | تمثيل تقليدي يعتمد على تواتر المصطلحات. |
 `/search-bm25-inverted` | BM25  | نموذج إحصائي متطور يستند إلى TF-IDF. |
 `/search-bert-inverted` | BERT  | تمثيل دلالي باستخدام Transformers. |
 `/search-hybrid-inverted` | Hybrid (TF-IDF + BERT)  | دمج التمثيلين مع إمكانية ضبط الوزن (α). |
 `/preprocess` | معالجة نصية |  تقوم بتنظيف وتحليل الاستعلامات قبل التمثيل. |

---

## 🏗️ بنية النظام (System Architecture)

النظام مبني وفق مفهوم **Service-Oriented Architecture (SOA)**، حيث:

- كل محرك بحث هو خدمة مستقلة.
- يتم التواصل مع خدمة خارجية للمعالجة المسبقة عبر HTTP.
- قاعدة بيانات SQLite موحدة لخزن الوثائق.
- النماذج والتمثيلات محملة مسبقًا باستخدام joblib.
- خدمة FAISS تستخدم لتمثيل BERT بشكل سريع.

```plaintext
Client
  │
  ├──> /search-xxx-inverted (TF-IDF, BM25, BERT, Hybrid)
  │       │
  │       └──> /preprocess (text cleaning & tokenization)
  │       └──> SQLite (document content)
  │       └──> Vectors/Indexes (joblib, FAISS)
