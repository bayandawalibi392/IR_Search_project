    import sqlite3
    from text_preprocessing_service import TextPreprocessingService
    from tqdm import tqdm

    # ربط قاعدة البيانات
    conn = sqlite3.connect("ir_project.db")
    cursor = conn.cursor()

    # إنشاء الجداول الجديدة
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preprocessed_documents (
        doc_id TEXT PRIMARY KEY,
        content TEXT,
        source TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preprocessed_queries (
        query_id TEXT PRIMARY KEY,
        query_text TEXT,
        source TEXT
    )
    """)

    conn.commit()

    # تهيئة خدمة المعالجة
    preprocessor = TextPreprocessingService()

    # الداتا سورس المراد معالجتها
    sources = ["quora"]

    # --- معالجة الوثائق ---
    for source in sources:
        print(f"\n🔄 Preprocessing documents from source: {source}")
        cursor.execute("SELECT doc_id, content FROM documents WHERE source = ?", (source,))
        docs = cursor.fetchall()

        for doc_id, content in tqdm(docs, desc=f"Docs - {source}"):
            cleaned = preprocessor.preprocess(content, return_as_string=True)
            cursor.execute("INSERT OR REPLACE INTO preprocessed_documents VALUES (?, ?, ?)",
                        (doc_id, cleaned, source))
        conn.commit()

    # --- معالجة الاستعلامات ---
    for source in sources:
        print(f"\n🔄 Preprocessing queries from source: {source}")
        cursor.execute("SELECT query_id, query_text FROM queries WHERE source = ?", (source,))
        queries = cursor.fetchall()

        for query_id, text in tqdm(queries, desc=f"Queries - {source}"):
            cleaned = preprocessor.preprocess(text, return_as_string=True)
            cursor.execute("INSERT OR REPLACE INTO preprocessed_queries VALUES (?, ?, ?)",
                        (query_id, cleaned, source))
        conn.commit()

    print("\n✅ تم تنفيذ المعالجة الأولية بنجاح وتخزين النتائج في الجداول الجديدة.")
    conn.close()
