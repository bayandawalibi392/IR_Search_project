# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer

# # تحميل الموارد المطلوبة
# nltk.download('stopwords')
# nltk.download('wordnet')

# class TextPreprocessor:
#     def __init__(self, language='english'):
#         self.stop_words = set(stopwords.words(language))
#         self.stemmer = PorterStemmer()
#         self.lemmatizer = WordNetLemmatizer()

#     def clean_text(self, text):
#         text = text.lower()  # تحويل إلى أحرف صغيرة
#         text = re.sub(r'<[^>]+>', ' ', text)  # إزالة HTML
#         text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # إزالة الرموز والأرقام
#         text = re.sub(r'\s+', ' ', text).strip()  # إزالة الفراغات الزائدة
#         return text

#     def tokenize(self, text):
#         return text.split()  # بدون استخدام word_tokenize

#     def remove_stopwords(self, tokens):
#         return [word for word in tokens if word not in self.stop_words]

#     def stem_tokens(self, tokens):
#         return [self.stemmer.stem(word) for word in tokens]

#     def lemmatize_tokens(self, tokens):
#         return [self.lemmatizer.lemmatize(word) for word in tokens]

#     def preprocess(self, text, use_stemming=True, use_lemmatization=False):
#         cleaned = self.clean_text(text)
#         tokens = self.tokenize(cleaned)
#         tokens = self.remove_stopwords(tokens)

#         if use_stemming:
#             tokens = self.stem_tokens(tokens)
#         elif use_lemmatization:
#             tokens = self.lemmatize_tokens(tokens)

#         return tokens

# # اختبار سريع:
# if __name__ == '__main__':
#     pre = TextPreprocessor()
#     example = "The quick brown foxes are jumping over lazy dogs in 2024!"
#     result = pre.preprocess(example, use_stemming=True)
#     print(result)
from flask import Flask, request, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# تحميل الموارد المطلوبة
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        return text.split()

    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in self.stop_words]

    def stem_tokens(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    def preprocess(self, text, use_stemming=True, use_lemmatization=False):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)

        if use_stemming:
            tokens = self.stem_tokens(tokens)
        elif use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)

        return tokens

# إعداد Flask
app = Flask(__name__)
preprocessor = TextPreprocessor()

@app.route('/preprocess', methods=['POST'])
def preprocess_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    use_stemming = data.get("use_stemming", True)
    use_lemmatization = data.get("use_lemmatization", False)

    if not text.strip():
        return jsonify({"error": "Text is empty"}), 400

    result = preprocessor.preprocess(
        text,
        use_stemming=use_stemming,
        use_lemmatization=use_lemmatization
    )

    return jsonify({"tokens": result})

if __name__ == '__main__':
    app.run(debug=True)
