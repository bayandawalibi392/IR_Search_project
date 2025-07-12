# text_preprocessor_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# تحميل الموارد
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

# أدوات المعالجة
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stem_tokens(tokens):
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.get_json()
    text = data.get('text', '')
    use_stemming = data.get('use_stemming', True)
    use_lemmatization = data.get('use_lemmatization', False)

    if not text.strip():
        return jsonify({"tokens": []})

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)

    if use_stemming:
        tokens = stem_tokens(tokens)
    elif use_lemmatization:
        tokens = lemmatize_tokens(tokens)

    return jsonify({"tokens": tokens})

if __name__ == '__main__':
    app.run(debug=True, port=5050)
