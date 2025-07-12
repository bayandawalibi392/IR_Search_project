from flask import Flask, request, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# تحميل الموارد من NLTK
nltk.download('punkt')
nltk.download('stopwords')

# إعداد Flask
app = Flask(__name__)

# كلاس المعالجة النصية
class TextPreprocessingService:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text, return_as_string=False):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return " ".join(tokens) if return_as_string else tokens


# إنشاء نسخة من الخدمة
service = TextPreprocessingService()

# ✅ Endpoint للمعالجة
@app.route('/preprocess', methods=['POST'])
def preprocess_endpoint():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({'error': '⚠️ يجب إرسال حقل "text" ضمن الطلب'}), 400

    text = data['text']
    return_as_string = data.get('return_as_string', False)

    tokens = service.preprocess(text, return_as_string=return_as_string)

    return jsonify({
        'tokens': tokens if not return_as_string else tokens.split(),
        'clean_text': tokens if return_as_string else " ".join(tokens)
    })

# تشغيل التطبيق
if __name__ == '__main__':
    app.run(port=5060, debug=True)
