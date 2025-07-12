
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# تحميل الموارد المطلوبة من nltk
nltk.download('punkt')
nltk.download('stopwords')


class TextPreprocessingService:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text, return_as_string=False):
        # 1. تحويل الأحرف إلى صغيرة
        text = text.lower()

        # 2. إزالة علامات الترقيم والرموز
        text = re.sub(r'[^\w\s]', '', text)

        # 3. إزالة الأرقام
        text = re.sub(r'\d+', '', text)

        # 4. Tokenization
        tokens = word_tokenize(text)

        # 5. إزالة الكلمات الشائعة
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1]

        # 6. Stemming
        tokens = [self.stemmer.stem(word) for word in tokens]

        # إرجاع القائمة أو النص
        return " ".join(tokens) if return_as_string else tokens


# ✅ مثال للتجريب المباشر
if __name__ == "__main__":
    service = TextPreprocessingService()

    sample_text = "This is an example sentence, with numbers like 123 and punctuation!!!"
    
    print("🚀 Tokens:", service.preprocess(sample_text))
    print("🧼 Clean text:", service.preprocess(sample_text, return_as_string=True))
