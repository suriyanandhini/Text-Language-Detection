"""
Text Language Detection using scikit-learn
--------------------------------------------------
This script demonstrates a simple way to detect the language of a given text using NLP techniques.
It uses a custom dataset, scikit-learn's TfidfVectorizer (with character n-grams), and Multinomial Naive Bayes classifier.

Note: For production-grade language detection, consider using libraries like 'langdetect', 'langid', or 'fastText'.

Requirements:
- scikit-learn
- numpy

How to run:
1. Install dependencies (if not already):
   pip install scikit-learn numpy
2. Run the script:
   python language_detection.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np

# 1. Create a larger, more varied custom dataset with texts in different languages
texts = [
    # English
    "Hello, how are you?",
    "What is your name?",
    "Good morning!",
    "I love programming in Python.",
    "This is a beautiful day.",
    "Can you help me?",
    "The quick brown fox jumps over the lazy dog.",
    "Learning new languages is fun and rewarding.",
    "Please enter your password.",
    "Where is the nearest restaurant?",
    # French
    "Bonjour, comment ça va?",
    "Quel est ton nom?",
    "Comment allez-vous?",
    "J'aime programmer en Python.",
    "C'est une belle journée.",
    "Pouvez-vous m'aider?",
    "Le renard brun rapide saute par-dessus le chien paresseux.",
    "Apprendre de nouvelles langues est amusant et enrichissant.",
    "Veuillez entrer votre mot de passe.",
    "Où est le restaurant le plus proche?",
    # Spanish
    "Hola, ¿cómo estás?",
    "¿Cuál es tu nombre?",
    "¿Dónde está la biblioteca?",
    "Me gusta programar en Python.",
    "Es un día hermoso.",
    "¿Puedes ayudarme?",
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "Aprender nuevos idiomas es divertido y gratificante.",
    "Por favor, introduce tu contraseña.",
    "¿Dónde está el restaurante más cercano?",
    # Hindi
    "नमस्ते, आप कैसे हैं?",
    "आपका नाम क्या है?",
    "मुझे हिंदी आती है।",
    "हाय, मैं सुरिया हूँ",
    "यह एक सुंदर दिन है।",
    "क्या आप मेरी मदद कर सकते हैं?",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "नई भाषाएँ सीखना मज़ेदार और लाभकारी है।",
    "कृपया अपना पासवर्ड दर्ज करें।",
    "सबसे नज़दीकी रेस्टोरेंट कहाँ है?",
    # German
    "Hallo, wie geht es dir?",
    "Wie heißt du?",
    "Guten Morgen!",
    "Ich programmiere gerne in Python.",
    "Das ist ein schöner Tag.",
    "Kannst du mir helfen?",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Neue Sprachen zu lernen macht Spaß und ist lohnend.",
    "Bitte geben Sie Ihr Passwort ein.",
    "Wo ist das nächste Restaurant?",
    # Italian
    "Ciao, come stai?",
    "Come ti chiami?",
    "Buongiorno!",
    "Mi piace programmare in Python.",
    "È una bella giornata.",
    "Puoi aiutarmi?",
    "La veloce volpe marrone salta sopra il cane pigro.",
    "Imparare nuove lingue è divertente e gratificante.",
    "Per favore, inserisci la tua password.",
    "Dov'è il ristorante più vicino?",
    # Chinese
    "你好，你好吗？",
    "你叫什么名字？",
    "早上好！",
    "我喜欢用Python编程。",
    "今天是美好的一天。",
    "你能帮我吗？",
    "敏捷的棕色狐狸跳过了懒狗。",
    "学习新语言既有趣又有收获。",
    "请输入您的密码。",
    "最近的餐厅在哪里？",
    # Arabic
    "مرحبًا، كيف حالك؟",
    "ما اسمك؟",
    "صباح الخير!",
    "أحب البرمجة بلغة بايثون.",
    "إنه يوم جميل.",
    "هل يمكنك مساعدتي؟",
    "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "تعلم لغات جديدة ممتع ومفيد.",
    "من فضلك أدخل كلمة المرور الخاصة بك.",
    "أين أقرب مطعم؟",
]
languages = [
    # English
    *["English"]*10,
    # French
    *["French"]*10,
    # Spanish
    *["Spanish"]*10,
    # Hindi
    *["Hindi"]*10,
    # German
    *["German"]*10,
    # Italian
    *["Italian"]*10,
    # Chinese
    *["Chinese"]*10,
    # Arabic
    *["Arabic"]*10,
]

# 2. Convert text into features using TfidfVectorizer with character n-grams
# 3. Train a simple classification model (Multinomial Naive Bayes)
model = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(1, 4)), MultinomialNB())
model.fit(texts, languages)

# 4. Function to predict language
def detect_language(text):
    return model.predict([text])[0]

# 5. Take input text from the user and predict the language
if __name__ == "__main__":
    print("\n=== Text Language Detection ===\n")
    user_text = input("Enter a sentence to detect its language: ")
    predicted_lang = detect_language(user_text)
    print(f"\nPredicted Language: {predicted_lang}\n")

    # 6. Test examples
    test_examples = [
        "Good morning!",           # English
        "Comment allez-vous?",    # French
        "¿Dónde está la biblioteca?", # Spanish
        "मुझे हिंदी आती है।",         # Hindi
        "Guten Morgen!",              # German
        "Buongiorno!",                # Italian
        "早上好！",                      # Chinese
        "صباح الخير!",                  # Arabic
    ]
    print("Some test examples:")
    for example in test_examples:
        lang = detect_language(example)
        print(f"  Text: {example}\n  Predicted: {lang}\n")