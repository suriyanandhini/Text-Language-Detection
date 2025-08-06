"""
Flask Web App for Text Language Detection
-----------------------------------------
Run with:
    python app.py
Then open http://127.0.0.1:5000/ in your browser.

Note: For production-grade language detection, consider using libraries like 'langdetect', 'langid', or 'fastText'.
"""
from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Model setup (expanded dataset, char n-grams) ---
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
model = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(1, 4)), MultinomialNB())
model.fit(texts, languages)

def detect_language(text):
    return model.predict([text])[0]

# --- Flask app ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang='en'>
<head>
    <meta charset='utf-8'>
    <title>Text Language Detection</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f9f9f9; }
        .container { max-width: 500px; margin: auto; background: #fff; padding: 2em; border-radius: 8px; box-shadow: 0 2px 8px #ccc; }
        h1 { color: #333; }
        input[type=text], textarea { width: 100%; padding: 0.5em; margin: 0.5em 0 1em 0; border: 1px solid #ccc; border-radius: 4px; }
        button { padding: 0.5em 1.5em; background: #007bff; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
        .result { margin-top: 1em; font-size: 1.2em; color: #007bff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Text Language Detection</h1>
        <form method="post">
            <label for="text">Enter a sentence:</label><br>
            <textarea name="text" id="text" rows="3" required>{{ request.form.text or '' }}</textarea><br>
            <button type="submit">Detect Language</button>
        </form>
        {% if result %}
            <div class="result">Predicted Language: <b>{{ result }}</b></div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        user_text = request.form['text']
        result = detect_language(user_text)
    return render_template_string(HTML_TEMPLATE, result=result, request=request)

if __name__ == '__main__':
    app.run(debug=True)