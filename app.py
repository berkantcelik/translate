import os
from flask import Flask, request, jsonify, render_template # render_template'i içe aktardık
from google import genai
from google.genai.errors import APIError

# Flask uygulamasını başlat
# templates_folder='templates' kısmı, Flask'a HTML'i nerede bulacağını söyler.
app = Flask(__name__, template_folder='templates') 

# --- API İstemcisi ve Model Konfigürasyonu ---

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY ortam değişkeni ayarlanmadı.")
    client = genai.Client(api_key=api_key)
except Exception as e:
    app.logger.error(f"Gemini istemcisi oluşturulamadı veya API anahtarı eksik: {e}")
    client = None

MODEL_NAME = "gemini-2.5-flash"

# --- API Uç Noktaları (Endpoints) ---

# 1. Ana Sayfa Uç Noktası (index.html'i sunar)
@app.route('/', methods=['GET'])
def home():
    """Tarayıcıya index.html dosyasını gönderir."""
    # Flask, templates klasöründeki index.html'i arayacaktır
    return render_template('index.html')


# 2. Çeviri Uç Noktası (API)
@app.route('/translate', methods=['POST'])
def translate_text():
    """Türkçe metni doğal ve akıcı İngilizceye çevirir."""
    
    if not client:
        return jsonify({"error": "Sunucu hatası: Gemini API istemcisi kullanıma hazır değil."}), 500

    data = request.get_json(silent=True)
    turkish_text = data.get('text') if data else None

    if not turkish_text:
        return jsonify({"error": "Lütfen çevrilecek metni ('text' alanında) içeren geçerli bir JSON gövdesi sağlayın."}), 400

    # İnsancıl Prompt
    prompt = f"""
    You are a professional translator who thinks like a human and uses fluent, natural language. 
    Translate the Turkish text below into English in a way that an English-speaking audience would perceive as completely natural and human-written. 
    Absolutely avoid formal dictionary-style translations, AI jargon, or dull repetitions. Only provide the translation and add no extra explanation or notes.
    
    Turkish Text: "{turkish_text}"
    """

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.8
            )
        )
        
        translated_text = response.text.strip()
        return jsonify({
            "original_text_tr": turkish_text,
            "translated_text_en": translated_text,
            "powered_by": "Gemini API"
        })

    except APIError as e:
        app.logger.error(f"Gemini API hatası: {e}")
        return jsonify({"error": "Gemini API çağrılırken bir hata oluştu. Anahtar veya kota sorun olabilir."}), 500
    except Exception as e:
        app.logger.error(f"Beklenmedik hata: {e}")
        return jsonify({"error": "Beklenmedik bir sunucu hatası oluştu."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)