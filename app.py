import os
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai.errors import APIError

# Flask uygulamasını başlat. Flask'a HTML dosyalarını 'templates' klasöründe aramasını söylüyoruz.
app = Flask(__name__, template_folder='templates') 

# --- API Client and Model Configuration ---

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)
except Exception as e:
    app.logger.error(f"Gemini client failed to initialize: {e}")
    client = None

MODEL_NAME = "gemini-2.5-flash"

# --- API Endpoints ---

# 1. Root Path: Serves the HTML interface
@app.route('/', methods=['GET'])
def home():
    """Serves the index.html file located in the 'templates' folder."""
    return render_template('index.html')


# 2. Translation API Endpoint
@app.route('/translate', methods=['POST'])
def translate_text():
    """Translates Turkish text to natural and fluent English."""
    
    if not client:
        return jsonify({"error": "Server error: Gemini API client is not ready."}), 500

    data = request.get_json(silent=True)
    turkish_text = data.get('text') if data else None

    if not turkish_text:
        return jsonify({"error": "Please provide valid JSON body with 'text' field."}), 400

    # Humanized Prompt for natural output
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
        app.logger.error(f"Gemini API error: {e}")
        return jsonify({"error": "Gemini API call failed. Check key or quota."}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)