import os
from flask import Flask, request, jsonify
from google import genai
from google.genai.errors import APIError

# Initialize the Flask application
app = Flask(__name__)

# --- API Client and Model Configuration ---

# The API Key is retrieved from the Render environment variable.
# Key: GEMINI_API_KEY
try:
    # Ensure the client is only created if the API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)
except Exception as e:
    app.logger.error(f"Failed to create Gemini client: {e}")
    client = None

MODEL_NAME = "gemini-2.5-flash"

# --- Translation Endpoint ---

@app.route('/translate', methods=['POST'])
def translate_text():
    """API endpoint that translates Turkish text to natural-sounding English."""
    
    if not client:
        return jsonify({"error": "API client is not ready. Please check the GEMINI_API_KEY environment variable."}), 500

    # 1. Get the input
    data = request.get_json()
    turkish_text = data.get('text')

    if not turkish_text:
        return jsonify({"error": "Please provide the text to be translated in the 'text' field."}), 400

    # 2. Humanized Prompt for Natural Output
    # This prompt instructs the model to avoid generic AI phrasing.
    prompt = f"""
    You are a professional translator who thinks like a human and uses fluent, natural language. 
    Translate the Turkish text below into English in a way that an English-speaking audience would perceive as completely natural and human-written. 
    Absolutely avoid formal dictionary-style translations, AI jargon, or dull repetitions. Only provide the translation and add no extra explanation or notes.
    
    Turkish Text: "{turkish_text}"
    """

    try:
        # 3. Call the Gemini API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                # High Temperature (0.8) ensures higher creativity and less robotic word choice
                temperature=0.8
            )
        )
        
        # 4. Clean up the response and return
        translated_text = response.text.strip()
        return jsonify({
            "original_text_tr": turkish_text,
            "translated_text_en": translated_text,
            "powered_by": "Gemini API"
        })

    except APIError as e:
        app.logger.error(f"Gemini API error: {e}")
        return jsonify({"error": "An error occurred while calling the Gemini API."}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected server error occurred."}), 500

# The following lines are necessary for the server to run
if __name__ == '__main__':
    # Run in development mode on port 8080
    app.run(debug=True, port=8080)