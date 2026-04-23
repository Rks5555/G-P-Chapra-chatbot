from flask import Flask, render_template, request, jsonify
from fuzzywuzzy import fuzz
import json
import re
import os
import random

# ---------------- LOAD DATA (PATH FIX) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data.json")

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# ---------------- CHATBOT FUNCTION (WITHOUT SPACY) -----------
def chatbot(user_input):
    user_input = clean_text(user_input)

    best_match = None
    best_score = 0
 for item in data:
     if not isinstance(item, dict):
        continue

    if "patterns" not in item or 
"response" not in item:
        continue

    for pattern in item["patterns"]:
        if not isinstance(pattern, str):
            continue

        pattern_clean = 
clean_text(pattern)
        score = 
fuzz.token_set_ratio(user_input,
pattern_clean)

        if score > best_score:
            best_score = score
            best_match = item
    
                

    # ---------------- SAFE RESPONSE HANDLING ----------------
if best_match and "response" in best_match:
    response_data = best_match["response"]

    if isinstance(response_data, list):
        response_text = random.choice(response_data)
    elif isinstance(response_data, str):
        response_text = response_data
    else:
        response_text = "Sorry, something went wrong."
else:
    response_text = "Sorry, I couldn't understand your question."
    if best_score >= 75:
        return response_text

    elif 60 <= best_score < 75:
        return f"🙂 I think you are asking this:\n{response_text}"

    elif 45 <= best_score < 60:
        return f"🤔 Not fully sure, but maybe this helps:\n{response_text}"

    else:
        return "Sorry, I couldn't understand your question. Try asking differently."
    

# ---------------- ROUTES ----------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data_json = request.get_json()

        if not data_json or "message" not in data_json:
            return jsonify({"reply": "Please type something"})

        user_input = data_json["message"]

        response = chatbot(user_input)

        return jsonify({"reply": response})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "Server error occurred"}), 500
      # ---------------- RUN APP ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # IMPORTANT
    app.run(host="0.0.0.0", port=port, debug=False)  
