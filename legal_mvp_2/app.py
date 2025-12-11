from flask import Flask, request, jsonify, render_template
from generator import ResponseGenerator
import config

app = Flask(__name__)

# Initialize Generator (Loads models on startup)
print("Initializing AI System...")
generator = ResponseGenerator()
print("AI System Ready.")

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    history = data.get('history', [])
    manual_domain = data.get('manual_domain')

    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        result = generator.generate(message, history, manual_domain)
        return jsonify(result)
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
