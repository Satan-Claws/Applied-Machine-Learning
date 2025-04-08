from flask import Flask, request, jsonify
from flask_cors import CORS
from score import load_model, score

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model, vectorizer = load_model()

@app.route('/score', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data['text']
        threshold = float(data.get('threshold', 0.5))
        
        prediction, propensity = score(text, model, vectorizer, threshold)
        
        return jsonify({
            "prediction": bool(prediction),
            "propensity": float(propensity)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 