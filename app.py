from flask import Flask, request, jsonify, send_from_directory
from train import spam_detector

app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.get("/")
def home():
    return {"message": "API online"}
    
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Campo 'text' é obrigatório."}), 400

    text = data.get("text", "")

    try:
        result = spam_detector(text)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)