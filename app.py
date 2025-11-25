from flask import Flask, request, jsonify, render_template
from train import spam_detector
from stats import get_spam_stats 
import os

app = Flask(__name__)

@app.route("/home")
def index():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:

        return jsonify({"error": "Campo 'text' é obrigatório."}), 400

    text = data["text"]

    try:

        result = spam_detector(text)

        return jsonify(result), 200

    except Exception as e:

        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():

    try:

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "emails.csv")

        if not os.path.exists(csv_path):

            return render_template("dashboard.html", stats=None, error="Arquivo emails.csv não encontrado.")

        stats = get_spam_stats(csv_path)

        return render_template("dashboard.html", stats=stats)

    except Exception as e:

        return render_template("dashboard.html", stats=None, error=str(e))


if __name__ == "__main__":

    app.run(debug=True)
