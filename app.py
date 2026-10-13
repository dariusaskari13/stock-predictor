from pathlib import Path

from flask import Flask, request, jsonify, send_file
from model import (
    knn_predict,
    moving_average_predict,
    predict_price,
    random_forest_predict,
    svr_predict,
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

@app.route("/")
def index():
    return send_file(BASE_DIR / "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return "Request body must be a JSON object.", 400

    symbol = data.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return "Missing symbol", 400

    try:
        symbol = symbol.strip()
        return jsonify({
            "linear_regression": predict_price(symbol),
            "moving_average": moving_average_predict(symbol),
            "random_forest": random_forest_predict(symbol),
            "knn": knn_predict(symbol),
            "svr": svr_predict(symbol),
        })
    except FileNotFoundError:
        return "CSV file for that symbol was not found.", 400
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run()
