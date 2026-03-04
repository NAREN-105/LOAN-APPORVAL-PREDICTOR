"""Flask REST API for loan approval predictions."""
from flask import Flask, request, jsonify
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import LoanPredictor

app = Flask(__name__)
predictor = LoanPredictor()

REQUIRED_FIELDS = ["income", "credit_score", "loan_amount", "employment_years", "debt_to_income"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "stacking_ensemble"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = predictor.predict(data)
        return jsonify({
            "approved": result["approved"],
            "probability": round(result["probability"], 4),
            "confidence": result["confidence"],
            "decision": "APPROVED" if result["approved"] else "DENIED",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/explain", methods=["POST"])
def predict_explain():
    data = request.get_json(force=True)
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = predictor.predict(data)
    return jsonify({
        "approved": result["approved"],
        "probability": round(result["probability"], 4),
        "confidence": result["confidence"],
        "note": "Run predictor.explain_prediction(data) locally for SHAP chart",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)