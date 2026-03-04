"""LoanPredictor — load trained model and make predictions with SHAP explanations."""
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


class LoanPredictor:
    MODEL_PATH = Path("models/stacking_ensemble.pkl")
    PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

    def __init__(self):
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run: python train_model.py")
        self.model = joblib.load(self.MODEL_PATH)
        self.preprocessor = joblib.load(self.PREPROCESSOR_PATH)
        self.feature_names = joblib.load("models/feature_names.pkl")

    def _prepare(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])
        # Fill optional fields with defaults
        defaults = {
            "age": 35, "assets": data.get("income", 50000) * 2,
            "savings": data.get("income", 50000) * 0.3,
            "num_credit_lines": 4, "past_defaults": 0,
            "credit_history_years": data.get("employment_years", 3),
            "loan_term_months": 360, "interest_rate": 7.5,
            "property_value": data.get("loan_amount", 100000) * 1.5,
            "ltv_ratio": 0.67, "property_type": "Single Family",
            "location_risk": "Medium", "monthly_income": data.get("income", 50000) / 12,
            "payment_to_income": 0.30,
        }
        for k, v in defaults.items():
            if k not in df.columns:
                df[k] = v
        return df[self.feature_names]

    def predict(self, data: dict) -> dict:
        df = self._prepare(data)
        X = self.preprocessor.transform(df)
        prob = float(self.model.predict_proba(X)[0][1])
        return {
            "probability": prob,
            "approved": prob >= 0.50,
            "confidence": "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low",
        }

    def explain_prediction(self, data: dict):
        df = self._prepare(data)
        X = self.preprocessor.transform(df)
        explainer = shap.TreeExplainer(self.model.named_estimators_["xgb"])
        shap_values = explainer.shap_values(X)
        impacts = dict(zip(self.feature_names, shap_values[0]))
        sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        print("\nSHAP Explanation:")
        print(f"+{'':─<21}+{'':─<14}+")
        print(f"| {'Feature':<19} | {'Impact':<12} |")
        print(f"+{'':─<21}+{'':─<14}+")
        for feat, val in sorted_impacts:
            bar = "█" * min(int(abs(val) * 20), 8)
            sign = "+" if val > 0 else "-"
            print(f"| {feat:<19} | {bar:<5} {sign}{abs(val):.2f}   |")
        print(f"+{'':─<21}+{'':─<14}+")