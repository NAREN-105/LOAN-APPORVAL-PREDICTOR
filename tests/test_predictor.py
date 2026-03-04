"""Unit tests for LoanPredictor."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SAMPLE_APPLICANT = {
    "income": 65000,
    "credit_score": 720,
    "loan_amount": 200000,
    "employment_years": 5,
    "debt_to_income": 0.35,
    "property_value": 350000,
    "education": "Bachelor",
    "loan_purpose": "Home Purchase",
}

RISKY_APPLICANT = {
    "income": 25000,
    "credit_score": 520,
    "loan_amount": 300000,
    "employment_years": 0,
    "debt_to_income": 0.80,
    "past_defaults": 2,
}


def get_predictor():
    from predictor import LoanPredictor
    return LoanPredictor()


def test_model_loads():
    p = get_predictor()
    assert p.model is not None
    assert p.preprocessor is not None


def test_prediction_returns_dict():
    p = get_predictor()
    result = p.predict(SAMPLE_APPLICANT)
    assert "probability" in result
    assert "approved" in result
    assert "confidence" in result


def test_probability_range():
    p = get_predictor()
    result = p.predict(SAMPLE_APPLICANT)
    assert 0.0 <= result["probability"] <= 1.0


def test_good_applicant_approved():
    p = get_predictor()
    result = p.predict(SAMPLE_APPLICANT)
    assert result["approved"] is True


def test_risky_applicant_denied():
    p = get_predictor()
    result = p.predict(RISKY_APPLICANT)
    assert result["approved"] is False


def test_confidence_values():
    p = get_predictor()
    result = p.predict(SAMPLE_APPLICANT)
    assert result["confidence"] in ["High", "Medium", "Low"]