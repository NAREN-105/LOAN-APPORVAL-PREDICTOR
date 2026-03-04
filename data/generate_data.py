"""Generates synthetic loan dataset."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

def generate_loan_dataset(n_samples=5000):
    education_levels = ["High School", "Associate", "Bachelor", "Master", "PhD"]
    education = np.random.choice(education_levels, n_samples, p=[0.20, 0.15, 0.40, 0.18, 0.07])
    employment_years = np.clip(np.random.exponential(scale=7, size=n_samples), 0, 40).astype(int)

    income_base = {"High School": 35000, "Associate": 45000, "Bachelor": 65000, "Master": 85000, "PhD": 100000}
    income = np.array([max(15000, int(income_base[e] * np.random.uniform(0.7, 1.5))) for e in education])
    age = np.clip(np.random.normal(38, 10, n_samples), 21, 70).astype(int)

    credit_score = np.clip(np.random.normal(680, 80, n_samples), 300, 850).astype(int)
    debt_to_income = np.clip(np.random.normal(0.35, 0.15, n_samples), 0.05, 0.95).round(2)
    assets = (income * np.random.uniform(0.5, 5.0, n_samples)).astype(int)
    savings = (assets * np.random.uniform(0.1, 0.4, n_samples)).astype(int)
    num_credit_lines = np.clip(np.random.poisson(4, n_samples), 0, 15)
    past_defaults = np.random.choice([0, 1, 2], n_samples, p=[0.75, 0.18, 0.07])
    credit_history_years = np.clip(np.random.normal(employment_years * 0.8, 3, n_samples), 0, 35).astype(int)

    loan_purpose = np.random.choice(
        ["Home Purchase", "Refinance", "Auto", "Personal", "Business", "Education"],
        n_samples, p=[0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
    )
    loan_amount = np.clip((income * np.random.uniform(1.5, 4.5, n_samples)).astype(int), 5000, 800000)
    loan_term_months = np.random.choice([12, 24, 36, 60, 120, 180, 240, 360], n_samples)
    interest_rate = np.clip(
        8.5 - (credit_score - 600) * 0.01 + np.random.normal(0, 0.5, n_samples), 2.5, 18.0
    ).round(2)

    property_value = (loan_amount * np.random.uniform(1.1, 2.5, n_samples)).astype(int)
    ltv_ratio = (loan_amount / property_value).round(2)
    property_type = np.random.choice(
        ["Single Family", "Condo", "Multi-Family", "Commercial"], n_samples, p=[0.55, 0.25, 0.12, 0.08]
    )
    location_risk = np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.5, 0.35, 0.15])

    monthly_income = income / 12
    monthly_payment = (loan_amount * (interest_rate/100/12) / (1 - (1 + interest_rate/100/12)**(-loan_term_months)))
    payment_to_income = (monthly_payment / monthly_income).round(2)

    approval_score = (
        (credit_score - 300) / 550 * 0.30
        + (1 - debt_to_income) * 0.20
        + np.clip(employment_years / 20, 0, 1) * 0.15
        + (1 - ltv_ratio) * 0.15
        + (1 - past_defaults / 2) * 0.10
        + np.clip(credit_history_years / 20, 0, 1) * 0.10
        + np.random.uniform(-0.05, 0.05, n_samples)
    )
    approved = (approval_score > 0.50).astype(int)

    return pd.DataFrame({
        "age": age, "education": education, "employment_years": employment_years,
        "income": income, "credit_score": credit_score, "debt_to_income": debt_to_income,
        "assets": assets, "savings": savings, "num_credit_lines": num_credit_lines,
        "past_defaults": past_defaults, "credit_history_years": credit_history_years,
        "loan_amount": loan_amount, "loan_term_months": loan_term_months,
        "interest_rate": interest_rate, "loan_purpose": loan_purpose,
        "property_value": property_value, "ltv_ratio": ltv_ratio,
        "property_type": property_type, "location_risk": location_risk,
        "monthly_income": monthly_income.round(2), "payment_to_income": payment_to_income,
        "approved": approved,
    })

if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df = generate_loan_dataset(5000)
    df.to_csv("data/raw/loan_data.csv", index=False)
    print(f"Saved 5000 rows | Approval rate: {df['approved'].mean():.1%}")