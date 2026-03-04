"""Train and evaluate all ensemble models, then save the best pipeline."""
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/raw/loan_data.csv")
if not DATA_PATH.exists():
    print("Generating dataset...")
    from data.generate_data import generate_loan_dataset
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_loan_dataset(5000)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} rows | Approval rate: {df['approved'].mean():.1%}")

# ── Features ─────────────────────────────────────────────────────────────────
TARGET = "approved"
CATEGORICAL = ["education", "loan_purpose", "property_type", "location_risk"]
NUMERICAL = [c for c in df.columns if c not in CATEGORICAL + [TARGET]]

X = df[NUMERICAL + CATEGORICAL]
y = df[TARGET]
FEATURE_NAMES = NUMERICAL + CATEGORICAL

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Preprocessor ─────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERICAL),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
])
preprocessor.fit(X_train)

X_train_t = preprocessor.transform(X_train)
X_test_t  = preprocessor.transform(X_test)

# ── Models ───────────────────────────────────────────────────────────────────
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                     eval_metric="logloss", random_state=42)

rf  = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                              random_state=42, n_jobs=-1)

gb  = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                                  random_state=42)

lr  = LogisticRegression(max_iter=1000, random_state=42)

stack = StackingClassifier(
    estimators=[("xgb", xgb), ("rf", rf), ("gb", gb)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1,
)

models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb, "Stacking Ensemble": stack}

# ── Train & Evaluate ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'AUC':>7}")
print("="*65)

results = {}
for name, model in models.items():
    model.fit(X_train_t, y_train)
    y_pred  = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]
    results[name] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_proba),
    }
    r = results[name]
    print(f"{name:<22} {r['accuracy']:>9.1%} {r['precision']:>10.1%} {r['recall']:>8.1%} {r['auc']:>7.3f}")

print("="*65)

# ── Save best model ───────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
joblib.dump(stack,        "models/stacking_ensemble.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(FEATURE_NAMES,"models/feature_names.pkl")
print("\nModels saved to models/")

# ── Confusion matrix plot ─────────────────────────────────────────────────────
y_pred_stack = stack.predict(X_test_t)
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Denied", "Approved"], yticklabels=["Denied", "Approved"])
plt.title("Stacking Ensemble — Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
print("Confusion matrix saved to models/confusion_matrix.png")

# ── ROC curve ────────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_t)[:, 1])
    auc = results[name]["auc"]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("ROC Curves — All Models"); plt.legend(); plt.tight_layout()
plt.savefig("models/roc_curves.png", dpi=150)
print("ROC curves saved to models/roc_curves.png")