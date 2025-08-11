# scripts/train_model.py
import os
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Resolve paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "enriched_transaction_data.csv",
    ROOT / "datasets" / "enriched_transaction_data.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("enriched_transaction_data.csv not found in data/ or datasets/")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using dataset: {DATA_PATH}")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

target = "IsSuspicious"
if target not in df.columns:
    raise ValueError("IsSuspicious column not found in the dataset.")

# Use only features you can supply at inference time (from your UI/API)
num_features = [c for c in ["Amount", "Hour", "LateNightTxn"] if c in df.columns]
cat_features = [c for c in ["Country", "Channel", "RiskRating", "TransactionType", "Currency"] if c in df.columns]
train_features = num_features + cat_features
if not train_features:
    raise ValueError("No usable features found. Check your enriched dataset columns.")

X = df[train_features].copy()
y = df[target].astype(int)

# -----------------------------
# Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),  # keep sparse-friendly
])

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(transformers=[
    ("num", numeric_tf, num_features),
    ("cat", categorical_tf, cat_features),
])

# -----------------------------
# Model + randomized search
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", rf),
])

param_distributions = {
    "model__n_estimators": [300, 400, 600, 800],
    "model__max_depth": [None, 8, 12, 16, 20],
    "model__min_samples_split": [2, 4, 6, 8],
    "model__min_samples_leaf": [1, 2, 3, 4],
    "model__max_features": ["sqrt", "log2", 0.5, None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1",      # optimize for minority-class F1
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,
    random_state=42
)

search.fit(X_train, y_train)
best_pipe = search.best_estimator_
print("\nBest params:", search.best_params_)

# -----------------------------
# Evaluate + choose threshold by max F1 on PR curve
# -----------------------------
y_proba = best_pipe.predict_proba(X_test)[:, 1]
prec, rec, thrs = precision_recall_curve(y_test, y_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = int(np.nanargmax(f1s))
best_thr = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5

y_pred = (y_proba >= best_thr).astype(int)

print(f"\n=== Classification Report (threshold = {best_thr:.3f}) ===\n")
cls_report = classification_report(y_test, y_pred, digits=3)
print(cls_report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

roc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)
test_acc = accuracy_score(y_test, y_pred)
print(f"ROC AUC: {roc:.3f} | PR AUC: {pr_auc:.3f} | Accuracy: {test_acc*100:.2f}%")

# Feature count (after preprocessing) for UI
try:
    n_features = len(best_pipe.named_steps["preprocess"].get_feature_names_out())
except Exception:
    n_features = len(train_features)

n_samples = int(len(X_train) + len(X_test))
model_name = type(best_pipe.named_steps["model"]).__name__
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trained_at_human = datetime.now().strftime("%Y-%m-%d %H:%M")

# -----------------------------
# Save artifacts
# -----------------------------
model_path = MODELS_DIR / f"individual_aml_model_{timestamp}.pkl"
joblib.dump(best_pipe, model_path)

# Save rich metadata the UI can read
meta = {
    # For UI cards
    "model_name": model_name,
    "accuracy": float(test_acc),
    "roc_auc": float(roc),
    "pr_auc": float(pr_auc),
    "best_threshold": float(best_thr),
    "n_features": int(n_features),
    "n_samples": int(n_samples),
    "trained_at": trained_at_human,

    # Extra (backward compatible + useful)
    "model_path": str(model_path),
    "threshold": float(best_thr),
    "train_features": train_features,
    "n_features": int(len(train_features)),
    "n_samples": int(len(X_train) + len(X_test)),
    "best_params": search.best_params_,
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "time": timestamp,

    # Nice-to-have for the Model page
    "classification_report": cls_report,
    "confusion_matrix": cm.astype(int).tolist(),
}

with open(MODELS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("\nSaved:")
print(" -", model_path)
print(" -", MODELS_DIR / "training_metadata.json")
