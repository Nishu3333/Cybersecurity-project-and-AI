# Cybersecurity-project-and-AI
AML Transaction Monitoring System (Academic Prototype)
An end-to-end Anti-Money Laundering (AML) transaction monitoring prototype that demonstrates how a bank could score transactions for risk, surface alerts, and view simple analytics—all on a local machine.

The system has three parts:
1. Model training (scripts/train_model.py)
  - Loads datasets/enriched_transaction_data.csv (or data/enriched_transaction_data.csv).
  - Builds a preprocessing pipeline (imputation, scaling, one-hot encoding) and trains a Random Forest with randomized search + cross-validation.
  - Picks an operating threshold by maximizing F1 on a hold-out set, and saves artifacts:
    models/individual_aml_model_YYYYMMDD_HHMMSS.pkl
    models/training_metadata.json (metrics, threshold, feature list, training date, best params, classification report, confusion matrix).

2. Scoring API (backend/api.py) – Flask
   - On start, auto-loads the latest model from models/training_metadata.json.
   - Exposes endpoints to score transactions, view model status, reload model, and serve dashboard analytics (recent transactions, basic aggregates, country volume).
   - Accepts a CSV upload to bulk-score rows.

3. Web UI (frontend/app.py) – Streamlit
   - Dashboard: API health, model status, counts, average risk score, recent transactions, charts.
   - Model Training: shows current model status (type, samples, threshold, ROC-AUC/PR-AUC), classification report, confusion matrix.
   - Transaction Analysis: score a single transaction and see reasons.
   - Data Upload: upload a CSV to bulk-score and populate the dashboard.
