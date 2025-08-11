# frontend/app.py
import os
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

# ---------------- Config helpers ----------------
def api_url() -> str:
    try:
        return st.secrets["API_URL"].rstrip("/")
    except Exception:
        try:
            from utils.config import Config  # optional .env support
            return f"http://{Config.API_HOST}:{Config.API_PORT}"
        except Exception:
            return os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

API = api_url()

def get_json(path: str, default=None, timeout=12):
    try:
        r = requests.get(f"{API}{path}", timeout=timeout)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return default

def post_json(path: str, payload=None, files=None, timeout=300):
    try:
        if files:
            r = requests.post(f"{API}{path}", files=files, timeout=timeout)
        else:
            r = requests.post(f"{API}{path}", json=payload or {}, timeout=timeout)
        if r.ok:
            return r.json(), True
        return (r.json() if r.headers.get("content-type","").startswith("application/json") else {"error": r.text}), False
    except Exception as e:
        return {"error": str(e)}, False

def pill(text, color="#e9f2ff"):
    st.markdown(
        f"""
        <div style="background:{color};border-radius:10px;padding:8px 12px;
                    border:1px solid rgba(0,0,0,0.06);display:inline-block;">
            <span style="font-weight:600;color:#203a59">{text}</span>
        </div>
        """, unsafe_allow_html=True
    )

# ---------------- Sidebar ----------------
st.set_page_config(page_title="AML Monitoring System", layout="wide")

with st.sidebar:
    st.markdown("## 🏛️ AML Monitoring System")
    health = get_json("/health", default={})
    model_info = get_json("/api/model/info", default={})

    if health and health.get("status") == "ok":
        st.success("API: Online", icon="✅")
    else:
        st.error("API: Offline", icon="⚠️")

    st.caption(f"API URL: [{API}]({API})")

    if model_info and model_info.get("ml_loaded"):
        st.success("Model: ML Active", icon="🧠")
        st.caption(f"Threshold: {model_info.get('threshold', 0.5):.3f}")
        if model_info.get("roc_auc") is not None:
            st.caption(f"ROC AUC: {model_info['roc_auc']:.3f} | PR AUC: {model_info.get('pr_auc', 0):.3f}")
    else:
        st.warning("Model: Not trained", icon="🧪")

    st.markdown("---")
    page = st.selectbox(
        "Navigate to:",
        ["Dashboard", "Model Training", "Transaction Analysis", "Data Upload", "About"],
        index=0
    )

# ---------------- Pages ----------------
def page_dashboard():
    st.markdown("## 🏛️ AML Transaction Monitoring Dashboard")
    st.caption("Real-time monitoring of suspicious transactions and AML compliance for Kathmandu banks.")

    metrics = get_json(
        "/api/dashboard/metrics",
        default={"total_transactions":0,"suspicious_today":0,"avg_risk_score":None,"active_alerts":0}
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", metrics.get("total_transactions", 0))
    c2.metric("Suspicious (today)", metrics.get("suspicious_today", 0))
    c3.metric("Avg Risk Score", "—" if metrics.get("avg_risk_score") is None else f"{metrics['avg_risk_score']:.3f}")
    c4.metric("Active Alerts", metrics.get("active_alerts", 0))

    st.markdown("### 📊 Recent Transactions")
    rec = get_json("/api/transactions/recent?limit=25", default=[])
    if not rec:
        st.info("No transactions yet. Score one in **Transaction Analysis** or upload data in **Data Upload**.")
    else:
        df = pd.DataFrame(rec)
        st.dataframe(df, use_container_width=True, height=320)

    st.markdown("### 💵 Transaction Amount Distribution")
    cont1 = st.container(border=True)
    cont2 = st.container(border=True)
    with cont1:
        if rec:
            df = pd.DataFrame(rec)
            if "amount" in df.columns:
                st.bar_chart(df["amount"], use_container_width=True)
            else:
                st.info("No amount column to plot.")
        else:
            st.info("No data to plot yet.")
    with cont2:
        st.markdown("### 🌍 Transaction Volume by Country")
        cdata = get_json("/api/analytics/country_volume", default=[])
        if cdata:
            cdf = pd.DataFrame(cdata)
            st.bar_chart(cdf.set_index("country")["count"], use_container_width=True)
        else:
            st.info("No country stats available.")

    # ---------- Risk Analysis pies ----------
    st.markdown("### 🚨 Risk Analysis")
    col1, col2 = st.columns(2)

    with col1:
        rdata = get_json("/api/analytics/risk_rating", default=[])
        if rdata:
            import matplotlib.pyplot as plt
            labels = [d["risk"] for d in rdata]
            sizes  = [d["count"] for d in rdata]
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct=lambda pct: f"{pct:.0f}%", startangle=90
            )
            ax.axis("equal")
            ax.set_title("Customer Risk Rating Distribution")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No risk-rating data yet. Upload or score some transactions.")

    with col2:
        pdata = get_json("/api/analytics/prediction_breakdown", default=[])
        if pdata:
            import matplotlib.pyplot as plt
            labels = [d["label"] for d in pdata]
            sizes  = [d["count"] for d in pdata]
            fig2, ax2 = plt.subplots()
            wedges, texts, autotexts = ax2.pie(
                sizes, labels=labels, autopct=lambda pct: f"{pct:.0f}%", startangle=90
            )
            ax2.axis("equal")
            ax2.set_title("Transaction Classification")
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info("No classification data yet. Upload or score some transactions.")

def page_model_training():
    st.markdown("## 🧠 Model Training")
    right = st.columns([1,1,1,1,2])[4]
    with right:
        pill("🎯 Target Accuracy: 80–90%")

    st.caption("Designed to achieve realistic performance with your dataset")
    st.write("")

    cta = st.columns([1,3,1])[1]
    with cta:
        if st.button("🎆 Start Model Training", type="primary", use_container_width=True):
            with st.status("Training started… this can take a minute ⏳", expanded=True) as status:
                out, ok = post_json("/api/train", payload={})  # optional route (only if you implement it)
                if ok:
                    st.write("• Grid search + cross-validation complete")
                    if out.get("best_params"):
                        st.write(f"• Best params: `{out['best_params']}`")
                    if out.get("roc_auc") is not None:
                        st.write(f"• ROC AUC: {out['roc_auc']:.3f} | PR AUC: {out.get('pr_auc', 0):.3f}")
                    st.write(f"• Threshold: {out.get('threshold', 0.5):.3f}")
                    status.update(label="Training finished. Model saved and hot-reloaded ✅", state="complete")
                    st.toast("Model trained and loaded", icon="✅")
                else:
                    status.update(label=f"Training failed ❌: {out.get('error','/api/train not available')}", state="error")
            st.rerun()

    st.markdown("### Current Model Status")
    info = get_json("/api/model/info", default={})

    # Fallbacks if backend returns the older "metrics" container
    if info and info.get("ml_loaded"):
        info.setdefault("features", len(info.get("train_features", [])))
        if not info.get("samples"):
            m = info.get("metrics", {}) if isinstance(info.get("metrics"), dict) else {}
            info["samples"] = int(m.get("train_samples", 0)) + int(m.get("test_samples", 0))
        if "roc_auc" not in info and isinstance(info.get("metrics"), dict):
            info["roc_auc"] = info["metrics"].get("roc_auc")
            info["pr_auc"]  = info["metrics"].get("pr_auc")

    if info.get("ml_loaded"):
        pill("✅ Model is trained and ready for predictions", "#e8f5e9")
        st.write("")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Model Type", info.get("model_type", "—"))
        r1c2.metric("Features Used", info.get("features", "—"))
        r1c3.metric("Training Samples", f"{info.get('samples','—'):,}" if info.get("samples") else "—")
        r1c4.metric("Threshold", f"{info.get('threshold', 0.5):.3f}")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        trained_at = info.get("trained_at")
        if trained_at:
            try:
                dt = datetime.fromisoformat(trained_at)
                r2c1.metric("Training Date", dt.strftime("%Y-%m-%d %H:%M"))
            except Exception:
                r2c1.metric("Training Date", trained_at)
        else:
            r2c1.metric("Training Date", "—")
        if info.get("roc_auc") is not None:
            r2c2.metric("ROC AUC", f"{info['roc_auc']:.3f}")
        if info.get("pr_auc") is not None:
            r2c3.metric("PR AUC", f"{info['pr_auc']:.3f}")
        r2c4.metric("Version", info.get("model_version", "1"))

        if info.get("classification_report"):
            st.markdown("#### Classification Report")
            st.code(info["classification_report"])

        if info.get("confusion_matrix"):
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                cm = np.array(info["confusion_matrix"])
                fig, ax = plt.subplots()
                im = ax.imshow(cm, interpolation="nearest")
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=[0,1], yticks=[0,1], xlabel='Predicted', ylabel='True')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha="center", va="center")
                st.pyplot(fig)
            except Exception:
                pass
    else:
        pill("⚠️ Model: Not trained", "#fff9e6")
        st.caption("Click **Start Model Training** (or run scripts/train_model.py) to train and deploy a model.")

def page_transaction_analysis():
    st.markdown("## 📈 Transaction Analysis & Prediction")
    st.caption("Analyze individual transactions and predict AML risk using trained models.")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (NPR)", min_value=0.0, step=1000.0, value=1200000.0, format="%.2f")
        country = st.selectbox("Country", ["Nepal", "India", "China", "Russia", "USA"], index=0)
        ttype = st.selectbox("Transaction Type", ["Transfer", "Deposit", "Cash Withdrawal"], index=0)
    with col2:
        channel = st.selectbox("Channel", ["Online", "ATM", "Branch"], index=0)
        risk = st.selectbox("Customer Risk Rating", ["Low", "Medium", "High"], index=2)
        hour = st.slider("Hour (0–23)", min_value=0, max_value=23, value=23)

    payload = {
        "amount": amount,
        "country": country,
        "transaction_type": ttype,
        "channel": channel,
        "risk_rating": risk,
        "hour": hour
    }

    if st.button("Score Transaction", type="primary"):
        out, ok = post_json("/api/score", payload=payload, timeout=20)  # alias served by backend
        if ok:
            score = out.get("risk_score", 0.0)
            pred = out.get("prediction", 0)
            colA, colB = st.columns([1,4])
            with colA:
                st.metric("Risk Score", f"{score:.3f}")
            with colB:
                st.markdown("#### Prediction")
                if pred == 1:
                    st.error("Suspicious", icon="🚨")
                else:
                    st.success("Normal", icon="✅")
            st.markdown("#### Reasons:")
            st.json(out.get("reasons", []))
        else:
            st.error(f"Scoring failed: {out.get('error','unknown')}")

def page_data_upload():
    st.markdown("## ⬆️ Data Management")
    st.caption("Upload and manage transaction data for training and analysis.")
    st.markdown("### Upload Transaction Data")
    f = st.file_uploader("Choose a CSV file", type=["csv"])
    if f is not None:
        if st.button("Upload & Score"):
            with st.status("Uploading…", expanded=True) as status:
                files = {"file": (f.name or "upload.csv", f.getvalue(), "text/csv")}
                out, ok = post_json("/api/upload", files=files)
                if ok:
                    status.update(label="Upload complete ✅", state="complete")
                    st.success(f"Uploaded and scored {out.get('added', 0)} rows.")
                else:
                    status.update(label="Upload failed ❌", state="error")
                    st.error(out.get("error","Unknown error"))

    st.markdown("### Current Data Status")
    metrics = get_json(
        "/api/dashboard/metrics",
        default={"total_transactions":0,"suspicious_today":0,"avg_risk_score":None}
    )
    t1, t2, t3 = st.columns(3)
    t1.metric("Total Transactions", metrics.get("total_transactions", 0))
    t2.metric("Average Amount", "—")  # extend API if you want this real
    t3.metric("Suspicious Transactions", metrics.get("suspicious_today", 0))
    st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

def page_about():
    st.markdown("## ℹ️ About")
    st.markdown("### Key Features")
    st.markdown("#### 😄 Machine Learning Detection")
    st.markdown("""
- **Bias-Reduced Models**: Trained to achieve realistic accuracy (~80–90%) avoiding overfitting  
- **Multiple Algorithms**: Random Forest, Logistic Regression, Gradient Boosting, XGBoost  
- **SMOTE Balancing**: Handles imbalanced datasets effectively  
- **Feature Engineering**: Advanced feature selection and engineering
    """)
    st.markdown("#### 📊 Real-time Monitoring")
    st.markdown("""
- **Live Dashboard**: Real-time transaction monitoring  
- **Risk Scoring**: Automated risk assessment for each transaction  
- **Alert System**: Immediate alerts for high-risk transactions  
- **Compliance Reporting**: Automated AML compliance reports
    """)
    st.markdown("#### 🔎 Advanced Analysis")
    st.markdown("""
- **Transaction Profiling**: Detailed analysis of transaction patterns  
- **Customer Risk Assessment**: Comprehensive customer risk profiling  
- **Cross-border Monitoring**: International transactions focus  
- **PEP Detection**: Politically Exposed Person identification
    """)
    st.markdown("### Technical Specifications")
    st.markdown("#### Backend Architecture")
    st.markdown("""
- **Language**: Python 3.11  
- **ML Libraries**: scikit-learn, XGBoost, imbalanced-learn  
- **Data Processing**: pandas, numpy
    """)
    st.markdown("### Data Requirements")
    st.markdown("""
- **Transaction Features**: Amount, Country, Channel, Time, Customer data  
- **Risk Indicators**: PEP status, High-risk countries, Late-night transactions  
- **Historical Data**: For model training and pattern analysis
    """)
    st.markdown("### Compliance Features")
    st.markdown("""
- **KYC**: Customer verification and monitoring  
- **SAR**: Automated report generation  
- **Continuous Monitoring**: Surveillance of transactions  
- **Risk-Based**: Decisions based on risk profiles
    """)
    st.markdown("### Audit Trail & Academic Purpose")
    st.markdown("""
- **Complete Logging** of decisions  
- **Investigation Notes** and historical analysis  
- **Scalable Architecture** designed for deployment
    """)
    st.caption("Version: 1.0.0  •  Academic Institution: Kathmandu University  •  Project Type: AML Transaction Monitoring Research")

# Router
if page == "Dashboard":
    page_dashboard()
elif page == "Model Training":
    page_model_training()
elif page == "Transaction Analysis":
    page_transaction_analysis()
elif page == "Data Upload":
    page_data_upload()
else:
    page_about()
