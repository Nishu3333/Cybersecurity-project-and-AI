# enrich_data.py
"""
Enrich transactions with dynamic indicators for AML model training.

Inputs (defaults):
  ../datasets/customer_data.csv
  ../datasets/transaction_data.csv

Output:
  ../datasets/enriched_transaction_data.csv

Fixes included:
  - No row explosion: quarter turnover merge on ['SenderAccount','Quarter']
  - Country case normalization + broader high-risk list
  - Safe Timestamp creation (from Timestamp or Date+Time)
  - Numeric-safe Amount handling
  - Rule-based label with interpretable features
"""

import argparse
import numpy as np
import pandas as pd


# ------------------- Config -------------------

# Uppercased names; add/remove as needed
HIGH_RISK_COUNTRIES = {
    "AFGHANISTAN", "PAKISTAN", "IRAN", "SYRIA",
    "RUSSIA", "RUSSIAN FEDERATION",
    "NORTH KOREA", "MYANMAR", "VENEZUELA",
    "SUDAN", "YEMEN", "ZIMBABWE", "CAMBODIA"
}

# Turnover thresholds (NPR) per quarter
CASH_TURNOVER_BINS = [-0.001, 100_000, 500_000, float("inf")]
CASH_TURNOVER_LABELS = ["low", "medium", "high"]

# Cross-border total thresholds (per sender)
CROSS_BORDER_BINS = [-0.001, 500_000, 1_500_000, float("inf")]
CROSS_BORDER_LABELS = ["low", "medium", "high"]

# Deviation % thresholds (per txn vs sender average)
DEVIATION_BINS = [-0.001, 50, 150, float("inf")]
DEVIATION_LABELS = ["low", "medium", "high"]


# ------------------- Helpers -------------------

def read_inputs(customers_path: str, tx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_c = pd.read_csv(customers_path)
    df_t = pd.read_csv(tx_path)

    needed_tx_cols = {
        "TransactionID", "Amount", "SenderAccount", "ReceiverAccount",
        "Country", "TransactionType", "Currency", "Channel"
    }
    missing = needed_tx_cols.difference(df_t.columns)
    if missing:
        raise ValueError(f"Transactions missing required columns: {missing}")

    if "AccountNumber" not in df_c.columns:
        raise ValueError("Customers file must include 'AccountNumber' column")

    return df_c, df_t


def ensure_timestamp(df_t: pd.DataFrame) -> pd.Series:
    """Prefer existing Timestamp; else build from Date + Time."""
    if "Timestamp" in df_t.columns:
        ts = pd.to_datetime(df_t["Timestamp"], errors="coerce")
    elif {"Date", "Time"}.issubset(df_t.columns):
        ts = pd.to_datetime(
            df_t["Date"].astype(str) + " " + df_t["Time"].astype(str),
            errors="coerce"
        )
    else:
        raise ValueError("Provide either 'Timestamp' or both 'Date' and 'Time' in transactions.")
    if ts.isna().any():
        print(f"[warn] {int(ts.isna().sum())} rows have invalid Timestamp.")
    return ts


def compute_cash_turnover(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly cash turnover per sender; safe merge to avoid row explosion."""
    is_cash = df["TransactionType"].isin(["Cash Withdrawal", "Deposit"]) | df["Channel"].isin(["ATM", "Branch"])
    cash_df = df.loc[is_cash, ["SenderAccount", "Quarter", "Amount"]].copy()

    cash_turnover = (
        cash_df.groupby(["SenderAccount", "Quarter"], as_index=False)["Amount"]
               .sum()
               .rename(columns={"Amount": "CashTurnoverQuarter"})
    )
    cash_turnover["CashTurnover_Risk"] = pd.cut(
        cash_turnover["CashTurnoverQuarter"],
        bins=CASH_TURNOVER_BINS,
        labels=CASH_TURNOVER_LABELS
    )
    return df.merge(cash_turnover, on=["SenderAccount", "Quarter"], how="left")


def compute_cross_border(df: pd.DataFrame) -> pd.DataFrame:
    country_upper = df["Country"].astype(str).str.upper()
    df["CrossBorderWireAmount"] = np.where(
        country_upper.isin(HIGH_RISK_COUNTRIES),
        df["Amount"].astype(float),
        0.0
    )
    cross_border = (
        df.groupby("SenderAccount", as_index=False)["CrossBorderWireAmount"]
          .sum()
          .rename(columns={"CrossBorderWireAmount": "CrossBorderTotal"})
    )
    cross_border["CrossBorder_Risk"] = pd.cut(
        cross_border["CrossBorderTotal"],
        bins=CROSS_BORDER_BINS,
        labels=CROSS_BORDER_LABELS
    )
    return df.merge(cross_border, on="SenderAccount", how="left")


def compute_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df["AvgAmount"] = df.groupby("SenderAccount")["Amount"].transform("mean")
    df["DeviationFromExpectedAmount"] = (
        (df["Amount"] - df["AvgAmount"]).abs() / df["AvgAmount"].replace(0, np.nan) * 100
    ).fillna(0.0).clip(0, 1000)
    df["Deviation_Risk"] = pd.cut(
        df["DeviationFromExpectedAmount"],
        bins=DEVIATION_BINS,
        labels=DEVIATION_LABELS
    )
    return df


def compute_label(row) -> int:
    """Simple interpretable rule score; adjust weights/threshold to taste."""
    score = 0
    if str(row.get("RiskRating", "")).strip().lower() == "high":
        score += 2
    if row.get("CashTurnover_Risk") == "high":
        score += 2
    if row.get("CrossBorder_Risk") == "high":
        score += 2
    if row.get("Deviation_Risk") == "high":
        score += 1
    if row.get("LateNightTxn", 0) == 1:
        score += 1
    return 1 if score >= 6 else 0


# ------------------- Main -------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", default="../datasets/customer_data.csv")
    ap.add_argument("--tx", default="../datasets/transaction_data.csv")
    ap.add_argument("--out", default="../datasets/enriched_transaction_data.csv")
    ap.add_argument("--dedupe", action="store_true",
                    help="If set, drop duplicate TransactionID rows (keep first) as a last-resort safeguard.")
    args = ap.parse_args()

    df_c, df_t = read_inputs(args.customers, args.tx)

    # Merge sender-side attributes
    df = pd.merge(
        df_t, df_c,
        left_on="SenderAccount", right_on="AccountNumber",
        how="left", suffixes=("", "_Sender")
    )

    # Ensure numeric amount + timestamp/time features
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    df["Timestamp"] = ensure_timestamp(df)
    df["Hour"] = df["Timestamp"].dt.hour
    df["LateNightTxn"] = ((df["Hour"] < 5) | (df["Hour"] > 22)).astype(int)
    df["Quarter"] = df["Timestamp"].dt.to_period("Q")

    # Dynamic indicators
    df = compute_cash_turnover(df)
    df = compute_cross_border(df)
    df = compute_deviation(df)

    # Label
    df["IsSuspicious"] = df.apply(compute_label, axis=1)

    # Optional dedupe safety
    if args.dedupe and "TransactionID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["TransactionID"], keep="first")
        print(f"[info] Deduped by TransactionID: {before} -> {len(df)} rows")

    # Save
    df.to_csv(args.out, index=False)

    # Summary
    uniq_tx = df["TransactionID"].is_unique if "TransactionID" in df.columns else None
    lbl = df["IsSuspicious"].value_counts(dropna=False)
    lbl_pct = (lbl / len(df) * 100).round(2)
    print(f"Saved: {args.out}")
    print(f"Rows: {len(df)} | Cols: {df.shape[1]} | TransactionID unique: {uniq_tx}")
    print("IsSuspicious distribution:\n", pd.concat([lbl, lbl_pct.rename("percent")], axis=1))


if __name__ == "__main__":
    main()
