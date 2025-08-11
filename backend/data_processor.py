import pandas as pd
import numpy as np

HIGH_RISK = {"AFGHANISTAN","PAKISTAN","IRAN","SYRIA","RUSSIA","RUSSIAN FEDERATION",
             "NORTH KOREA","MYANMAR","VENEZUELA","SUDAN","YEMEN","ZIMBABWE","CAMBODIA"}

class DataProcessor:
    def ensure_timestamp(self, df: pd.DataFrame) -> pd.Series:
        if "Timestamp" in df.columns:
            return pd.to_datetime(df["Timestamp"], errors="coerce")
        if {"Date","Time"}.issubset(df.columns):
            return pd.to_datetime(df["Date"].astype(str)+" "+df["Time"].astype(str), errors="coerce")
        # fallback: now
        return pd.to_datetime(pd.Timestamp.utcnow()).repeat(len(df))

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ensure required fields exist
        for c in ["Amount","Country","TransactionType","Channel","RiskRating"]:
            if c not in df.columns: df[c] = np.nan

        # numeric amount
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

        # --- IMPORTANT: honor provided Hour if present ---
        hour_provided = "Hour" in df.columns
        if hour_provided:
            df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)

        # build/normalize Timestamp
        df["Timestamp"] = self.ensure_timestamp(df)
        # if no valid timestamp but Hour is provided, synthesize one for today at that Hour
        if df["Timestamp"].isna().any() and hour_provided:
            today = pd.Timestamp.now().normalize()
            df.loc[df["Timestamp"].isna(), "Timestamp"] = (
                today + pd.to_timedelta(df.loc[df["Timestamp"].isna(), "Hour"], unit="h")
            )

        # derive Hour only if it wasn't provided
        if not hour_provided:
            df["Hour"] = df["Timestamp"].dt.hour

        # late night flag always from the Hour column
        df["LateNightTxn"] = ((df["Hour"] < 5) | (df["Hour"] > 22)).astype(int)

        # country helpers
        df["CountryUpper"] = df["Country"].astype(str).str.upper()
        df["IsHighRiskCountry"] = df["CountryUpper"].isin(HIGH_RISK).astype(int)

        return df
