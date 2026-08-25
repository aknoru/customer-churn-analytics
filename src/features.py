import pandas as pd

TARGET = "churn"

def validate(df):
    required = {"customer_id","tenure_months","monthly_charges","support_tickets",
                "contract_type","payment_method","internet_service",
                "satisfaction_score",TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

def build_features(df):
    validate(df)
    out = df.copy()
    out["charges_per_tenure"] = out["monthly_charges"] / out["tenure_months"].clip(lower=1)
    out["support_intensity"] = out["support_tickets"] / out["tenure_months"].clip(lower=1)
    return out
