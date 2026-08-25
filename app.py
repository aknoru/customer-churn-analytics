from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from src.features import build_features

st.set_page_config(page_title="Customer Churn Analytics", layout="wide")
st.title("Customer Churn Analytics & Prediction")
st.caption("Customer segmentation, churn-risk analysis and model evaluation")

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"/"customers.csv"
MODEL = ROOT/"models"/"churn_model.joblib"
METRICS = ROOT/"models"/"metrics.json"

if not DATA.exists():
    st.error("Run python scripts/generate_data.py first.")
    st.stop()
if not MODEL.exists():
    st.error("Run python scripts/train.py first.")
    st.stop()

df = build_features(pd.read_csv(DATA))
model = joblib.load(MODEL)
X = df.drop(columns=["churn","customer_id"])
df["churn_probability"] = model.predict_proba(X)[:,1]
df["risk_band"] = pd.cut(
    df["churn_probability"], bins=[-.01,.33,.66,1.01],
    labels=["Low","Medium","High"]
)

a,b,c,d = st.columns(4)
a.metric("Customers",f"{len(df):,}")
b.metric("Observed churn",f"{df.churn.mean():.1%}")
c.metric("High-risk customers",f"{(df.risk_band=='High').sum():,}")
d.metric("Average monthly charge",f"${df.monthly_charges.mean():.2f}")

left,right=st.columns(2)
with left:
    seg=df.groupby("contract_type",as_index=False).agg(
        churn_rate=("churn","mean"),customers=("customer_id","count"))
    fig=px.bar(seg,x="contract_type",y="churn_rate",title="Churn Rate by Contract Type")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig,use_container_width=True)

with right:
    sample=df.sample(min(1000,len(df)),random_state=42)
    fig=px.scatter(sample,x="tenure_months",y="monthly_charges",
                   size="support_tickets",color="risk_band",
                   hover_data=["customer_id","satisfaction_score"],
                   title="Customer Risk Distribution")
    st.plotly_chart(fig,use_container_width=True)

st.subheader("Model Evaluation")
if METRICS.exists():
    metrics=json.loads(METRICS.read_text())
    st.json({k:round(v,4) for k,v in metrics.items()})

st.subheader("High-Risk Customer Analysis")
st.dataframe(
    df[df.risk_band=="High"].sort_values("churn_probability",ascending=False)[
        ["customer_id","tenure_months","monthly_charges","support_tickets",
         "satisfaction_score","contract_type","churn_probability","risk_band"]
    ].head(100),
    use_container_width=True
)
