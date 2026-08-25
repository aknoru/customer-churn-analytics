import pandas as pd
from src.features import build_features
from src.model import make_pipeline

def test_features():
    df = pd.DataFrame([{
        "customer_id":"C1","tenure_months":10,"monthly_charges":100,
        "support_tickets":2,"contract_type":"Month-to-month",
        "payment_method":"Credit card","internet_service":"DSL",
        "satisfaction_score":4,"churn":0
    }])
    out = build_features(df)
    assert "charges_per_tenure" in out
    assert "support_intensity" in out

def test_pipeline():
    rows=[]
    for i in range(20):
        rows.append({
            "tenure_months":5+i,"monthly_charges":50+i,
            "support_tickets":i%4,
            "contract_type":"Month-to-month" if i%2 else "One year",
            "payment_method":"Credit card","internet_service":"DSL",
            "satisfaction_score":2+(i%4)/2,
            "charges_per_tenure":(50+i)/(5+i),
            "support_intensity":(i%4)/(5+i)
        })
    X=pd.DataFrame(rows)
    y=pd.Series([i%2 for i in range(20)])
    model=make_pipeline()
    model.fit(X,y)
    assert len(model.predict(X)) == 20
