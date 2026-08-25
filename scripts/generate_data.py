from pathlib import Path
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    rng = np.random.default_rng(42)
    rows = 2500
    tenure = rng.integers(1, 73, rows)
    charges = np.round(rng.uniform(25, 150, rows), 2)
    tickets = rng.poisson(2.2, rows)
    satisfaction = np.clip(np.round(rng.normal(3.4, .9, rows), 1), 1, 5)

    contract = rng.choice(["Month-to-month","One year","Two year"], rows, p=[.55,.28,.17])
    payment = rng.choice(["Electronic check","Bank transfer","Credit card","Mailed check"], rows)
    internet = rng.choice(["Fiber optic","DSL","No internet"], rows, p=[.48,.38,.14])

    effect = {"Month-to-month":.9, "One year":-.3, "Two year":-.9}
    score = (-1.8 + .015*charges - .025*tenure + .18*tickets
             - .55*satisfaction
             + np.array([effect[x] for x in contract])
             + np.where(internet == "Fiber optic", .25, 0)
             + rng.normal(0,.45,rows))
    churn = rng.binomial(1, sigmoid(score))

    df = pd.DataFrame({
        "customer_id":[f"CUST-{i:05d}" for i in range(1,rows+1)],
        "tenure_months":tenure,
        "monthly_charges":charges,
        "support_tickets":tickets,
        "contract_type":contract,
        "payment_method":payment,
        "internet_service":internet,
        "satisfaction_score":satisfaction,
        "churn":churn
    })
    out = Path(__file__).resolve().parents[1] / "data" / "customers.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic customers to {out}")

if __name__ == "__main__":
    main()
