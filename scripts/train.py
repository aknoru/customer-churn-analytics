from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import build_features
from src.model import make_pipeline, evaluate, save_model

ROOT = Path(__file__).resolve().parents[1]

def main():
    path = ROOT / "data" / "customers.csv"
    if not path.exists():
        raise SystemExit("Run python scripts/generate_data.py first.")
    df = build_features(pd.read_csv(path))
    X = df.drop(columns=["churn","customer_id"])
    y = df["churn"]
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=.2,random_state=42,stratify=y
    )
    model = make_pipeline()
    model.fit(X_train,y_train)
    metrics = evaluate(model,X_test,y_test)
    save_model(model,ROOT/"models"/"churn_model.joblib")
    (ROOT/"models"/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__ == "__main__":
    main()
