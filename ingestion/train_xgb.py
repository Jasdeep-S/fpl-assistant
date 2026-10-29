import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from xgboost import XGBRegressor

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
df = pd.read_sql("SELECT * FROM features", engine)
df = df.dropna(subset=["avg_pts_last5"])

df["was_home"] = df["was_home"].astype(int)
df["pos"] = df["position"]            # keep the original for per-position breakdowns
df = pd.get_dummies(df, columns=["position"])

FEATURES = [c for c in df.columns if c.startswith("avg_") or c.startswith("position_")]
FEATURES += ["was_home", "price", "games_in_window"]
print(len(FEATURES), "features:", FEATURES)

train = df[df["season"] != "2025-26"]
test = df[df["season"] == "2025-26"].copy()

# Model that predicts expected points (the mean)
model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                     objective="reg:squarederror")
model.fit(train[FEATURES], train["target_points"])
test["pred"] = model.predict(test[FEATURES])

def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())

# Regular starters only (60+ minutes recently)
r = test[test["avg_minutes_last5"] >= 60].copy()
r["err_base"] = (r.target_points - r.avg_pts_last5).abs()
r["err_model"] = (r.target_points - r.pred).abs()

print("\n--- Regular starters ---")
print(f"Baseline  MAE {r.err_base.mean():.3f}  RMSE {rmse(r.target_points, r.avg_pts_last5):.3f}")
print(f"XGBoost   MAE {r.err_model.mean():.3f}  RMSE {rmse(r.target_points, r.pred):.3f}")

print("\nMAE by position (regulars):")
print(r.groupby("pos")[["err_base", "err_model"]].mean().round(3))

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop features:\n", imp.head(8).round(3))

# Save the model for the prediction script
joblib.dump(model, "model.joblib")
joblib.dump(FEATURES, "features.joblib")
print("\nSaved model.joblib and features.joblib")