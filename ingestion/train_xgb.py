import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from xgboost import XGBRegressor

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
df = pd.read_sql("SELECT * FROM features", engine)
df = df.dropna(subset=["avg_pts_last5"])

df["was_home"] = df["was_home"].astype(int)
df = pd.get_dummies(df, columns=["position"])

FEATURES = [c for c in df.columns if c.startswith("avg_") or c.startswith("position_")]
FEATURES += ["was_home", "price", "games_in_window"]

train = df[df["season"] != "2025-26"]
test = df[df["season"] == "2025-26"]

model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                     objective="reg:absoluteerror")
model.fit(train[FEATURES], train["target_points"])

test = test.assign(pred=model.predict(test[FEATURES]))
test["err"] = (test["target_points"] - test["pred"]).abs()
base_err = (test["target_points"] - test["avg_pts_last5"]).abs()

print(f"XGBoost MAE (all rows):        {test['err'].mean():.3f}")
print(f"Baseline MAE (all rows):       {base_err.mean():.3f}")

reg = test["avg_minutes_last5"] >= 60
print(f"XGBoost MAE (regulars):        {test.loc[reg, 'err'].mean():.3f}")
print(f"Baseline MAE (regulars):       {base_err[reg].mean():.3f}")

import numpy as np

# Second model: predicts the MEAN (expected points)
model_mse = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         objective="reg:squarederror")
model_mse.fit(train[FEATURES], train["target_points"])
test["pred_mse"] = model_mse.predict(test[FEATURES])

def rmse(a, b): return np.sqrt(((a - b) ** 2).mean())

print("\n--- Regular starters ---")
r = test[reg]
print(f"Baseline        MAE {(r.target_points - r.avg_pts_last5).abs().mean():.3f}  RMSE {rmse(r.target_points, r.avg_pts_last5):.3f}")
print(f"XGB (abs-error) MAE {(r.target_points - r.pred).abs().mean():.3f}  RMSE {rmse(r.target_points, r.pred):.3f}")
print(f"XGB (sq-error)  MAE {(r.target_points - r.pred_mse).abs().mean():.3f}  RMSE {rmse(r.target_points, r.pred_mse):.3f}")

# Which features matter?
imp = pd.Series(model_mse.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop features:\n", imp.head(8).round(3))