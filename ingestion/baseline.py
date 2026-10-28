import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

df = pd.read_sql("SELECT * FROM features", engine)

# Only keep rows where we have history, and where the player actually has a recent average
df = df.dropna(subset=["avg_pts_last5"])

# Time-based split: test on the latest season only
test = df[df["season"] == "2025-26"]

# Baseline: predict next game's points = average of last 5
mae = (test["target_points"] - test["avg_pts_last5"]).abs().mean()
print(f"Baseline MAE on 2025-26: {mae:.3f}")

# Break it down by position
test = test.assign(err=(test["target_points"] - test["avg_pts_last5"]).abs())
print(test.groupby("position")["err"].mean().round(3))

# Context: how good is a "dumb" guess?
mean_guess = df[df["season"] != "2025-26"]["target_points"].mean()
dumb_mae = (test["target_points"] - mean_guess).abs().mean()
print(f"\nPredict-the-overall-average MAE: {dumb_mae:.3f}")

# Regular starters only (averaging 60+ minutes recently)
regulars = test[test["avg_minutes_last5"] >= 60]
reg_mae = regulars["err"].mean()
print(f"Baseline MAE, regular starters only: {reg_mae:.3f} ({len(regulars)} rows)")
print(regulars.groupby("position")["err"].mean().round(3))