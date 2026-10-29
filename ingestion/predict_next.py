import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
model = joblib.load("model.joblib")
FEATURES = joblib.load("features.joblib")

hist = pd.read_sql("SELECT * FROM raw_player_history", engine)
players = pd.read_sql(
    "SELECT id, web_name, element_type, team, now_cost, news, "
    "chance_of_playing_next_round FROM raw_players", engine)
gws = pd.read_sql("SELECT id, is_next FROM raw_gameweeks", engine)
fixtures = pd.read_sql("SELECT event, team_h, team_a FROM raw_fixtures", engine)

next_gw = int(gws.loc[gws["is_next"], "id"].iloc[0])
print(f"Predicting gameweek {next_gw}")

# Rolling features from each player's most recent games
hist["ict_index"] = hist["ict_index"].astype(float)
hist = hist.sort_values("kickoff_time")
last5 = hist.groupby("player_id").tail(5)
last3 = hist.groupby("player_id").tail(3)

f = last5.groupby("player_id").agg(
    avg_pts_last5=("total_points", "mean"),
    avg_minutes_last5=("minutes", "mean"),
    avg_goals_last5=("goals_scored", "mean"),
    avg_assists_last5=("assists", "mean"),
    avg_bonus_last5=("bonus", "mean"),
    avg_bps_last5=("bps", "mean"),
    avg_ict_last5=("ict_index", "mean"),
    games_in_window=("total_points", "count"),
)
f["avg_pts_last3"] = last3.groupby("player_id")["total_points"].mean()
f = f.reset_index()

# Who plays next gameweek, and home or away
fx = fixtures[fixtures["event"] == next_gw]
games = pd.concat([
    fx.assign(team=fx["team_h"], was_home=1)[["team", "was_home"]],
    fx.assign(team=fx["team_a"], was_home=0)[["team", "was_home"]],
])

df = (players.rename(columns={"id": "player_id"})
      .merge(f, on="player_id")
      .merge(games, on="team"))

df["pos"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
df["position"] = df["pos"]          # copy that get_dummies will turn into columns
df["price"] = df["now_cost"]
df = pd.get_dummies(df, columns=["position"])
X = df.reindex(columns=FEATURES, fill_value=0).astype(float)

df["predicted_points"] = model.predict(X)

# Availability adjustment: NULL means no injury flag, so treat as 100% available
chance = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100) / 100
df["predicted_points"] = df["predicted_points"] * chance

# A player with a double gameweek has two rows, so add them up
out = (df.groupby(["player_id", "web_name", "pos", "team", "price", "news"], dropna=False)
         ["predicted_points"].sum().reset_index()
         .sort_values("predicted_points", ascending=False))
out["gameweek"] = next_gw
out["predicted_points"] = out["predicted_points"].round(3)

print(out[["player_id", "web_name", "pos", "news", "predicted_points"]]
      .head(20).to_string(index=False))

# Save so the optimizer and chatbot can read predictions from the database
out.to_sql("predictions", engine, if_exists="replace", index=False)
print(f"\nSaved {len(out)} predictions for GW{next_gw} to table 'predictions'")