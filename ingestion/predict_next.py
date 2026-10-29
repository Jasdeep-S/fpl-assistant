import os
import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
model = joblib.load("model.joblib")
FEATURES = joblib.load("features.joblib")

# Hand-picked heuristic, not learned from data: +/-5% per FDR step away from 3.
# FDR 1 (easiest) -> +10%, FDR 5 (hardest) -> -10%.
FDR_WEIGHT = 0.05

hist = pd.read_sql("SELECT * FROM raw_player_history", engine)
players = pd.read_sql(
    "SELECT id, web_name, element_type, team, now_cost, news, "
    "chance_of_playing_next_round FROM raw_players", engine)
gws = pd.read_sql("SELECT id, is_next FROM raw_gameweeks", engine)
fixtures = pd.read_sql(
    "SELECT event, team_h, team_a, team_h_difficulty, team_a_difficulty "
    "FROM raw_fixtures", engine)
teams = pd.read_sql("SELECT id, name, short_name FROM raw_teams", engine)
team_name = dict(zip(teams["id"], teams["name"]))
team_short = dict(zip(teams["id"], teams["short_name"]))

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

# Next gameweek's games: home/away, opponent, and FPL's difficulty rating
fx = fixtures[fixtures["event"] == next_gw]
home = pd.DataFrame({"team": fx["team_h"], "opp": fx["team_a"],
                     "was_home": 1, "fdr": fx["team_h_difficulty"]})
away = pd.DataFrame({"team": fx["team_a"], "opp": fx["team_h"],
                     "was_home": 0, "fdr": fx["team_a_difficulty"]})
games = pd.concat([home, away], ignore_index=True)
games["fdr"] = games["fdr"].fillna(3)
games["fixture"] = (games["opp"].map(team_short)
                    + games["was_home"].map({1: " (H)", 0: " (A)"}))
games = games.drop(columns="opp")

df = (players.rename(columns={"id": "player_id"})
      .merge(f, on="player_id")
      .merge(games, on="team"))

df["pos"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"})
df["position"] = df["pos"]          # copy that get_dummies will turn into columns
df["price"] = df["now_cost"]
df = pd.get_dummies(df, columns=["position"])
X = df.reindex(columns=FEATURES, fill_value=0).astype(float)

# The model itself does not use fixture difficulty; it is applied afterwards
df["model_points"] = model.predict(X)

# NULL availability means no injury flag, so treat as 100% available
chance = pd.to_numeric(df["chance_of_playing_next_round"], errors="coerce").fillna(100) / 100
fixture_factor = 1 + FDR_WEIGHT * (3 - df["fdr"])
df["model_points"] = df["model_points"] * chance
df["predicted_points"] = df["model_points"] * fixture_factor
df["team_name"] = df["team"].map(team_name)

# A player with a double gameweek has two rows, so add them up
out = (df.groupby(["player_id", "web_name", "pos", "team", "team_name", "price", "news"],
                  dropna=False)
         .agg(predicted_points=("predicted_points", "sum"),
              model_points=("model_points", "sum"),
              fdr=("fdr", "mean"),
              fixture=("fixture", lambda s: ", ".join(s)))
         .reset_index()
         .sort_values("predicted_points", ascending=False))
out["gameweek"] = next_gw
out[["predicted_points", "model_points", "fdr"]] = \
    out[["predicted_points", "model_points", "fdr"]].round(3)

print(out[["web_name", "pos", "fixture", "fdr", "model_points", "predicted_points"]]
      .head(20).to_string(index=False))

out.to_sql("predictions", engine, if_exists="replace", index=False)
print(f"\nSaved {len(out)} predictions for GW{next_gw} to table 'predictions'")