import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

if len(sys.argv) < 2:
    sys.exit("Usage: python ingestion/suggest_transfers.py YOUR_TEAM_ID")
team_id = int(sys.argv[1])

preds = pd.read_sql("SELECT * FROM predictions", engine)
next_gw = int(preds["gameweek"].iloc[0])
last_gw = max(next_gw - 1, 1)

url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_gw}/picks/"
resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
if resp.status_code == 404:
    sys.exit(f"No picks found for team {team_id} in GW{last_gw}. Check the ID.")
resp.raise_for_status()
data = resp.json()

squad_ids = [p["element"] for p in data["picks"]]
bank = data["entry_history"]["bank"]  # in tenths of a million, e.g. 5 = £0.5m

squad = preds[preds["player_id"].isin(squad_ids)].reset_index(drop=True)
print(f"Matched {len(squad)} of {len(squad_ids)} squad players. Bank: £{bank / 10:.1f}m")

POSITIONS = ["GK", "DEF", "MID", "FWD"]

def best_xi_points(df):
    """Best legal starting XI total, trying every valid formation."""
    by = {p: sorted(df.loc[df["pos"] == p, "predicted_points"], reverse=True)
          for p in POSITIONS}
    if not by["GK"]:
        return 0.0
    best = 0.0
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3:
                continue
            if len(by["DEF"]) < d or len(by["MID"]) < m or len(by["FWD"]) < f:
                continue
            pts = (by["GK"][0] + sum(by["DEF"][:d])
                   + sum(by["MID"][:m]) + sum(by["FWD"][:f]))
            best = max(best, pts)
    return best

base = best_xi_points(squad)
print(f"Current best XI for GW{next_gw}: {base:.1f} expected points\n")

team_counts = squad["team"].value_counts().to_dict()
suggestions = []

for _, out in squad.iterrows():
    budget = bank + out["price"]  # approximation: sells at current price
    pool = preds[(preds["pos"] == out["pos"])
                 & (~preds["player_id"].isin(squad_ids))
                 & (preds["price"] <= budget)]
    pool = pool.sort_values("predicted_points", ascending=False).head(40)

    for _, cand in pool.iterrows():
        # max 3 per club after the swap
        club_count = team_counts.get(cand["team"], 0)
        if cand["team"] == out["team"]:
            club_count -= 1
        if club_count >= 3:
            continue

        new_squad = pd.concat(
            [squad[squad["player_id"] != out["player_id"]], cand.to_frame().T],
            ignore_index=True)
        new_squad["predicted_points"] = new_squad["predicted_points"].astype(float)
        gain = best_xi_points(new_squad) - base
        suggestions.append({
            "sell": out["web_name"], "buy": cand["web_name"], "pos": out["pos"],
            "cost_change": (cand["price"] - out["price"]) / 10,
            "gain": gain, "buy_news": cand["news"],
        })

result = pd.DataFrame(suggestions).sort_values("gain", ascending=False).head(8)
result["gain"] = result["gain"].round(2)
print("Best single transfers (gain = change in expected XI points):")
print(result.to_string(index=False))
print("\nReminder: an extra transfer beyond your free one costs -4 points,")
print("so only take a paid transfer if the gain is clearly above 4.")