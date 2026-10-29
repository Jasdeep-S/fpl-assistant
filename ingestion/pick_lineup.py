import os
import sys
import pulp
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

if len(sys.argv) < 2:
    sys.exit("Usage: python ingestion/pick_lineup.py YOUR_TEAM_ID")
team_id = int(sys.argv[1])

preds = pd.read_sql("SELECT * FROM predictions", engine)
next_gw = int(preds["gameweek"].iloc[0])

# The picks endpoint only has data once a gameweek's deadline has passed,
# so import the squad from the last completed gameweek
last_gw = max(next_gw - 1, 1)
url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_gw}/picks/"
resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
if resp.status_code == 404:
    sys.exit(f"No picks found for team {team_id} in GW{last_gw}. Check the ID.")
resp.raise_for_status()
squad_ids = [p["element"] for p in resp.json()["picks"]]

squad = preds[preds["player_id"].isin(squad_ids)].reset_index(drop=True)
print(f"Matched {len(squad)} of {len(squad_ids)} squad players to predictions")
if len(squad) < 11:
    sys.exit("Too few players matched to build a lineup.")

# Pick the 11 that maximise predicted points under FPL formation rules
prob = pulp.LpProblem("lineup", pulp.LpMaximize)
x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in squad.index}

prob += pulp.lpSum(x[i] * squad.loc[i, "predicted_points"] for i in squad.index)
prob += pulp.lpSum(x.values()) == 11

def count(pos):
    return pulp.lpSum(x[i] for i in squad.index if squad.loc[i, "pos"] == pos)

prob += count("GK") == 1
prob += count("DEF") >= 3
prob += count("DEF") <= 5
prob += count("MID") >= 2
prob += count("MID") <= 5
prob += count("FWD") >= 1
prob += count("FWD") <= 3

prob.solve(pulp.PULP_CBC_CMD(msg=0))

squad["starts"] = [int(round(x[i].value())) for i in squad.index]
xi = squad[squad["starts"] == 1].sort_values("predicted_points", ascending=False)
bench = squad[squad["starts"] == 0]

# Bench order: backup keeper first, then outfield players by predicted points
bench_gk = bench[bench["pos"] == "GK"]
bench_out = bench[bench["pos"] != "GK"].sort_values("predicted_points", ascending=False)
bench = pd.concat([bench_gk, bench_out])

# Captain and vice: attackers only (MID/FWD), since defender points are
# clean-sheet driven and captains are usually picked for ceiling
cap_pool = xi[xi["pos"].isin(["MID", "FWD"])]
if len(cap_pool) >= 2:
    captain, vice = cap_pool.iloc[0], cap_pool.iloc[1]
else:
    captain, vice = xi.iloc[0], xi.iloc[1]

print(f"\nStarting XI for GW{next_gw}:")
print(xi[["web_name", "pos", "predicted_points"]].to_string(index=False))
print(f"\nCaptain: {captain['web_name']}   Vice: {vice['web_name']}")
print("\nBench (in order):")
print(bench[["web_name", "pos", "predicted_points"]].to_string(index=False))

total = xi["predicted_points"].sum() + captain["predicted_points"]
print(f"\nExpected points (XI + captain bonus): {total:.1f}")