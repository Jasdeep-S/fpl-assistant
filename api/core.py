import json
import os
from typing import Optional

import pandas as pd
import pulp
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
HEADERS = {"User-Agent": "Mozilla/5.0"}
POSITIONS = ["GK", "DEF", "MID", "FWD"]


class TeamNotFound(Exception):
    pass


def _records(df: pd.DataFrame):
    # to_json handles numpy types and NaN (becomes null), so the result is JSON-safe
    return json.loads(df.to_json(orient="records"))


def load_predictions() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM predictions", engine)


def top_predictions(limit: int = 20, pos: Optional[str] = None):
    preds = load_predictions()
    if pos:
        preds = preds[preds["pos"] == pos.upper()]
    preds = preds.sort_values("predicted_points", ascending=False).head(limit)
    cols = ["player_id", "web_name", "pos", "team", "price", "news", "predicted_points"]
    return _records(preds[cols])


def fetch_squad(team_id: int, preds: pd.DataFrame):
    next_gw = int(preds["gameweek"].iloc[0])
    last_gw = max(next_gw - 1, 1)
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{last_gw}/picks/"
    resp = requests.get(url, timeout=30, headers=HEADERS)
    if resp.status_code == 404:
        raise TeamNotFound(f"No picks found for team {team_id} in GW{last_gw}")
    resp.raise_for_status()
    data = resp.json()
    squad_ids = [p["element"] for p in data["picks"]]
    bank = data["entry_history"]["bank"]
    squad = preds[preds["player_id"].isin(squad_ids)].reset_index(drop=True)
    return squad, squad_ids, bank, next_gw


def pick_lineup(team_id: int):
    preds = load_predictions()
    squad, squad_ids, bank, next_gw = fetch_squad(team_id, preds)
    if len(squad) < 11:
        raise ValueError("Too few players matched to build a lineup")

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
    bench = pd.concat([
        bench[bench["pos"] == "GK"],
        bench[bench["pos"] != "GK"].sort_values("predicted_points", ascending=False),
    ])

    cap_pool = xi[xi["pos"].isin(["MID", "FWD"])]
    pool = cap_pool if len(cap_pool) >= 2 else xi
    captain, vice = pool.iloc[0], pool.iloc[1]

    cols = ["player_id", "web_name", "pos", "predicted_points"]
    return {
        "team_id": team_id,
        "gameweek": next_gw,
        "starting_xi": _records(xi[cols]),
        "bench": _records(bench[cols]),
        "captain": captain["web_name"],
        "vice_captain": vice["web_name"],
        "expected_points": round(float(xi["predicted_points"].sum()
                                       + captain["predicted_points"]), 2),
    }


def best_xi_points(df: pd.DataFrame) -> float:
    by = {p: sorted(df.loc[df["pos"] == p, "predicted_points"].astype(float),
                    reverse=True) for p in POSITIONS}
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


def suggest_transfers(team_id: int, top_n: int = 8):
    preds = load_predictions()
    squad, squad_ids, bank, next_gw = fetch_squad(team_id, preds)
    base = best_xi_points(squad)
    team_counts = squad["team"].value_counts().to_dict()
    suggestions = []

    for _, out in squad.iterrows():
        budget = bank + out["price"]  # approximation: sells at current price
        pool = preds[(preds["pos"] == out["pos"])
                     & (~preds["player_id"].isin(squad_ids))
                     & (preds["price"] <= budget)]
        pool = pool.sort_values("predicted_points", ascending=False).head(40)

        for _, cand in pool.iterrows():
            club_count = team_counts.get(cand["team"], 0)
            if cand["team"] == out["team"]:
                club_count -= 1
            if club_count >= 3:
                continue
            new_squad = pd.concat(
                [squad[squad["player_id"] != out["player_id"]], cand.to_frame().T],
                ignore_index=True)
            gain = best_xi_points(new_squad) - base
            suggestions.append({
                "sell": out["web_name"], "buy": cand["web_name"], "pos": out["pos"],
                "cost_change": round(float(cand["price"] - out["price"]) / 10, 1),
                "gain": round(float(gain), 2), "buy_news": cand["news"],
            })

    suggestions.sort(key=lambda s: s["gain"], reverse=True)
    return {
        "team_id": team_id,
        "gameweek": next_gw,
        "bank": bank / 10,
        "current_xi_points": round(float(base), 2),
        "suggestions": suggestions[:top_n],
    }