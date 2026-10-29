import json
import os
from typing import List, Optional

import pandas as pd
import pulp
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
HEADERS = {"User-Agent": "Mozilla/5.0"}
POSITIONS = ["GK", "DEF", "MID", "FWD"]

PLAYER_COLS = ["player_id", "web_name", "pos", "team_name", "price", "news",
               "fixture", "fdr", "model_points", "predicted_points"]


class TeamNotFound(Exception):
    pass


def _records(df: pd.DataFrame):
    # to_json handles numpy types and NaN (becomes null), so the result is JSON-safe
    return json.loads(df.to_json(orient="records"))


def load_predictions() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM predictions", engine)


def top_predictions(limit: int = 20, pos: Optional[str] = None,
                    max_price: Optional[float] = None,
                    min_price: Optional[float] = None):
    """Prices in millions (e.g. 5.0). Stored prices are in tenths of a million."""
    preds = load_predictions()
    if pos:
        preds = preds[preds["pos"] == pos.upper()]
    if max_price is not None:
        preds = preds[preds["price"] <= max_price * 10]
    if min_price is not None:
        preds = preds[preds["price"] >= min_price * 10]
    preds = preds.sort_values("predicted_points", ascending=False).head(limit)
    return _records(preds[PLAYER_COLS])


def find_players(names: List[str]):
    preds = load_predictions()
    rows = []
    for name in names:
        hits = preds[preds["web_name"].str.contains(name, case=False, na=False)]
        rows.append(hits.sort_values("predicted_points", ascending=False).head(3))
    if not rows:
        return []
    return _records(pd.concat(rows)[PLAYER_COLS])


def get_team_fixtures(team: str, num_gameweeks: int = 5):
    """Upcoming fixtures and FPL difficulty ratings for a club (name or short name)."""
    teams = pd.read_sql("SELECT id, name, short_name FROM raw_teams", engine)
    exact = teams[(teams["name"].str.lower() == team.lower())
                  | (teams["short_name"].str.lower() == team.lower())]
    match = exact if len(exact) else teams[
        teams["name"].str.contains(team, case=False, na=False)
        | teams["short_name"].str.contains(team, case=False, na=False)]
    if match.empty:
        raise ValueError(f"No club found matching '{team}'")
    t = match.iloc[0]

    gws = pd.read_sql("SELECT id, is_next FROM raw_gameweeks", engine)
    next_gw = int(gws.loc[gws["is_next"], "id"].iloc[0])
    fx = pd.read_sql(
        "SELECT event, team_h, team_a, team_h_difficulty, team_a_difficulty "
        "FROM raw_fixtures", engine)
    fx = fx[(fx["event"] >= next_gw) & ((fx["team_h"] == t["id"]) | (fx["team_a"] == t["id"]))]
    fx = fx.sort_values("event").head(num_gameweeks)

    short = dict(zip(teams["id"], teams["short_name"]))
    out = []
    for _, r in fx.iterrows():
        is_home = r["team_h"] == t["id"]
        opp = r["team_a"] if is_home else r["team_h"]
        out.append({
            "gameweek": int(r["event"]),
            "opponent": short[opp],
            "venue": "H" if is_home else "A",
            "difficulty": int(r["team_h_difficulty"] if is_home else r["team_a_difficulty"]),
        })
    return {"club": t["name"], "fixtures": out,
            "note": "Difficulty: 1 = easiest, 5 = hardest (FPL's own rating)"}


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

    cols = ["player_id", "web_name", "pos", "team_name", "fixture", "fdr",
            "predicted_points"]
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


def suggest_transfers(team_id: int, top_n: int = 8,
                      bank_override: Optional[float] = None):
    """bank_override is in millions (e.g. 2.5) and replaces the real bank balance."""
    preds = load_predictions()
    squad, squad_ids, bank, next_gw = fetch_squad(team_id, preds)
    if bank_override is not None:
        bank = int(round(bank_override * 10))   # £2.5m -> 25 (tenths of a million)
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
                "gain": round(float(gain), 2),
                "buy_fixture": cand["fixture"], "buy_news": cand["news"],
            })

    suggestions.sort(key=lambda s: s["gain"], reverse=True)
    return {
        "team_id": team_id,
        "gameweek": next_gw,
        "bank_millions": bank / 10,
        "current_xi_points": round(float(base), 2),
        "suggestions": suggestions[:top_n],
    }