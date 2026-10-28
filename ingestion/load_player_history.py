import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])
HEADERS = {"User-Agent": "Mozilla/5.0"}

players = pd.read_sql("SELECT id FROM raw_players", engine)["id"].tolist()

rows = []
for i, pid in enumerate(players, 1):
    url = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
    resp = requests.get(url, timeout=30, headers=HEADERS)
    if resp.status_code != 200:
        print(f"Skipping player {pid}: status {resp.status_code}")
        continue
    for game in resp.json()["history"]:
        game["player_id"] = pid
        rows.append(game)
    if i % 50 == 0:
        print(f"{i}/{len(players)} players done")

    # sleep 3 seconds to not break the api 
    time.sleep(0.3)

df = pd.DataFrame(rows)
df.to_sql("raw_player_history", engine, if_exists="replace", index=False)
print(f"Loaded {len(df)} rows into raw_player_history")