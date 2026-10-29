import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

frames = []
for season in SEASONS:
    df = pd.read_csv(f"{BASE}/{season}/teams.csv")
    df["season"] = season
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df.to_sql("raw_teams_seasons", engine, if_exists="replace", index=False)
print(f"Loaded {len(all_df)} rows into raw_teams_seasons")