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
    url = f"{BASE}/{season}/gws/merged_gw.csv"
    try:
        df = pd.read_csv(url, on_bad_lines="skip", encoding="utf-8")
    except Exception as e:
        print(f"Skipping {season}: {e}")
        continue
    df["season"] = season
    frames.append(df)
    print(f"{season}: {len(df)} rows")

all_df = pd.concat(frames, ignore_index=True)
all_df.to_sql("raw_history_seasons", engine, if_exists="replace", index=False)
print(f"Loaded {len(all_df)} rows into raw_history_seasons")