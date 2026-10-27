import json
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
resp = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
data = resp.json()

def flatten_nested(df):
    # Postgres can't store raw Python lists/dicts, so convert them to JSON text
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].apply(json.dumps)
    return df

for key, table in [("elements", "raw_players"),
                   ("teams", "raw_teams"),
                   ("events", "raw_gameweeks")]:
    df = flatten_nested(pd.DataFrame(data[key]))
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"Loaded {len(df)} rows into {table}")