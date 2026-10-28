import json
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.environ["DATABASE_URL"])

URL = "https://fantasy.premierleague.com/api/fixtures/"
resp = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()

df = pd.DataFrame(resp.json())
# stats is a nested list, so store it as JSON text
for col in df.columns:
    if df[col].apply(lambda v: isinstance(v, (list, dict))).any():
        df[col] = df[col].apply(json.dumps)

df.to_sql("raw_fixtures", engine, if_exists="replace", index=False)
print(f"Loaded {len(df)} rows into raw_fixtures")