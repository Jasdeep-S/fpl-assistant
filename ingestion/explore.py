import requests

MAIN_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
resp = requests.get(MAIN_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
resp.raise_for_status()
data = resp.json()

print("Top-level keys:", list(data.keys()))
print("Number of players:", len(data["elements"]))
print("First player:", data["elements"][0]["web_name"])