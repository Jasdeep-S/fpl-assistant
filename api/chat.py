import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from . import core

load_dotenv()

# 30-second timeout (in milliseconds) so a stuck call raises an error instead of freezing
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=30000),
)

# Tried in order. If one is overloaded or unavailable, the next is used.
MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

# Status codes that mean "try the next model"
RETRY_CODES = (404, 429, 500, 503)

SYSTEM = """You are FPL Gaffer, a Fantasy Premier League assistant.

Rules:
- All numbers (predicted points, gains, prices) must come from the tools. Never invent stats.
- Predicted points are model estimates of expected points, not guarantees. Say so when relevant.
- The model only looks at recent form and does not know fixture difficulty yet, so be honest
  about that when a user asks about fixtures.
- If the user has given a team ID, use it for lineup and transfer questions. If they ask about
  their team and no ID is available, ask them for it.
- Prices are in tenths of a million (100 = £10.0m) in the raw data; show them as £x.xm.
- If a tool returns an error, tell the user plainly and do not guess the answer.
- Keep answers short and explain the reasoning behind a recommendation."""


# Tools: Gemini reads the docstrings and type hints to decide when to call them.
# Each returns an {"error": ...} dict on failure so the model can explain it to the user.
def get_top_predictions(limit: int = 10, pos: str = "") -> list:
    """Top players by predicted points for the next gameweek.
    pos can be GK, DEF, MID or FWD, or empty for all positions."""
    try:
        return core.top_predictions(min(limit, 30), pos or None)
    except Exception as e:
        return [{"error": str(e)}]


def get_player(name: str) -> list:
    """Look up a player's predicted points, price, team and injury news by name, e.g. 'Haaland'."""
    try:
        preds = core.load_predictions()
        hits = preds[preds["web_name"].str.contains(name, case=False, na=False)]
        hits = hits.sort_values("predicted_points", ascending=False).head(5)
        return core._records(hits[["player_id", "web_name", "pos", "team", "price",
                                   "news", "predicted_points"]])
    except Exception as e:
        return [{"error": str(e)}]


def get_lineup(team_id: int) -> dict:
    """Best starting XI, captain, vice-captain and bench order for a user's FPL team ID."""
    try:
        return core.pick_lineup(team_id)
    except Exception as e:
        return {"error": str(e)}


def suggest_transfers(team_id: int) -> dict:
    """Best single transfers for a user's FPL team ID, with the expected gain in XI points."""
    try:
        return core.suggest_transfers(team_id)
    except Exception as e:
        return {"error": str(e)}


TOOLS = [get_top_predictions, get_player, get_lineup, suggest_transfers]


def chat(message: str, team_id: Optional[int] = None, history: Optional[list] = None) -> str:
    contents = []
    for turn in history or []:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append(types.Content(
            role=role, parts=[types.Part(text=str(turn.get("content", "")))]))

    text = message if team_id is None else f"[User's team ID: {team_id}]\n{message}"
    contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=TOOLS,
        max_output_tokens=1000,
    )

    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config)
            print(f"[answered by {model}]")  # handy while testing; remove later
            return response.text or "Sorry, I couldn't answer that. Try rephrasing."
        except errors.APIError as e:
            if e.code in RETRY_CODES:
                print(f"[{model} failed with {e.code}, trying next model]")
                last_error = e
                continue
            raise
        except httpx.TimeoutException as e:
            print(f"[{model} timed out, trying next model]")
            last_error = e
            continue

    raise last_error