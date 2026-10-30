import os
import warnings
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from . import core

warnings.filterwarnings("ignore", message=".*non-text parts.*")
load_dotenv()


class AssistantBusy(Exception):
    """Raised when every model in the fallback list is overloaded or unavailable."""


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
RETRY_CODES = (404, 408, 429, 500, 502, 503, 504)

SYSTEM = """You are TheDugout, a Fantasy Premier League assistant.

Rules:
- All numbers (predicted points, gains, prices, fixtures) must come from the tools. Never invent stats.
- predicted_points are model estimates of expected points for the next gameweek, not guarantees.
- How predictions are built: a model based on recent form (model_points), then adjusted by
  about +/-10% for FPL's fixture difficulty rating (fdr, 1 = easiest, 5 = hardest), and scaled
  by injury availability. The fixture adjustment is a simple rule of thumb, not a learned effect,
  so say so if asked. Predictions cover only the next gameweek.
- Only describe a fixture as easy or hard if the fdr supports it (1-2 easy, 4-5 hard).
- For questions about upcoming runs of fixtures, use get_fixtures.
- For double or blank gameweek questions, use get_gameweek_flags. Predictions already count a
  double gameweek as two fixtures added together, and a blank as zero points. Future gameweeks
  can change, so say the data may be provisional.
- If the user gives a budget (e.g. "I have 2.5m"), pass it as budget_millions to suggest_transfers.
- If the message lists locked player IDs, pass them to suggest_transfers as locked_player_ids.
  Locked players are never suggested as sells.
- If the user has given a team ID, use it for lineup and transfer questions. If they ask about
  their team and no ID is available, ask them for it.
- Prices from tools are in tenths of a million (100 = £10.0m); show them as £x.xm.
- You cannot predict price changes, and you have no data on chips or wildcards. Say so if asked.
- If a tool returns an error, tell the user plainly and do not guess the answer.
- Keep answers short and explain the reasoning behind a recommendation."""


# Tools: Gemini reads the docstrings and type hints to decide when to call them.
# Each returns an error dict on failure so the model can explain it to the user.
def get_top_predictions(limit: int = 10, pos: str = "",
                        max_price_millions: float = -1) -> list:
    """Top players by predicted points for the next gameweek.
    pos can be GK, DEF, MID or FWD, or empty for all positions.
    max_price_millions caps the player's price in millions (e.g. 5.0 for budget
    defenders under £5.0m); leave at -1 for no cap."""
    try:
        return core.top_predictions(
            min(limit, 30), pos or None,
            max_price=max_price_millions if max_price_millions >= 0 else None)
    except Exception as e:
        return [{"error": str(e)}]


def get_player(name: str) -> list:
    """Look up a player's predicted points, price, fixture and injury news by name, e.g. 'Haaland'."""
    try:
        return core.find_players([name])
    except Exception as e:
        return [{"error": str(e)}]


def compare_players(names: List[str]) -> list:
    """Compare several players side by side, e.g. ['Saka', 'Palmer']."""
    try:
        return core.find_players(names[:5])
    except Exception as e:
        return [{"error": str(e)}]


def get_fixtures(team: str, num_gameweeks: int = 5) -> dict:
    """Upcoming fixtures and FPL difficulty ratings (1 easiest, 5 hardest) for a
    Premier League club, e.g. 'Arsenal' or 'ARS'."""
    try:
        return core.get_team_fixtures(team, min(num_gameweeks, 8))
    except Exception as e:
        return {"error": str(e)}


def get_gameweek_flags(num_gameweeks: int = 6) -> dict:
    """Which clubs have a double gameweek (two fixtures) or a blank gameweek
    (no fixture) in the next few gameweeks."""
    try:
        return core.gameweek_flags(min(num_gameweeks, 10))
    except Exception as e:
        return {"error": str(e)}


def get_lineup(team_id: int) -> dict:
    """Best starting XI, captain, vice-captain and bench order for a user's FPL team ID."""
    try:
        return core.pick_lineup(team_id)
    except Exception as e:
        return {"error": str(e)}


def suggest_transfers(team_id: int, budget_millions: float = -1,
                      locked_player_ids: Optional[List[int]] = None) -> dict:
    """Best single transfers for a user's FPL team ID, with the expected gain in XI points.
    Optionally pass budget_millions (money in the bank, e.g. 2.5) to override the real bank
    balance; leave at -1 to use the real one. If the user's message lists locked player IDs,
    pass them as locked_player_ids: those players are never suggested as sells."""
    try:
        return core.suggest_transfers(
            team_id,
            bank_override=budget_millions if budget_millions >= 0 else None,
            locked=set(locked_player_ids or []))
    except Exception as e:
        return {"error": str(e)}


TOOLS = [get_top_predictions, get_player, compare_players, get_fixtures,
         get_gameweek_flags, get_lineup, suggest_transfers]


def chat(message: str, team_id: Optional[int] = None, history: Optional[list] = None,
         locked: Optional[List[int]] = None) -> str:
    contents = []
    for turn in history or []:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append(types.Content(
            role=role, parts=[types.Part(text=str(turn.get("content", "")))]))

    # Context the user can't be trusted to type correctly is added by the app
    prefix = ""
    if team_id is not None:
        prefix += f"[User's team ID: {team_id}]\n"
    if locked:
        prefix += f"[User's locked player IDs: {list(locked)}]\n"
    contents.append(types.Content(
        role="user", parts=[types.Part(text=prefix + message)]))

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
            print(f"[answered by {model}]")  # remove before deploying
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

    raise AssistantBusy(
        "The assistant is busy right now. Please try again in a minute."
    ) from last_error