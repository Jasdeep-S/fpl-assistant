import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from . import chat as chat_module
from . import core

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]
CHAT_DAILY_LIMIT = int(os.environ.get("CHAT_DAILY_LIMIT", "20"))
CHAT_GLOBAL_DAILY_LIMIT = int(os.environ.get("CHAT_GLOBAL_DAILY_LIMIT", "300"))
TODAY = "(now() AT TIME ZONE 'UTC')::date"   # usage resets at midnight UTC

app = FastAPI(title="TheDugout API")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.on_event("startup")
def create_tables():
    with core.engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_prefs (
                user_id TEXT PRIMARY KEY,
                team_id INT,
                locked JSONB NOT NULL DEFAULT '[]',
                updated_at TIMESTAMPTZ DEFAULT now())"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_usage (
                user_id TEXT NOT NULL,
                day DATE NOT NULL,
                count INT NOT NULL DEFAULT 0,
                PRIMARY KEY (user_id, day))"""))


# ---------- authentication ----------
_token_cache: dict = {}


def current_user(authorization: Optional[str] = Header(None)) -> dict:
    """Checks the sign-in token with Supabase and returns the user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Please sign in to use this feature.")
    token = authorization[7:].strip()

    hit = _token_cache.get(token)
    if hit and hit[1] > time.time():
        return hit[0]

    try:
        r = requests.get(f"{SUPABASE_URL}/auth/v1/user",
                         headers={"Authorization": f"Bearer {token}",
                                  "apikey": SUPABASE_ANON_KEY},
                         timeout=10)
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503,
                            detail="Sign-in service unavailable, try again shortly.")
    if r.status_code != 200:
        raise HTTPException(status_code=401,
                            detail="Your session expired. Please sign in again.")
    data = r.json()
    user = {"id": data["id"], "email": data.get("email")}
    if len(_token_cache) > 500:
        _token_cache.clear()
    _token_cache[token] = (user, time.time() + 60)  # re-check with Supabase every minute
    return user


# ---------- chat limits ----------
def reserve_chat(user_id: str) -> int:
    """Counts one message for today and returns how many the user has left.
    Raises 429 if a limit is reached. Raising inside the transaction rolls the count back."""
    with core.engine.begin() as conn:
        total = conn.execute(text(
            f"SELECT COALESCE(SUM(count), 0) FROM chat_usage WHERE day = {TODAY}")).scalar()
        if total >= CHAT_GLOBAL_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="TheDugout has reached its daily chat limit. Please try again tomorrow.")
        used = conn.execute(text(f"""
            INSERT INTO chat_usage (user_id, day, count) VALUES (:u, {TODAY}, 1)
            ON CONFLICT (user_id, day) DO UPDATE SET count = chat_usage.count + 1
            RETURNING count"""), {"u": user_id}).scalar()
        if used > CHAT_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"You've used your {CHAT_DAILY_LIMIT} chat messages for today. "
                       "It resets at midnight UTC.")
    return CHAT_DAILY_LIMIT - used


def refund_chat(user_id: str) -> None:
    """Gives the message back when the assistant failed, so errors aren't charged."""
    with core.engine.begin() as conn:
        conn.execute(text(
            f"UPDATE chat_usage SET count = GREATEST(count - 1, 0) "
            f"WHERE user_id = :u AND day = {TODAY}"), {"u": user_id})


def chat_remaining(user_id: str) -> int:
    with core.engine.begin() as conn:
        used = conn.execute(text(
            f"SELECT COALESCE(SUM(count), 0) FROM chat_usage "
            f"WHERE user_id = :u AND day = {TODAY}"), {"u": user_id}).scalar()
    return max(CHAT_DAILY_LIMIT - int(used), 0)


def _handle(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except core.TeamNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="FPL API unavailable, try again shortly")


def _parse_ids(text_value: Optional[str]) -> set:
    if not text_value:
        return set()
    try:
        return {int(x) for x in text_value.split(",") if x.strip()}
    except ValueError:
        raise HTTPException(status_code=422,
                            detail="locked must be comma-separated player IDs")


# ---------- public endpoints ----------
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/deadline")
def deadline():
    return _handle(core.next_deadline)


@app.get("/gameweeks")
def gameweeks(n: int = Query(6, ge=1, le=10)):
    return _handle(core.gameweek_flags, n)


@app.get("/predictions")
def predictions(limit: int = Query(20, ge=1, le=100), pos: Optional[str] = None,
                max_price: Optional[float] = None):
    return core.top_predictions(limit, pos, max_price)


@app.get("/fixtures/{team}")
def fixtures(team: str, n: int = Query(5, ge=1, le=8)):
    return _handle(core.get_team_fixtures, team, n)

@app.get("/fixture-grid")
def fixture_grid(n: int = Query(6, ge=1, le=8)):
    return _handle(core.fixture_grid, n)

# ---------- signed-in endpoints ----------
class Prefs(BaseModel):
    team_id: Optional[int] = Field(None, ge=1, le=100000000)
    locked: List[int] = Field(default_factory=list, max_length=30)


@app.get("/me")
def me(user: dict = Depends(current_user)):
    with core.engine.begin() as conn:
        row = conn.execute(text("SELECT team_id, locked FROM user_prefs WHERE user_id = :u"),
                           {"u": user["id"]}).first()
    return {"email": user["email"],
            "team_id": row[0] if row else None,
            "locked": row[1] if row else [],
            "chat_remaining": chat_remaining(user["id"])}


@app.put("/me")
def save_me(prefs: Prefs, user: dict = Depends(current_user)):
    with core.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_prefs (user_id, team_id, locked)
            VALUES (:u, :t, CAST(:l AS jsonb))
            ON CONFLICT (user_id) DO UPDATE
            SET team_id = :t, locked = CAST(:l AS jsonb), updated_at = now()"""),
            {"u": user["id"], "t": prefs.team_id, "l": json.dumps(prefs.locked)})
    return {"ok": True}


@app.get("/lineup/{team_id}")
def lineup(team_id: int, user: dict = Depends(current_user)):
    return _handle(core.pick_lineup, team_id)


@app.get("/transfers/{team_id}")
def transfers(team_id: int, top_n: int = Query(8, ge=1, le=20),
              bank: Optional[float] = None, locked: Optional[str] = None,
              user: dict = Depends(current_user)):
    ids = set(list(_parse_ids(locked))[:30])
    return _handle(core.suggest_transfers, team_id, top_n, bank, ids)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500)
    team_id: Optional[int] = None
    locked: List[int] = Field(default_factory=list, max_length=15)
    history: list = Field(default_factory=list, max_length=10)


@app.post("/chat")
def chat_endpoint(req: ChatRequest, user: dict = Depends(current_user)):
    remaining = reserve_chat(user["id"])        # 429 here if a limit is reached
    try:
        reply = chat_module.chat(req.message, req.team_id, req.history, req.locked)
    except chat_module.AssistantBusy as e:
        refund_chat(user["id"])                 # not the user's fault, so don't charge
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        refund_chat(user["id"])
        logging.exception("Chat failed")
        raise HTTPException(status_code=500,
                            detail="Something went wrong. Please try again.")
    return {"reply": reply, "remaining": remaining}