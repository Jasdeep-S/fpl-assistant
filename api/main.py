import logging
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field

from . import chat as chat_module
from . import core

app = FastAPI(title="TheDugout API")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _handle(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except core.TeamNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="FPL API unavailable, try again shortly")


def _parse_ids(text: Optional[str]) -> set:
    if not text:
        return set()
    try:
        return {int(x) for x in text.split(",") if x.strip()}
    except ValueError:
        raise HTTPException(status_code=422,
                            detail="locked must be comma-separated player IDs")


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


@app.get("/lineup/{team_id}")
def lineup(team_id: int):
    return _handle(core.pick_lineup, team_id)


@app.get("/transfers/{team_id}")
def transfers(team_id: int, top_n: int = Query(8, ge=1, le=20),
              bank: Optional[float] = None, locked: Optional[str] = None):
    ids = set(list(_parse_ids(locked))[:30])
    return _handle(core.suggest_transfers, team_id, top_n, bank, ids)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500)
    team_id: Optional[int] = None
    locked: List[int] = Field(default_factory=list, max_length=15)
    history: list = Field(default_factory=list, max_length=10)


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        reply = chat_module.chat(req.message, req.team_id, req.history, req.locked)
    except chat_module.AssistantBusy as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Chat failed")
        raise HTTPException(status_code=500,
                            detail="Something went wrong. Please try again.")
    return {"reply": reply}