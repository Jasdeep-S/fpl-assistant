import logging
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import chat as chat_module
from . import core

app = FastAPI(title="FPL Gaffer API")


def _handle(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except core.TeamNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="FPL API unavailable, try again shortly")


@app.get("/")
def health():
    return {"status": "ok"}


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
              bank: Optional[float] = None):
    return _handle(core.suggest_transfers, team_id, top_n, bank)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=500)
    team_id: Optional[int] = None
    history: list = Field(default_factory=list, max_length=10)


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        reply = chat_module.chat(req.message, req.team_id, req.history)
    except chat_module.AssistantBusy as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logging.exception("Chat failed")  # full error goes to your server logs
        raise HTTPException(status_code=500,
                            detail="Something went wrong. Please try again.")
    return {"reply": reply}