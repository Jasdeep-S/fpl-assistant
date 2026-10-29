import requests
from fastapi import FastAPI, HTTPException, Query

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
def predictions(limit: int = Query(20, ge=1, le=100), pos: str = None):
    return core.top_predictions(limit, pos)


@app.get("/lineup/{team_id}")
def lineup(team_id: int):
    return _handle(core.pick_lineup, team_id)


@app.get("/transfers/{team_id}")
def transfers(team_id: int, top_n: int = Query(8, ge=1, le=20)):
    return _handle(core.suggest_transfers, team_id, top_n)