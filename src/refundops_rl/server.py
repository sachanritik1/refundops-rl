"""Local development HTTP adapter. NEVER expose to the public internet as-is.

State is process-local; for multiple replicas move session state into a durable
store and implement a transaction/CAS per episode. Add auth + rate limits before
any shared deployment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .environment import RefundEnvironment
from .scenarios import FAMILIES

TTL = timedelta(minutes=30)
MAX_SESSIONS = 200

app = FastAPI(title="RefundOps RL Environment", version="0.1.0")


class CreateEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=0, ge=0, le=2_000_000_000)
    family: str | None = None


class ToolAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str = Field(min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Session:
    env: RefundEnvironment
    expires_at: datetime
    lock: RLock = field(default_factory=RLock)


_sessions: dict[str, Session] = {}
_store_lock = RLock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/episodes", status_code=201)
def create_episode(request: CreateEpisode) -> dict[str, Any]:
    if request.family is not None and request.family not in FAMILIES:
        raise HTTPException(status_code=422, detail="unknown family")
    env = RefundEnvironment()
    observation = env.reset(seed=request.seed, family=request.family)
    with _store_lock:
        now = _now()
        for episode_id, session in list(_sessions.items()):
            if session.expires_at <= now:
                del _sessions[episode_id]
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(status_code=429, detail="episode_capacity_exceeded")
        episode_id = uuid4().hex
        _sessions[episode_id] = Session(env=env, expires_at=now + TTL)
    return {"episode_id": episode_id, "observation": observation}


@app.post("/v1/episodes/{episode_id}/step")
def step_episode(episode_id: str, action: ToolAction) -> dict[str, Any]:
    with _store_lock:
        session = _sessions.get(episode_id)
        if not session or session.expires_at <= _now():
            raise HTTPException(status_code=404, detail="episode_not_found_or_expired")
        # Hold store lock only for lookup, then per-session lock for state mutation.
        session.lock.acquire()
    try:
        try:
            result = session.env.step(action.model_dump())
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return asdict(result)
    finally:
        session.lock.release()


@app.delete("/v1/episodes/{episode_id}", status_code=204)
def delete_episode(episode_id: str) -> None:
    with _store_lock:
        session = _sessions.get(episode_id)
        if session is None:
            raise HTTPException(status_code=404, detail="episode_not_found")
        session.lock.acquire()
        try:
            del _sessions[episode_id]
        finally:
            session.lock.release()
