"""
NyayaMitra — Session Management Router.

REST endpoints for managing conversation sessions.
Sessions are stored in Redis and maintain multi-turn conversation
history between the user and NyayaMitra.

Endpoints:
    POST   /api/v1/sessions              Create a new session
    GET    /api/v1/sessions/{id}          Get session info + history
    GET    /api/v1/sessions/{id}/history  Get conversation history only
    DELETE /api/v1/sessions/{id}          Delete a session

The frontend creates a session when the user opens a new chat,
then passes session_id with each /query or /query/stream request.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1")


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response Models
# ═══════════════════════════════════════════════════════════════════════════════


class CreateSessionRequest(BaseModel):
    """Request to create a new conversation session."""

    domain: str | None = Field(
        default=None,
        description="Initial legal domain hint (criminal, property, family, etc.)",
    )
    jurisdiction: str | None = Field(
        default=None,
        description="User's state/jurisdiction.",
    )


class SessionInfo(BaseModel):
    """Session metadata returned to the client."""

    session_id: str
    created_at: str
    updated_at: str
    domain: str
    jurisdiction: str
    query_count: int


class SessionWithHistory(BaseModel):
    """Session metadata plus full conversation history."""

    session_id: str
    created_at: str
    updated_at: str
    domain: str
    jurisdiction: str
    query_count: int
    messages: list[dict]


class MessageEntry(BaseModel):
    """A single message in conversation history."""

    role: str
    content: str
    timestamp: str | None = None


class DeleteResponse(BaseModel):
    """Response after deleting a session."""

    deleted: bool
    session_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/sessions", response_model=SessionInfo)
async def create_session(request: CreateSessionRequest | None = None):
    """
    Create a new conversation session.

    Returns the session_id to be passed with subsequent /query requests.
    Optionally accepts a domain hint and jurisdiction to pre-configure
    the session context.
    """
    from app.services.session import get_session_manager

    manager = await get_session_manager()

    domain = request.domain if request else None
    jurisdiction = request.jurisdiction if request else None

    meta = await manager.create_session(
        domain=domain,
        jurisdiction=jurisdiction,
    )

    return SessionInfo(
        session_id=meta["session_id"],
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        domain=meta.get("domain", ""),
        jurisdiction=meta.get("jurisdiction", ""),
        query_count=meta.get("query_count", 0),
    )


@router.get("/sessions/{session_id}", response_model=SessionWithHistory)
async def get_session(session_id: str):
    """
    Get session info and full conversation history.

    Returns 404 if session doesn't exist or has expired.
    """
    from app.services.session import get_session_manager

    manager = await get_session_manager()

    meta = await manager.get_session(session_id)
    if not meta:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found", "session_id": session_id},
        )

    messages = await manager.get_full_history(session_id)

    return SessionWithHistory(
        session_id=meta["session_id"],
        created_at=meta["created_at"],
        updated_at=meta["updated_at"],
        domain=meta.get("domain", ""),
        jurisdiction=meta.get("jurisdiction", ""),
        query_count=meta.get("query_count", 0),
        messages=messages,
    )


@router.get("/sessions/{session_id}/history", response_model=list[MessageEntry])
async def get_session_history(session_id: str):
    """
    Get just the conversation history for a session.

    Lighter endpoint for the frontend to poll or refresh chat history.
    Returns 404 if session doesn't exist.
    """
    from app.services.session import get_session_manager

    manager = await get_session_manager()

    # Check session exists
    meta = await manager.get_session(session_id)
    if not meta:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found", "session_id": session_id},
        )

    messages = await manager.get_full_history(session_id)

    return [
        MessageEntry(
            role=m.get("role", ""),
            content=m.get("content", ""),
            timestamp=m.get("timestamp"),
        )
        for m in messages
    ]


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str):
    """
    Delete a session and all its conversation history.

    Returns whether the session was found and deleted.
    """
    from app.services.session import get_session_manager

    manager = await get_session_manager()
    deleted = await manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"error": "Session not found", "session_id": session_id},
        )

    return DeleteResponse(deleted=True, session_id=session_id)