"""
NyayaMitra — Session Manager.

Redis-backed session storage for multi-turn conversation history.
Each session tracks the conversation between a user and NyayaMitra,
storing messages, metadata, and the domain context.

Sessions are identified by a UUID and expire after a configurable TTL
(default 24 hours). The frontend sends session_id with each request
to maintain conversation continuity.

Storage format (Redis hash per session):
    session:{id}:messages  → JSON list of {role, content, timestamp}
    session:{id}:meta      → JSON dict of {created_at, domain, jurisdiction, query_count}

Usage:
    from app.services.session import get_session_manager

    manager = await get_session_manager()

    # Create a new session
    session = await manager.create_session()

    # Add messages
    await manager.add_message(session_id, "user", "Can police arrest without warrant?")
    await manager.add_message(session_id, "assistant", "Under Section 41 CrPC...")

    # Get history for LLM context
    history = await manager.get_history(session_id)

    # Get session info
    info = await manager.get_session(session_id)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import structlog

from app.config import settings

logger = structlog.get_logger()

# Default session TTL: 24 hours
SESSION_TTL_SECONDS = int(getattr(settings, "SESSION_TTL_HOURS", 24)) * 3600

# Maximum messages per session (to prevent unbounded growth)
MAX_MESSAGES_PER_SESSION = 100

# Maximum history messages sent to LLM (to stay within context window)
MAX_HISTORY_FOR_LLM = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Session Manager
# ═══════════════════════════════════════════════════════════════════════════════


class SessionManager:
    """
    Redis-backed session manager for conversation history.

    Each session is stored as two Redis keys:
        session:{id}:messages — JSON list of message dicts
        session:{id}:meta     — JSON dict of session metadata

    Both keys share the same TTL and are refreshed on every interaction.
    """

    def __init__(self):
        self.redis = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        try:
            import redis.asyncio as aioredis

            self.redis = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
            )

            # Verify connection
            await self.redis.ping()
            self._connected = True
            logger.info("session_manager_connected", redis_url=settings.redis_url)

        except Exception as e:
            logger.warning(
                "session_manager_redis_unavailable",
                error=str(e),
                message="Redis unavailable. Sessions will not persist across requests.",
            )
            self._connected = False

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self._connected = False
            logger.info("session_manager_closed")

    def _messages_key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    def _meta_key(self, session_id: str) -> str:
        return f"session:{session_id}:meta"

    # ─── Session Lifecycle ───────────────────────────────────────────────

    async def create_session(
        self,
        domain: str | None = None,
        jurisdiction: str | None = None,
    ) -> dict:
        """
        Create a new conversation session.

        Returns:
            dict with session_id, created_at, domain, jurisdiction
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        meta = {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "domain": domain or "",
            "jurisdiction": jurisdiction or "",
            "query_count": 0,
        }

        if self._connected:
            try:
                pipe = self.redis.pipeline()
                pipe.set(self._messages_key(session_id), json.dumps([]))
                pipe.set(self._meta_key(session_id), json.dumps(meta))
                pipe.expire(self._messages_key(session_id), SESSION_TTL_SECONDS)
                pipe.expire(self._meta_key(session_id), SESSION_TTL_SECONDS)
                await pipe.execute()
            except Exception as e:
                logger.warning("session_create_redis_error", error=str(e))

        logger.info("session_created", session_id=session_id)
        return meta

    async def get_session(self, session_id: str) -> dict | None:
        """
        Get session metadata.

        Returns None if session doesn't exist or Redis is unavailable.
        """
        if not self._connected:
            return None

        try:
            raw = await self.redis.get(self._meta_key(session_id))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("session_get_error", session_id=session_id, error=str(e))

        return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        if not self._connected:
            return False

        try:
            pipe = self.redis.pipeline()
            pipe.delete(self._messages_key(session_id))
            pipe.delete(self._meta_key(session_id))
            results = await pipe.execute()
            deleted = sum(results) > 0
            if deleted:
                logger.info("session_deleted", session_id=session_id)
            return deleted
        except Exception as e:
            logger.warning("session_delete_error", session_id=session_id, error=str(e))
            return False

    # ─── Message Management ──────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to the session history.

        Args:
            session_id: The session UUID.
            role: "user" or "assistant".
            content: The message text.
        """
        if not self._connected:
            return

        now = datetime.now(timezone.utc).isoformat()
        message = {
            "role": role,
            "content": content,
            "timestamp": now,
        }

        try:
            # Get existing messages
            raw = await self.redis.get(self._messages_key(session_id))
            messages = json.loads(raw) if raw else []

            # Enforce max messages (drop oldest if needed)
            if len(messages) >= MAX_MESSAGES_PER_SESSION:
                messages = messages[-(MAX_MESSAGES_PER_SESSION - 1):]

            messages.append(message)

            # Update messages and meta
            pipe = self.redis.pipeline()
            pipe.set(self._messages_key(session_id), json.dumps(messages))

            # Update meta
            meta_raw = await self.redis.get(self._meta_key(session_id))
            if meta_raw:
                meta = json.loads(meta_raw)
                meta["updated_at"] = now
                if role == "user":
                    meta["query_count"] = meta.get("query_count", 0) + 1
                pipe.set(self._meta_key(session_id), json.dumps(meta))

            # Refresh TTL
            pipe.expire(self._messages_key(session_id), SESSION_TTL_SECONDS)
            pipe.expire(self._meta_key(session_id), SESSION_TTL_SECONDS)
            await pipe.execute()

        except Exception as e:
            logger.warning(
                "session_add_message_error",
                session_id=session_id,
                role=role,
                error=str(e),
            )

    async def get_history(
        self,
        session_id: str,
        max_messages: int | None = None,
    ) -> list[dict]:
        """
        Get conversation history for a session.

        Returns a list of {role, content} dicts suitable for
        passing to the LLM as conversation history.

        Args:
            session_id: The session UUID.
            max_messages: Max messages to return (default: MAX_HISTORY_FOR_LLM).

        Returns:
            List of message dicts, oldest first.
        """
        if not self._connected:
            return []

        max_messages = max_messages or MAX_HISTORY_FOR_LLM

        try:
            raw = await self.redis.get(self._messages_key(session_id))
            if not raw:
                return []

            messages = json.loads(raw)

            # Return last N messages, stripping timestamp for LLM
            recent = messages[-max_messages:]
            return [
                {"role": m["role"], "content": m["content"]}
                for m in recent
            ]

        except Exception as e:
            logger.warning("session_get_history_error", session_id=session_id, error=str(e))
            return []

    async def get_full_history(self, session_id: str) -> list[dict]:
        """
        Get the complete conversation history with timestamps.

        Returns all messages including timestamps, for display purposes.
        """
        if not self._connected:
            return []

        try:
            raw = await self.redis.get(self._messages_key(session_id))
            if not raw:
                return []
            return json.loads(raw)
        except Exception as e:
            logger.warning("session_get_full_error", session_id=session_id, error=str(e))
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_session_manager: SessionManager | None = None


async def get_session_manager() -> SessionManager:
    """Get or create the singleton session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        await _session_manager.connect()
    return _session_manager


async def close_session_manager() -> None:
    """Close the singleton session manager. Call on app shutdown."""
    global _session_manager
    if _session_manager is not None:
        await _session_manager.close()
        _session_manager = None