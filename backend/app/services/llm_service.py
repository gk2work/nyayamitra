"""
NyayaMitra — LLM Service.

Generates structured legal responses by sending retrieved context
to an LLM. Supports streaming (token-by-token SSE) and multi-turn
conversation history.

Providers (tried in order):
    1. vLLM local server (Llama 3.1 8B/70B)
    2. OpenAI API (GPT-4o-mini or configured model)
    3. Anthropic API (Claude Sonnet or configured model)
    4. Any generic OpenAI-compatible endpoint
    5. Fallback: format retrieved context without LLM

Usage:
    from app.services.llm_service import get_llm_service

    service = await get_llm_service()

    # Non-streaming
    result = await service.generate_legal_response(query, context)

    # Streaming
    async for token in service.generate_stream(query, context):
        print(token, end="")

    # Multi-turn
    result = await service.generate_legal_response(
        query, context, history=[{"role": "user", "content": "..."}]
    )
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Legal System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

LEGAL_SYSTEM_PROMPT = """You are NyayaMitra (न्यायमित्र), an AI legal assistant for Indian citizens. You provide accurate, well-cited legal information based on Indian law.

CRITICAL RULES:
1. ONLY cite sections and cases that appear in the CONTEXT below. NEVER fabricate or guess citations.
2. If the context does not contain enough information to fully answer, say so honestly rather than speculating.
3. Always include the disclaimer at the end: "⚖️ This is legal information, not legal advice. For case-specific advice, consult a qualified advocate."
4. Explain legal concepts in simple, plain language that a non-lawyer can understand.
5. Be jurisdiction-aware — note if the law varies by state.
6. When citing a section, always include the full format: "Section X of the Act Name, Year".
7. When citing a case, include: "Case Name (Year) — Court".

RESPONSE FORMAT:
Structure your response clearly with these sections:

## Answer
A clear, plain-language explanation answering the user's question (2-4 paragraphs).

## Applicable Law
For each relevant statutory provision from the context:
- **Section X of Act Name, Year**: Brief explanation of what it provides.

## Key Precedents
For each relevant judgment from the context:
- **Case Name (Year) — Court**: Key principle established.

## What You Should Do
If applicable, provide numbered practical steps the person can take.

## Important Notes
- Jurisdiction-specific caveats.
- When to consult a lawyer.

If any section has no relevant content, omit it rather than writing "None"."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Service
# ═══════════════════════════════════════════════════════════════════════════════


class LLMService:
    """
    Generates legal responses using an LLM.

    Supports vLLM, OpenAI, Anthropic, and any OpenAI-compatible endpoint.
    Provides both non-streaming and streaming generation.
    """

    def __init__(self):
        self.api_url: str = ""
        self.model_name: str = ""
        self.api_key: str = ""
        self.provider: str = ""  # "vllm", "openai", "anthropic", "generic", ""
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the LLM service.

        Tries providers in order: vLLM → OpenAI → Anthropic → Generic → Fallback.
        """
        if self._initialized:
            return

        # Option 1: vLLM local server
        vllm_url = settings.vllm_api_url
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{vllm_url}/models")
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    if models:
                        self.api_url = vllm_url
                        self.model_name = models[0].get("id", settings.VLLM_MODEL_NAME)
                        self.api_key = "not-needed"
                        self.provider = "vllm"
                        self._initialized = True
                        logger.info("llm_initialized", provider="vllm", model=self.model_name)
                        return
        except Exception:
            pass

        # Option 2: OpenAI
        if settings.OPENAI_API_KEY:
            self.api_url = "https://api.openai.com/v1"
            self.model_name = settings.OPENAI_MODEL
            self.api_key = settings.OPENAI_API_KEY
            self.provider = "openai"
            self._initialized = True
            logger.info("llm_initialized", provider="openai", model=self.model_name)
            return

        # Option 3: Anthropic
        anthropic_key = getattr(settings, "ANTHROPIC_API_KEY", "")
        if anthropic_key:
            self.api_url = "https://api.anthropic.com"
            self.model_name = getattr(settings, "ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            self.api_key = anthropic_key
            self.provider = "anthropic"
            self._initialized = True
            logger.info("llm_initialized", provider="anthropic", model=self.model_name)
            return

        # Option 4: Generic OpenAI-compatible
        if settings.LLM_API_URL and settings.LLM_API_KEY:
            self.api_url = settings.LLM_API_URL
            self.model_name = settings.LLM_MODEL_NAME or "default"
            self.api_key = settings.LLM_API_KEY
            self.provider = "generic"
            self._initialized = True
            logger.info("llm_initialized", provider="generic", model=self.model_name)
            return

        # No provider — fallback mode
        self.provider = ""
        self._initialized = True
        logger.warning(
            "llm_no_provider",
            message=(
                "No LLM provider found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "start vLLM, or set LLM_API_URL + LLM_API_KEY. "
                "Using context-only fallback responses."
            ),
        )

    # ─── Non-Streaming Generation ────────────────────────────────────────

    async def generate_legal_response(
        self,
        query: str,
        context: str,
        jurisdiction: str | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Generate a structured legal response.

        Args:
            query: The user's legal question.
            context: Formatted retrieval results from format_context_for_llm().
            jurisdiction: User's state/jurisdiction if provided.
            history: Previous conversation messages for multi-turn.

        Returns:
            dict with: answer, llm_used, model, duration_ms
        """
        if not self._initialized:
            await self.initialize()

        if not self.provider:
            return self._fallback_response(query, context)

        start = time.time()
        user_message = self._build_user_message(query, context, jurisdiction)
        messages = self._build_messages(user_message, history)

        try:
            if self.provider == "anthropic":
                response_text = await self._call_anthropic(messages)
            else:
                response_text = await self._call_openai_compat(messages)

            duration_ms = round((time.time() - start) * 1000, 2)

            logger.info(
                "llm_generation_complete",
                provider=self.provider,
                model=self.model_name,
                query_length=len(query),
                response_length=len(response_text),
                duration_ms=duration_ms,
            )

            return {
                "answer": response_text,
                "llm_used": True,
                "model": self.model_name,
                "provider": self.provider,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error("llm_generation_error", provider=self.provider, error=str(e))
            return self._fallback_response(query, context)

    # ─── Streaming Generation ────────────────────────────────────────────

    async def generate_stream(
        self,
        query: str,
        context: str,
        jurisdiction: str | None = None,
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream legal response tokens.

        Yields text chunks as they arrive from the LLM.
        Falls back to yielding the full fallback response if no provider.

        Args:
            query: The user's legal question.
            context: Formatted retrieval results.
            jurisdiction: Optional jurisdiction.
            history: Previous conversation for multi-turn.

        Yields:
            Text chunks (tokens or small groups of tokens).
        """
        if not self._initialized:
            await self.initialize()

        if not self.provider:
            fallback = self._fallback_response(query, context)
            yield fallback["answer"]
            return

        user_message = self._build_user_message(query, context, jurisdiction)
        messages = self._build_messages(user_message, history)

        try:
            if self.provider == "anthropic":
                async for chunk in self._stream_anthropic(messages):
                    yield chunk
            else:
                async for chunk in self._stream_openai_compat(messages):
                    yield chunk
        except Exception as e:
            logger.error("llm_stream_error", provider=self.provider, error=str(e))
            fallback = self._fallback_response(query, context)
            yield fallback["answer"]

    # ─── Message Building ────────────────────────────────────────────────

    def _build_user_message(
        self,
        query: str,
        context: str,
        jurisdiction: str | None = None,
    ) -> str:
        """Build the user message with query and retrieved context."""
        parts = [f"USER QUESTION: {query}"]

        if jurisdiction:
            parts.append(f"\nJURISDICTION: {jurisdiction}")

        parts.append(
            f"\n\nCONTEXT (Retrieved Legal Provisions and Precedents):\n{context}"
        )

        parts.append(
            "\n\nPlease provide a comprehensive legal response based on the "
            "above context. Only cite sections and cases that appear in the context."
        )

        return "\n".join(parts)

    def _build_messages(
        self,
        user_message: str,
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build the full message list for the LLM.

        Includes system prompt, conversation history (if any),
        and the current user message with context.
        """
        messages = [{"role": "system", "content": LEGAL_SYSTEM_PROMPT}]

        if history:
            # Include up to last 10 turns to stay within context limits
            for msg in history[-10:]:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        messages.append({"role": "user", "content": user_message})
        return messages

    # ─── OpenAI-Compatible API (vLLM, OpenAI, Generic) ───────────────────

    async def _call_openai_compat(self, messages: list[dict]) -> str:
        """Non-streaming call to any OpenAI-compatible /v1/chat/completions."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    async def _stream_openai_compat(self, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming call to any OpenAI-compatible /v1/chat/completions."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    # ─── Anthropic API ───────────────────────────────────────────────────

    async def _call_anthropic(self, messages: list[dict]) -> str:
        """Non-streaming call to Anthropic Messages API."""
        # Extract system message and convert to Anthropic format
        system_content = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model_name,
            "max_tokens": 2000,
            "system": system_content,
            "messages": api_messages,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.api_url}/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content_blocks = data.get("content", [])
        texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        return "".join(texts)

    async def _stream_anthropic(self, messages: list[dict]) -> AsyncIterator[str]:
        """Streaming call to Anthropic Messages API."""
        system_content = ""
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model_name,
            "max_tokens": 2000,
            "system": system_content,
            "messages": api_messages,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.api_url}/v1/messages",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                        event_type = event.get("type", "")
                        if event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text
                        elif event_type == "message_stop":
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

    # ─── Fallback ────────────────────────────────────────────────────────

    def _fallback_response(self, query: str, context: str) -> dict:
        """
        Generate a response without an LLM — format retrieved context directly.

        Used when no LLM provider is available.
        """
        if not context or context == "No relevant legal provisions or judgments found.":
            answer = (
                "I could not find specific legal provisions matching your query. "
                "Please try rephrasing your question or specifying the legal domain "
                "(e.g., criminal, property, family law)."
            )
        else:
            answer = (
                "Based on the legal provisions and precedents found in the database, "
                "here is the relevant information:\n\n"
                f"{context[:3000]}"
                "\n\n⚖️ This is legal information, not legal advice. "
                "For case-specific advice, consult a qualified advocate."
                "\n\n*Note: This response is compiled directly from retrieved legal text. "
                "An LLM-generated natural language explanation will be available when "
                "an LLM provider is configured.*"
            )

        return {
            "answer": answer,
            "llm_used": False,
            "model": "fallback",
            "provider": "",
            "duration_ms": 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════════

_llm_service: LLMService | None = None


async def get_llm_service() -> LLMService:
    """Get or create the singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
        await _llm_service.initialize()
    return _llm_service