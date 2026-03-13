"""
NyayaMitra — LLM Service.

Generates structured legal responses by sending retrieved context
to an LLM (via vLLM's OpenAI-compatible API or any OpenAI-compatible endpoint).

The service:
    1. Constructs a legal system prompt
    2. Formats retrieved context into the prompt
    3. Calls the LLM for generation
    4. Parses the structured response

Supports two modes:
    - vLLM (local): Llama 3.1 8B/70B running via vLLM server
    - OpenAI-compatible API: Any provider with /v1/chat/completions endpoint

Usage:
    from app.services.llm_service import get_llm_service

    service = await get_llm_service()
    answer = await service.generate_legal_response(query, context)
"""

from __future__ import annotations

import json
import time

import httpx
import structlog

from app.config import settings

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Legal System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

LEGAL_SYSTEM_PROMPT = """You are NyayaMitra, an AI legal assistant for Indian citizens. You provide accurate, cited legal information based on Indian law.

CRITICAL RULES:
1. ONLY cite sections and cases provided in the CONTEXT below. Never fabricate citations.
2. If the context does not contain enough information, say so honestly.
3. Always include the disclaimer: "This is legal information, not legal advice."
4. Explain legal concepts in simple, plain language that a non-lawyer can understand.
5. Be jurisdiction-aware — note if the law varies by state.

RESPONSE FORMAT:
Structure your response with these sections:

**Answer:** A clear, plain-language explanation answering the user's question (2-4 paragraphs).

**Applicable Law:**
- For each relevant section from the context, cite: Act Name, Section Number, and briefly explain what it says.

**Key Precedents:**
- For each relevant case from the context, cite: Case Name (Year), Court, Citation, and explain its significance.

**What You Should Do (Procedure):**
- If applicable, provide numbered step-by-step practical guidance.

**Important Notes:**
- Any jurisdiction-specific variations or caveats.
- When to consult a human lawyer.

Keep your answer focused, accurate, and grounded in the provided context."""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Service
# ═══════════════════════════════════════════════════════════════════════════════


class LLMService:
    """
    Generates legal responses using an LLM.

    Connects to vLLM or any OpenAI-compatible chat completions API.
    """

    def __init__(self):
        self.api_url: str = ""
        self.model_name: str = ""
        self.api_key: str = ""
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the LLM service.

        Tries in order:
        1. vLLM local server (if running on configured port)
        2. OpenAI-compatible API (if OPENAI_API_KEY or ANTHROPIC_API_KEY is set)
        """
        if self._initialized:
            return

        # Option 1: Check if vLLM is running locally
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
                        self._initialized = True
                        logger.info(
                            "llm_service_initialized",
                            provider="vllm",
                            model=self.model_name,
                            url=self.api_url,
                        )
                        return
        except Exception:
            pass

        # Option 2: Check for OpenAI API key
        if settings.OPENAI_API_KEY:
            self.api_url = "https://api.openai.com/v1"
            self.model_name = settings.OPENAI_MODEL
            self.api_key = settings.OPENAI_API_KEY
            self._initialized = True
            logger.info(
                "llm_service_initialized",
                provider="openai",
                model=self.model_name,
            )
            return

        # Option 3: Check for any generic OpenAI-compatible endpoint
        if settings.LLM_API_URL and settings.LLM_API_KEY:
            self.api_url = settings.LLM_API_URL
            self.model_name = settings.LLM_MODEL_NAME or "default"
            self.api_key = settings.LLM_API_KEY
            self._initialized = True
            logger.info(
                "llm_service_initialized",
                provider="generic",
                model=self.model_name,
                url=self.api_url,
            )
            return

        # No LLM available — service will use fallback generation
        logger.warning(
            "llm_service_no_provider",
            message="No LLM provider found. Set OPENAI_API_KEY, or start vLLM, or set LLM_API_URL + LLM_API_KEY. Falling back to context-only responses.",
        )
        self._initialized = True

    async def generate_legal_response(
        self,
        query: str,
        context: str,
        jurisdiction: str | None = None,
    ) -> dict:
        """
        Generate a structured legal response.

        Args:
            query: The user's legal question
            context: Formatted retrieval results (from RetrievalService.format_context_for_llm)
            jurisdiction: User's state/jurisdiction if provided

        Returns:
            dict with keys: answer, applicable_law, precedents, procedure, jurisdiction_notes
        """
        if not self._initialized:
            await self.initialize()

        # If no LLM provider available, return a formatted version of context
        if not self.api_url:
            return self._fallback_response(query, context)

        start = time.time()

        # Build the user message with context
        user_message = self._build_user_message(query, context, jurisdiction)

        try:
            response_text = await self._call_llm(user_message)
            duration_ms = round((time.time() - start) * 1000, 2)

            logger.info(
                "llm_generation_complete",
                query_length=len(query),
                context_length=len(context),
                response_length=len(response_text),
                duration_ms=duration_ms,
            )

            return {
                "answer": response_text,
                "llm_used": True,
                "model": self.model_name,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            logger.error("llm_generation_error", error=str(e))
            return self._fallback_response(query, context)

    async def _call_llm(self, user_message: str) -> str:
        """
        Call the LLM chat completions API.

        Works with vLLM, OpenAI, and any OpenAI-compatible endpoint.
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
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

        # Extract the response text
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")

        return ""

    def _build_user_message(
        self,
        query: str,
        context: str,
        jurisdiction: str | None = None,
    ) -> str:
        """Build the user message with query and retrieved context."""
        parts = []

        parts.append(f"USER QUESTION: {query}")

        if jurisdiction:
            parts.append(f"\nJURISDICTION: {jurisdiction}")

        parts.append(f"\n\nCONTEXT (Retrieved Legal Provisions and Precedents):\n{context}")

        parts.append(
            "\n\nPlease provide a comprehensive legal response based on the above context. "
            "Only cite sections and cases that appear in the context."
        )

        return "\n".join(parts)

    def _fallback_response(self, query: str, context: str) -> dict:
        """
        Generate a response without an LLM — just format the retrieved context.

        Used when no LLM provider is available.
        """
        if not context or context == "No relevant legal provisions or judgments found.":
            answer = (
                "I could not find specific legal provisions matching your query. "
                "Please try rephrasing or specifying the legal domain."
            )
        else:
            # Format a readable response from context
            answer = (
                "Based on the legal provisions and precedents found in the database, "
                "here is the relevant information:\n\n"
                f"{context[:3000]}"
                "\n\nNote: This response is compiled directly from retrieved legal text. "
                "For a more natural explanation, an LLM will be connected in the next update."
            )

        return {
            "answer": answer,
            "llm_used": False,
            "model": "fallback",
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