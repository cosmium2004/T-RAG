"""
Module 5.2 — LLM Client for T-RAG Generator.
Supports OpenAI, Anthropic, Ollama (local inference), and a template fallback.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Multi-provider LLM client with retry logic.

    Supported providers:
        - 'ollama'    — Local inference via Ollama (OpenAI-compatible API)
        - 'openai'    — OpenAI GPT models (requires API key)
        - 'anthropic' — Anthropic Claude models (requires API key)
        - 'local'     — Template-based fallback (no model, no API key)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ):
        load_dotenv()
        self.provider = provider.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature

        if self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "qwen2:0.5b")
            self.api_base = os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            self.api_key = "ollama"  # Ollama ignores this but SDK requires it
        elif self.provider == "openai":
            self.model = model or "gpt-4"
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            self.api_base = None
        elif self.provider == "anthropic":
            self.model = model or "claude-3-haiku-20240307"
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            self.api_base = None
        else:
            self.provider = "local"
            self.model = "local-template"
            self.api_key = ""
            self.api_base = None

    # ── Generate ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response using the configured LLM.

        Args:
            messages: Chat-format list of {role, content} dicts.

        Returns:
            Generated text string.
        """
        if self.provider == "ollama":
            return self._generate_ollama(messages)
        elif self.provider == "openai":
            return self._generate_openai(messages)
        elif self.provider == "anthropic":
            return self._generate_anthropic(messages)
        else:
            return self._generate_local(messages)

    def _generate_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Generate via Ollama's OpenAI-compatible API."""
        import openai

        client = openai.OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content.strip()
        logger.info(f"Ollama response: {len(text)} chars, model={self.model}")
        return text

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = response.choices[0].message.content.strip()
        logger.info(f"OpenAI response: {len(text)} chars, model={self.model}")
        return text

    def _generate_anthropic(self, messages: List[Dict[str, str]]) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        # Separate system from user messages
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=user_msgs,
        )
        text = response.content[0].text.strip()
        logger.info(f"Anthropic response: {len(text)} chars, model={self.model}")
        return text

    def _generate_local(self, messages: List[Dict[str, str]]) -> str:
        """
        Local fallback: extract the context and generate a template answer.
        This does not call any API — useful for testing without API keys.
        """
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

        # Extract first few context lines
        lines = user_msg.split("\n")
        question = ""
        context_lines = []
        in_context = False
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Context"):
                in_context = True
            elif line.startswith("Instructions:"):
                in_context = False
            elif in_context and line.strip():
                context_lines.append(line.strip())

        if context_lines:
            facts = "; ".join(context_lines[:3])
            answer = (
                f"Based on the available context, {facts}. "
                f"This information is derived from the provided temporal knowledge base."
            )
        else:
            answer = "I don't have sufficient current information to answer this question."

        logger.info(f"Local fallback response: {len(answer)} chars")
        return answer

    # ── Health ─────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        result = {
            "provider": self.provider,
            "model": self.model,
            "has_api_key": bool(self.api_key),
        }

        if self.provider == "ollama":
            try:
                import httpx

                # Ollama's native API (not the /v1 compat layer)
                base = self.api_base.replace("/v1", "")
                resp = httpx.get(f"{base}/api/tags", timeout=3)
                models = [m["name"] for m in resp.json().get("models", [])]
                result["status"] = "healthy"
                result["available_models"] = models
            except Exception as e:
                result["status"] = "unreachable"
                result["error"] = str(e)

        return result
