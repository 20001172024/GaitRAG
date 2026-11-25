"""Pluggable LLM client.

OpenAI and HuggingFace Inference Providers both expose the OpenAI
chat-completions API (POST {base_url}/chat/completions), so one small httpx
client covers both. `LLM_PROVIDER` picks the base URL and key; the prompt is
identical, which keeps the provider comparison in the evaluation fair.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "huggingface": "https://router.huggingface.co/v1",
}


class LLMError(RuntimeError):
    """The LLM call failed (network, auth, quota or a malformed response)."""


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient(Protocol):
    """What the generator needs from a language model."""

    model: str

    def complete(self, messages: list[Message]) -> str:
        """Return the assistant's reply text."""
        ...


class OpenAICompatibleClient:
    """Chat completions over HTTP for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
            transport=transport,
        )

    def complete(self, messages: list[Message]) -> str:
        """POST /chat/completions and return the first choice's text."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            r = self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM returned HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"LLM call failed: {type(exc).__name__}") from exc
        if not isinstance(content, str):
            raise LLMError("LLM returned no text")
        return content


class FakeLLM:
    """Scripted LLM for tests: replies come from a function of the messages."""

    def __init__(self, reply: Callable[[list[Message]], str], model: str = "fake"):
        self.model = model
        self._reply = reply
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> str:
        """Record the call and return the scripted reply."""
        self.calls.append(messages)
        return self._reply(messages)


def build_llm(settings: Settings) -> LLMClient:
    """Create the configured client.

    Raises:
        LLMError: if the provider's key is not configured.
    """
    secret = settings.openai_api_key if settings.llm_provider == "openai" else settings.hf_api_token
    if secret is None or not secret.get_secret_value():
        raise LLMError(f"No API key configured for LLM provider {settings.llm_provider!r}")
    return OpenAICompatibleClient(
        base_url=PROVIDER_BASE_URLS[settings.llm_provider],
        api_key=secret.get_secret_value(),
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
    )
