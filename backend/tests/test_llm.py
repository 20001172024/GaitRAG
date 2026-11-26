"""Tests for the OpenAI-compatible LLM client (no network: mock transport)."""

import json

import httpx
import pytest

from app.config import Settings
from app.rag.llm import LLMError, Message, OpenAICompatibleClient, build_llm


def _client(handler) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        "https://api.example.test/v1", "sk-test", "gpt-test", transport=httpx.MockTransport(handler)
    )


def test_sends_chat_completion_request_and_returns_text() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi [P001, p.2]"}}]})

    reply = _client(handler).complete([Message("system", "s"), Message("user", "u")])
    assert reply == "Hi [P001, p.2]"
    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "gpt-test"
    assert seen["body"]["temperature"] == 0.0
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": "bad key"}),
        httpx.Response(429, json={"error": "rate limit"}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="not json"),
    ],
)
def test_failures_raise_llm_error(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        _client(lambda request: response).complete([Message("user", "u")])


def test_network_error_raises_llm_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(LLMError, match="ConnectError"):
        _client(boom).complete([Message("user", "u")])


def test_build_llm_requires_a_key() -> None:
    with pytest.raises(LLMError, match="No API key"):
        build_llm(Settings(llm_provider="openai", openai_api_key=None))


def test_build_llm_picks_provider_base_url() -> None:
    llm = build_llm(Settings(llm_provider="huggingface", hf_api_token="hf_x", llm_model="m/x"))
    assert str(llm._client.base_url) == "https://router.huggingface.co/v1/"
    assert llm.model == "m/x"
