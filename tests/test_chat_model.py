"""ChatModel against a fake OpenAI-compatible endpoint (no network)."""

import asyncio
import io
import json
import urllib.error
import urllib.request

import pytest

from axiom.chat_model import ChatModel

URL = "https://api.example.com/v1"
REPLY = {"role": "assistant", "content": "Hi!"}


def fake_endpoint(monkeypatch, status=200, body=None):
    """Replace urlopen with a fake server; return the list of (request, timeout) it saw."""
    seen = []
    body = body or {"choices": [{"message": REPLY}]}

    def fake_urlopen(request, timeout):
        seen.append((request, timeout))
        response = io.BytesIO(json.dumps(body).encode())
        if status >= 400:
            raise urllib.error.HTTPError(request.full_url, status, "error", {}, response)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_timeout_reaches_the_request(monkeypatch):
    seen = fake_endpoint(monkeypatch)

    asyncio.run(ChatModel("gpt-x", api_key="k", base_url=URL, timeout=7).chat([]))

    request, timeout = seen[0]
    assert timeout == 7


def test_returns_the_assistant_message(monkeypatch):
    fake_endpoint(monkeypatch)
    model = ChatModel("gpt-x", api_key="sk-test", base_url=URL)

    reply = asyncio.run(model.chat([{"role": "user", "content": "hello"}]))

    assert reply == {"role": "assistant", "content": "Hi!"}


def test_posts_model_and_messages_to_chat_completions(monkeypatch):
    seen = fake_endpoint(monkeypatch)
    model = ChatModel("llama3", base_url="http://localhost:11434/v1/", api_key="k")
    messages = [{"role": "user", "content": "hello"}]

    asyncio.run(model.chat(messages))

    request, _ = seen[0]
    assert request.full_url == "http://localhost:11434/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer k"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data) == {"model": "llama3", "messages": messages}


def test_sends_tools_when_given(monkeypatch):
    seen = fake_endpoint(monkeypatch)
    model = ChatModel("gpt-x", api_key="k", base_url=URL)
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    asyncio.run(model.chat([], tools=tools))

    request, _ = seen[0]
    assert json.loads(request.data)["tools"] == tools


def test_http_error_shows_the_providers_message(monkeypatch):
    fake_endpoint(monkeypatch, status=401, body={"error": {"message": "Incorrect API key"}})

    with pytest.raises(RuntimeError, match="401.*Incorrect API key"):
        asyncio.run(ChatModel("gpt-x", api_key="k", base_url=URL).chat([]))
