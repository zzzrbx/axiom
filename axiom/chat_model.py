"""Talk to any OpenAI-compatible chat endpoint."""

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any


class ChatModel:
    """A chat model behind an OpenAI-compatible endpoint.

    Switching provider means changing `base_url`, `api_key` and `model`.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 300,
    ) -> None:
        """`timeout` is in seconds, per request."""
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send the conversation and return the model's reply.

        Messages are dicts in OpenAI format, e.g. {"role": "user", "content": "hi"}.
        The reply is an assistant message, exactly as the endpoint returns it,
        so it can be appended to the history and saved as-is.
        """
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools

        # An HTTP POST (giving `data` makes it one), with the body as JSON bytes.
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",  # the standard way to send an API key
                "Content-Type": "application/json",
            },
        )
        # urlopen blocks the whole program while it waits. to_thread runs it on the side,
        # so only this function pauses and other work (e.g. tools) can go on.
        response = await asyncio.to_thread(self._send, request)
        # The endpoint answers {"choices": [{"message": {...}}]}: we asked for one reply, so take the first.
        return response["choices"][0]["message"]

    def _send(self, request: urllib.request.Request) -> dict[str, Any]:
        """Send the request and return the endpoint's JSON response."""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            # Show the provider's explanation, e.g. "Incorrect API key". `from None` hides urllib's own error.
            raise RuntimeError(f"{error.code}: {error.read().decode()}") from None
