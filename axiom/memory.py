"""Keep one message history per session."""

from typing import Any


class Memory:
    """Message histories, one per session, kept in memory while the program runs."""

    def __init__(self) -> None:
        self.sessions: dict[str, list[dict[str, Any]]] = {}

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """A copy of a session's messages, or an empty list if it is new.

        It's a copy, so the session only changes when `save` is called.
        """
        return list(self.sessions.get(session_id, []))

    def save(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Replace a session's messages with a copy of `messages`."""
        self.sessions[session_id] = list(messages)
