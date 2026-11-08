"""Sessions kept in memory."""

from axiom.memory import Memory

MESSAGES = [
    {"role": "user", "content": "Weather in Rome?"},
    {"role": "assistant", "content": "18°C and sunny."},
]


def test_new_session_is_empty():
    assert Memory().load("new") == []


def test_saved_messages_load_back():
    memory = Memory()

    memory.save("chat", MESSAGES)

    assert memory.load("chat") == MESSAGES


def test_sessions_are_kept_apart():
    memory = Memory()

    memory.save("a", MESSAGES)
    memory.save("b", [{"role": "user", "content": "hi"}])

    assert memory.load("a") == MESSAGES
    assert memory.load("b") == [{"role": "user", "content": "hi"}]


def test_changing_loaded_messages_does_not_change_memory():
    memory = Memory()
    memory.save("chat", [{"role": "user", "content": "hi"}])

    memory.load("chat").append({"role": "user", "content": "unsaved"})

    assert memory.load("chat") == [{"role": "user", "content": "hi"}]


def test_changing_saved_messages_does_not_change_memory():
    memory = Memory()
    messages = []
    memory.save("chat", messages)

    messages.append({"role": "user", "content": "unsaved"})

    assert memory.load("chat") == []
