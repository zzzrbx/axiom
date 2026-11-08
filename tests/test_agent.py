"""The agent loop, driven by a scripted fake model (no network)."""

import asyncio
import copy
import json
from dataclasses import dataclass

import pytest

from axiom.agent import Agent
from axiom.memory import Memory
from axiom.tools import tool


class FakeModel:
    """Stands in for ChatModel: returns scripted replies, records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
        return self.replies.pop(0)


def text(content):
    return {"role": "assistant", "content": content}


def tool_calls(*calls):
    """An assistant reply asking for tools: tool_calls(("name", {args}), ...)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for i, (name, arguments) in enumerate(calls)
        ],
    }


@tool
async def get_weather(city: str) -> str:
    """Current weather in a city."""
    return f"{city}: 18°C, sunny"


@dataclass
class Weather:
    city: str
    celsius: int


def run(agent, *args, **kwargs):
    return asyncio.run(agent.run(*args, **kwargs))


def test_plain_text_reply_ends_the_run():
    model = FakeModel(text("Hi!"))
    agent = Agent(model, verbose=False)

    assert run(agent, "hello") == "Hi!"
    assert model.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_instructions_become_the_system_message():
    model = FakeModel(text("Ciao!"))
    agent = Agent(model, instructions="Answer in Italian.", verbose=False)

    run(agent, "hello")

    assert model.calls[0]["messages"][0] == {
        "role": "system",
        "content": "Answer in Italian.",
    }


def test_tool_result_goes_back_to_the_model():
    model = FakeModel(tool_calls(("get_weather", {"city": "Rome"})), text("Sunny."))
    agent = Agent(model, tools=[get_weather], verbose=False)

    assert run(agent, "Weather in Rome?") == "Sunny."
    assert model.calls[0]["tools"] == [get_weather.schema()]
    assert model.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "Rome: 18°C, sunny",
    }


def test_tools_in_one_reply_run_in_parallel():
    a_started, b_started = asyncio.Event(), asyncio.Event()

    @tool
    async def a() -> str:
        """Waits for b."""
        a_started.set()
        await asyncio.wait_for(b_started.wait(), timeout=1)
        return "a done"

    @tool
    async def b() -> str:
        """Waits for a."""
        b_started.set()
        await asyncio.wait_for(a_started.wait(), timeout=1)
        return "b done"

    model = FakeModel(tool_calls(("a", {}), ("b", {})), text("Both done."))
    agent = Agent(model, tools=[a, b], verbose=False)

    run(agent, "go")

    # Run one after the other, `a` would time out waiting for `b`.
    assert [m["content"] for m in model.calls[1]["messages"][-2:]] == ["a done", "b done"]


def test_unknown_tool_is_reported_to_the_model():
    model = FakeModel(tool_calls(("teleport", {})), text("Sorry."))
    agent = Agent(model, verbose=False)

    run(agent, "go")

    assert model.calls[1]["messages"][-1]["content"].startswith("Error:")


def test_final_answer_returns_an_output_type_instance():
    model = FakeModel(tool_calls(("final_answer", {"city": "Rome", "celsius": 18})))
    agent = Agent(model, verbose=False)

    answer = run(agent, "Weather in Rome?", output_type=Weather)

    assert answer == Weather(city="Rome", celsius=18)
    assert model.calls[0]["tools"][-1] == {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Give your final answer.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "celsius": {"type": "integer"}},
                "required": ["city", "celsius"],
            },
        },
    }


def test_invalid_final_answer_is_sent_back_for_another_try():
    model = FakeModel(
        tool_calls(("final_answer", {"city": "Rome", "celsius": "warm"})),
        tool_calls(("final_answer", {"city": "Rome", "celsius": 18})),
    )
    agent = Agent(model, verbose=False)

    answer = run(agent, "Weather in Rome?", output_type=Weather)

    assert answer == Weather(city="Rome", celsius=18)
    assert model.calls[1]["messages"][-1]["content"].startswith("Error:")


def test_final_answer_with_a_missing_field_is_sent_back():
    model = FakeModel(
        tool_calls(("final_answer", {"city": "Rome"})),
        tool_calls(("final_answer", {"city": "Rome", "celsius": 18})),
    )
    agent = Agent(model, verbose=False)

    run(agent, "Weather in Rome?", output_type=Weather)

    assert "celsius" in model.calls[1]["messages"][-1]["content"]


@dataclass
class Temperature:
    celsius: float


def test_a_whole_number_fits_a_float_field():
    model = FakeModel(tool_calls(("final_answer", {"celsius": 18})))
    agent = Agent(model, verbose=False)

    assert run(agent, "Temperature?", output_type=Temperature) == Temperature(celsius=18)


def test_true_is_not_accepted_as_a_number():
    model = FakeModel(
        tool_calls(("final_answer", {"city": "Rome", "celsius": True})),
        tool_calls(("final_answer", {"city": "Rome", "celsius": 18})),
    )
    agent = Agent(model, verbose=False)

    run(agent, "Weather in Rome?", output_type=Weather)

    assert model.calls[1]["messages"][-1]["content"].startswith("Error:")


@dataclass
class Forecast:
    celsius: int

    def __post_init__(self):
        if self.celsius < -90:
            raise ValueError("colder than anywhere on Earth")


def test_value_error_in_output_type_goes_back_to_the_model():
    model = FakeModel(
        tool_calls(("final_answer", {"celsius": -100})),
        tool_calls(("final_answer", {"celsius": 18})),
    )
    agent = Agent(model, verbose=False)

    run(agent, "Forecast?", output_type=Forecast)

    assert "colder" in model.calls[1]["messages"][-1]["content"]


def test_text_reply_with_output_type_asks_for_final_answer():
    model = FakeModel(
        text("It's 18°C in Rome."),
        tool_calls(("final_answer", {"city": "Rome", "celsius": 18})),
    )
    agent = Agent(model, verbose=False)

    answer = run(agent, "Weather in Rome?", output_type=Weather)

    assert answer == Weather(city="Rome", celsius=18)
    assert "final_answer" in model.calls[1]["messages"][-1]["content"]


def test_a_session_remembers_earlier_runs():
    model = FakeModel(text("Nice to meet you."), text("You're Ada."))
    agent = Agent(model, verbose=False)

    run(agent, "I'm Ada.", session_id="s1")
    run(agent, "Who am I?", session_id="s1")

    assert [m["content"] for m in model.calls[1]["messages"]] == [
        "I'm Ada.",
        "Nice to meet you.",
        "Who am I?",
    ]


def test_sessions_are_kept_apart():
    model = FakeModel(text("Nice to meet you."), text("I don't know."))
    agent = Agent(model, memory=Memory(), verbose=False)

    run(agent, "I'm Ada.", session_id="s1")
    run(agent, "Who am I?", session_id="s2")

    assert model.calls[1]["messages"] == [{"role": "user", "content": "Who am I?"}]


def test_run_stops_after_max_steps():
    endless = [tool_calls(("get_weather", {"city": "Rome"})) for _ in range(3)]
    model = FakeModel(*endless)
    agent = Agent(model, tools=[get_weather], max_steps=3, verbose=False)

    with pytest.raises(RuntimeError):
        run(agent, "Weather forever")
    assert len(model.calls) == 3
