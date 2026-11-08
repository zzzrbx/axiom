"""The agent loop: call the model, run tools, repeat until done."""

import asyncio
import json
from typing import Any

from .chat_model import ChatModel
from .memory import Memory
from .tools import Tool, parameters_schema

FINAL_ANSWER_ACCEPTED = "Answer accepted."  # what _run_tool returns for a valid final_answer


class Agent:
    """A chat model that can use tools, with one memory per session."""

    def __init__(
        self,
        model: ChatModel,
        tools: list[Tool] | None = None,
        instructions: str = "",
        memory: Memory | None = None,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> None:
        """`instructions` becomes the system message of every new session.

        `max_steps` caps the model replies per run, so a confused model cannot loop forever.
        """
        self.model = model
        self.tools = {tool.name: tool for tool in tools or []}
        self.instructions = instructions
        self.memory = memory or Memory()
        self.max_steps = max_steps
        self.verbose = verbose

    async def run(
        self,
        task: str,
        session_id: str = "default",
        output_type: type | None = None,
    ) -> Any:
        """Work on `task` until done, and return the answer.

        Without `output_type`, the run ends when the model replies with text.
        With `output_type` (a dataclass), it ends when the model calls
        `final_answer` with arguments that match its fields, and an
        `output_type` instance is returned. The session is saved after every step.

        Example:
            @dataclass
            class Weather:
                city: str
                celsius: float

            weather = await agent.run("Weather in Rome?", output_type=Weather)
            weather.celsius  # 18.0
        """
        messages = self.memory.load(session_id)
        if not messages and self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": task})

        tools = [tool.schema() for tool in self.tools.values()]
        if output_type:
            tools.append(self._final_answer_tool(output_type))

        # One step = one model reply, plus running the tools it asked for.
        for _ in range(self.max_steps):
            reply = await self.model.chat(messages, tools)
            messages.append(reply)
            self._print(reply)
            # The tools the model asked for, if any, e.g.
            # [{"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]
            calls = reply.get("tool_calls") or []

            # A text reply: done, unless we're waiting for a final_answer.
            if not calls:
                if output_type is None:
                    self.memory.save(session_id, messages)
                    return reply["content"]
                messages.append({"role": "user", "content": "Answer by calling final_answer."})
                self.memory.save(session_id, messages)
                continue

            # Tool calls: run them all, and send every result back.
            results = await self._run_tools(calls, output_type)
            messages.extend(results)
            self.memory.save(session_id, messages)

            # A valid final_answer: done. _run_tool already checked it, so building it again cannot fail.
            for call, result in zip(calls, results):  # zip pairs each call with its result
                if call["function"]["name"] == "final_answer" and result["content"] == FINAL_ANSWER_ACCEPTED:
                    return build_answer(output_type, json.loads(call["function"]["arguments"]))

        raise RuntimeError(f"No answer after {self.max_steps} steps")

    async def _run_tools(
        self, calls: list[dict[str, Any]], output_type: type | None
    ) -> list[dict[str, Any]]:
        """Run all requested tools in parallel; one tool message per call."""
        # gather runs all the calls at the same time, waits for every one,
        # and returns their results in the same order as `calls`.
        results = await asyncio.gather(*[self._run_tool(call, output_type) for call in calls])
        messages = []
        for call, result in zip(calls, results):
            message = {"role": "tool", "tool_call_id": call["id"], "content": result}
            self._print(message)
            messages.append(message)
        return messages

    async def _run_tool(self, call: dict[str, Any], output_type: type | None) -> str:
        """Run one tool call and return its result as text."""
        name = call["function"]["name"]
        try:
            arguments = json.loads(call["function"]["arguments"])  # the model sends JSON text
        except json.JSONDecodeError as error:
            return f"Error: arguments are not valid JSON: {error}"

        if name == "final_answer" and output_type:
            try:
                build_answer(output_type, arguments)
            except (TypeError, ValueError) as error:
                return f"Error: {error}"
            return FINAL_ANSWER_ACCEPTED

        if name not in self.tools:
            return f"Error: there is no tool called {name}"
        return await self.tools[name].run(arguments)

    def _final_answer_tool(self, output_type: type) -> dict[str, Any]:
        """The `final_answer` tool schema, built from the fields of `output_type`."""
        return {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Give your final answer.",
                "parameters": parameters_schema(output_type),  # a dataclass's fields are its __init__ arguments
            },
        }

    def _print(self, message: dict[str, Any]) -> None:
        """Print one message readably, if `verbose`."""
        if not self.verbose:
            return
        if message["role"] == "tool":
            print(f"  ← {message['content']}")
            return
        for call in message.get("tool_calls") or []:
            print(f"  → {call['function']['name']}({call['function']['arguments']})")
        if message.get("content"):
            print(f"{message['role']}: {message['content']}")


def build_answer(output_type: type, arguments: dict[str, Any]) -> Any:
    """Make an `output_type` instance, checking the type of every field.

    Raises TypeError if a field is missing, unknown, or of the wrong type.
    The dataclass itself may raise ValueError, e.g. in `__post_init__`.
    """
    answer = output_type(**arguments)  # missing or unknown fields raise TypeError
    for name, value in arguments.items():
        expected = output_type.__annotations__[name]  # the field's type hint, e.g. str
        if not matches(value, expected):
            raise TypeError(f"{name} should be {expected.__name__}, got {value!r}")
    return answer


def matches(value: Any, expected: type) -> bool:
    """Is `value`, read from JSON, of the `expected` type?"""
    if expected is float:
        return type(value) in (int, float)  # JSON has one kind of number: 18 is a fine float
    return type(value) is expected  # not isinstance(): bool is a kind of int, so True would pass
