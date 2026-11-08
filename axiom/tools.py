"""Turn a Python function into a tool the model can call."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Python type -> JSON schema type
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    """A function the model can call, plus the schema that describes it."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema of the function's arguments
    function: Callable  # the original function; it must be `async def`

    def schema(self) -> dict[str, Any]:
        """This tool in the OpenAI `tools` format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, arguments: dict[str, Any]) -> str:
        """Call the function and return its result as text.

        Errors are returned as text too, so the model can read them and recover.
        """
        try:
            result = await self.function(**arguments)
        except Exception as error:
            return f"Error: {type(error).__name__}: {error}"
        return str(result)


def tool(function: Callable) -> Tool:
    """Build a Tool from an async function's signature and docstring.

    Writing `@tool` above `async def get_weather(...)` means `get_weather = tool(get_weather)`:
    the name now holds a Tool, not the plain function.
    """
    return Tool(
        name=function.__name__,
        description=function.__doc__ or "",
        parameters=parameters_schema(function),
        function=function,
    )


def parameters_schema(function: Callable) -> dict[str, Any]:
    """JSON schema of a function's arguments, from its type hints."""
    properties = {}
    required = []
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.annotation not in JSON_TYPES:
            raise TypeError(f"{function.__name__}: the type hint of `{name}` must be str, int, float or bool")
        properties[name] = {"type": JSON_TYPES[parameter.annotation]}
        if parameter.default is parameter.empty:  # no default, so the model must give it
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}
