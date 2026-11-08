"""Turning functions into tools, and running them."""

import asyncio

import pytest

from axiom.tools import tool


@tool
async def get_weather(city: str, days: int = 1) -> str:
    """Weather forecast for a city.

    Args:
        city: City name, e.g. "Rome".
        days: How many days ahead.
    """
    return f"{city}: sunny for {days} day(s)"


@tool
async def broken() -> str:
    """Always fails."""
    raise ValueError("no network")


def test_name_and_whole_docstring_become_the_description():
    assert get_weather.name == "get_weather"
    assert get_weather.description == (
        "Weather forecast for a city.\n"
        "\n"
        "    Args:\n"
        '        city: City name, e.g. "Rome".\n'
        "        days: How many days ahead.\n"
        "    "
    )


def test_parameters_come_from_type_hints():
    assert get_weather.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    }


def test_unsupported_type_hint_is_a_clear_error():
    async def tag(labels: list) -> str:
        return ""

    with pytest.raises(TypeError, match="labels"):
        tool(tag)


def test_missing_type_hint_is_a_clear_error():
    async def tag(label) -> str:
        return ""

    with pytest.raises(TypeError, match="label"):
        tool(tag)


def test_schema_is_in_openai_tools_format():
    schema = get_weather.schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": get_weather.description,
            "parameters": get_weather.parameters,
        },
    }


def test_run_returns_the_result_as_text():
    result = asyncio.run(get_weather.run({"city": "Rome", "days": 2}))

    assert result == "Rome: sunny for 2 day(s)"


def test_run_returns_errors_as_text():
    result = asyncio.run(broken.run({}))

    assert result == "Error: ValueError: no network"
