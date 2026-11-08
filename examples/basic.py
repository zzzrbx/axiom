"""One small agent with one tool."""

import asyncio
import os

from axiom import Agent, ChatModel, tool


@tool  # get_weather becomes a Tool; its name, docstring and type hints describe it to the model
async def get_weather(city: str) -> str:
    """Current weather in a city."""
    return f"{city}: 18°C, sunny"  # a real tool would call a weather API


async def main() -> None:
    model = ChatModel(
        model="gpt-5-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
    )
    agent = Agent(model, tools=[get_weather])
    answer = await agent.run("What's the weather in Rome and in Oslo?")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
