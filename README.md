# axiom

A tiny agent SDK, written to be read. An agent is a chat model that can call
your Python functions ("tools") until it has an answer. axiom does that in
plain Python, with no dependencies, in four short files.

## Try it

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/basic.py
```

```python
from axiom import Agent, ChatModel, tool

@tool
async def get_weather(city: str) -> str:
    """Current weather in a city."""
    return f"{city}: 18°C, sunny"

model = ChatModel(model="gpt-5-mini", api_key="sk-...", base_url="https://api.openai.com/v1")
agent = Agent(model, tools=[get_weather])
answer = await agent.run("What's the weather in Rome and in Oslo?")
```

Any OpenAI-compatible provider works: change `model`, `api_key` and `base_url`.

## Read the code in this order

1. `axiom/tools.py`: turns a Python function into a tool the model can call.
2. `axiom/chat_model.py`: sends the conversation to the model and gets its reply.
3. `axiom/memory.py`: keeps each conversation (session) while the program runs.
4. `axiom/agent.py`: the loop that ties them together.

## How the agent works

```text
your task ──► model ──► text reply? ──► done, return it
                ▲            │
                │            ▼ no: it asked for tools
                └── tool results ◄── run the tools (all at once)
```

Each pass around the loop is one *step*. `max_steps` stops a model that never finishes.

For a typed answer instead of text, pass a dataclass as `output_type`. The agent
gives the model one extra tool, `final_answer`, whose arguments are the
dataclass's fields, and returns an instance once the model calls it correctly.

## Words you'll meet

- **Message**: one entry in the conversation, a dict like
  `{"role": "user", "content": "hi"}`. The `role` says who is speaking:
  - `system`: standing instructions for the model
  - `user`: the person (or the agent, nudging the model)
  - `assistant`: the model
  - `tool`: the result of a tool the model asked for
- **async / await**: `async def` makes a function that can pause while it waits,
  for example for the network, so other work can run meanwhile. `await` waits
  for its result. This is how several tools run at the same time.
- **Decorator**: `@tool` above a function means `get_weather = tool(get_weather)`.
  The name then holds a `Tool` object instead of the plain function.
- **Type hints**: `city: str` says `city` is text. Usually hints are only
  documentation; axiom reads them to tell the model what arguments a tool takes.
- **JSON schema**: a description of the shape of some data, such as
  `{"type": "string"}`. The model reads one per tool to know its arguments.
- **Dataclass**: `@dataclass` above a class writes its `__init__` from the
  fields, so `Weather(city="Rome", celsius=18)` just works.
- **Tool call**: the model doesn't run your functions itself. It replies
  "call `get_weather` with `{"city": "Rome"}`", the agent runs it, and sends the
  result back.

## Run the tests

```bash
uv run pytest
```
