import asyncio
from pathlib import Path

from dotenv import load_dotenv

from codeact_runtime.cache import SQLiteLLMCache
from codeact_runtime.codeact.agent import CodeAct
from codeact_runtime.codeact.events import (
    AsyncQueuedEventBus,
    MessageLogger,
    ModelTokenLogger,
)
from codeact_runtime.codeact.html_trace_logger import HtmlTraceListener
from codeact_runtime.codeact.json_model_event_logger import JsonModelEventLogger
from codeact_runtime.codeact.json_trace_logger import JsonTraceLogger
from codeact_runtime.codeact.tool import Tool
from codeact_runtime.llm import LiteLlmProxy


class WeatherTool(Tool):
    name: str = "weather"
    doc: str = "Returns the weather in the given country"
    arg_doc: dict[str, str] = {"country": "the country to return weather in"}

    def __init__(self):
        super().__init__()

    async def run(self, country: str) -> float:
        return 5.0


load_dotenv()


async def main():
    model_name = "gpt-4o-mini"
    cache = SQLiteLLMCache(Path("cache/llm_cache.sqlite"))
    llm_proxy = LiteLlmProxy(model_name, cache=cache, cache_enabled=True)

    # response = await caching_proxy.acompletion(messages)

    # print(response.choices[0].message.content)

    bus = AsyncQueuedEventBus(
        listeners=[
            MessageLogger(),
            ModelTokenLogger(),
            HtmlTraceListener("trace.html"),
            JsonTraceLogger("trace.json"),
            JsonModelEventLogger("model_events.json"),
        ]
    )

    async with bus:
        agent = CodeAct(
            llm_proxy=llm_proxy, max_num_turns=5, tools=[WeatherTool()], bus=bus
        )

        await agent.run("Print the weather in Slovenia")

    # await close_litellm_async_clients()  # ✅ correct


if __name__ == "__main__":
    asyncio.run(main())
