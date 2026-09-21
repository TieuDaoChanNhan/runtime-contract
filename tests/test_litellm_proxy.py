import asyncio

import pytest

from codeact_runtime.cache import SQLiteLLMCache
from codeact_runtime.llm import LiteLlmProxy

litellm = pytest.importorskip("litellm")
ModelResponse = pytest.importorskip("litellm.types.utils").ModelResponse


def _build_response(
    *,
    content: str = "ok",
    finish_reason: str = "stop",
    prompt_tokens: int = 2,
    completion_tokens: int = 3,
    total_tokens: int = 5,
):
    return ModelResponse(
        id="chatcmpl-123",
        object="chat.completion",
        created=123,
        model="gpt-4",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    )


@pytest.mark.asyncio
async def test_complete_uses_cache(tmp_path, monkeypatch):
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return _build_response(content="cached")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    cache = SQLiteLLMCache(tmp_path / "llm_cache.sqlite")
    proxy = LiteLlmProxy("gpt-4", cache=cache, cache_enabled=True)
    messages = [{"role": "user", "content": "hello"}]

    first = await proxy.complete(messages=messages)
    second = await proxy.complete(messages=messages)

    assert len(calls) == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == "cached"


@pytest.mark.asyncio
async def test_complete_retries_with_backoff(monkeypatch):
    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("boom")
        return _build_response(content="retry-ok")

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    proxy = LiteLlmProxy(
        "gpt-4",
        max_retries=1,
        backoff_base_s=0.0,
        backoff_max_s=0.0,
        jitter_s=0.0,
    )

    result = await proxy.complete(messages=[{"role": "user", "content": "hi"}])

    assert attempts == 2
    assert result.text == "retry-ok"
