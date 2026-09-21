import asyncio
import os
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union

import litellm
from litellm.types.utils import ModelResponse

from codeact_runtime.cache import SQLiteLLMCache

litellm.disable_aiohttp_transport = True


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResult:
    text: str
    finish_reason: str
    cache_hit: bool
    token_usage: TokenUsage


class LlmProxy(Protocol):
    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> LLMResult: ...


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, bytes):
        return {"__bytes__": obj.hex()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump(mode="json"))  # type: ignore[no-any-return]
    return str(obj)


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LiteLlmProxy:
    model: str
    cache: Optional[SQLiteLLMCache] = None
    cache_enabled: bool = True
    max_retries: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 30.0
    jitter_s: float = 0.2
    max_concurrent_requests: int = 8
    timeout_s: float = 60.0
    retry_status_codes: list[int] = field(default_factory=list)
    retry_timeout_sec: int = 5
    temperature: float | None = 1.0
    max_tokens: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_sem", asyncio.Semaphore(self.max_concurrent_requests)
        )

    def _cache_key(self, payload: dict[str, Any]) -> str:
        return _hash_payload(payload)

    def _build_payload(self, **kwargs: Any) -> dict[str, Any]:
        payload = {"model": self.model}
        for key, value in kwargs.items():
            if value is None:
                continue
            if key == "messages":
                payload[key] = value or []  # type: ignore
                continue
            payload[key] = value
        if "messages" not in payload:
            payload["messages"] = []  # type: ignore
        # Optional passthrough of provider-specific params (e.g. Qwen3
        # {"chat_template_kwargs": {"enable_thinking": false}} to match /nothink
        # serving). Set CODEACT_EXTRA_BODY_JSON to a JSON object; forwarded to the
        # OpenAI-compatible endpoint via litellm's extra_body.
        _eb = os.environ.get("CODEACT_EXTRA_BODY_JSON")
        if _eb:
            try:
                payload["extra_body"] = json.loads(_eb)
            except (json.JSONDecodeError, ValueError):
                pass
        return payload

    async def _call_with_backoff(
        self, payload: dict[str, Any], timeout: Optional[Union[float, int]]
    ) -> ModelResponse:
        attempt = 0
        last_err: Optional[BaseException] = None

        # Default to 120s or whatever your config is
        effective_timeout = self.timeout_s if timeout is None else timeout
        while attempt <= self.max_retries:
            try:
                async with self._sem:  # type: ignore
                    return await asyncio.wait_for(  # type: ignore
                        litellm.acompletion(**payload, timeout=effective_timeout),
                        timeout=effective_timeout,
                    )

            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError) as e:
                last_err = e
                # print(f"LLM Timeout (Attempt {attempt+1}): {repr(e)}")
            except Exception as e:
                last_err = e
                # print(f"LLM Error (Attempt {attempt+1}): {repr(e)}")

            if attempt >= self.max_retries:
                break

            # [FIX] SLEEP OUTSIDE THE LOCK
            # Now other workers can use the connection slot while this task sleeps.
            backoff = min(
                self.backoff_max_s,
                self.backoff_base_s * (2**attempt),
            )
            jitter = random.random() * self.jitter_s
            await asyncio.sleep(backoff + jitter)
            attempt += 1

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_err}"
        ) from last_err

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            # a None seed is dropped by _build_payload, leaving the server's own sampling
            "seed": seed if seed is not None else self.seed,
        }
        if extra:
            payload.update(extra)

        payload = self._build_payload(**payload)

        cache_hit = False
        response = None
        if self.cache_enabled and self.cache is not None:
            key = self._cache_key(payload)
            cached = await self.cache.get(key)
            if cached is not None:
                response = ModelResponse(**cached)
                cache_hit = True
        if response is None:
            response = await self._call_with_backoff(payload, self.timeout_s)
            if self.cache_enabled and self.cache is not None:
                await self.cache.set(
                    self._cache_key(payload),
                    request_payload=payload,
                    response_payload=response.model_dump(),
                )

        assert isinstance(response, ModelResponse)
        raw_usage = response.usage  # type: ignore
        assert raw_usage is not None
        token_usage = TokenUsage(
            prompt_tokens=raw_usage.prompt_tokens,
            completion_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
        )

        response_raw = response.model_dump()
        message = _extract_text(response_raw)

        return LLMResult(
            text=message,
            cache_hit=cache_hit,
            token_usage=token_usage,
            finish_reason=response.choices[0].finish_reason or "unknown",
        )


def _extract_text(raw_response: dict[str, Any]) -> str:
    """Best-effort extraction of assistant content from common provider schemas."""
    # OpenAI-like
    try:
        return raw_response["choices"][0]["message"]["content"]
    except Exception:
        pass

    # Some providers use `text`
    try:
        return raw_response["choices"][0]["text"]
    except Exception:
        pass

    # Fallback: dump JSON
    return json.dumps(raw_response, ensure_ascii=False)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    pr = LiteLlmProxy("gpt-5-nano")
    response = asyncio.run(
        pr.complete(messages=[{"role": "user", "content": "just checking"}])
    )
    print(response)
