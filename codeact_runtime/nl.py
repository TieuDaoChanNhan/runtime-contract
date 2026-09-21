from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from codeact_runtime.families.base import TaskData
from codeact_runtime.llm import LlmProxy


class NLWrapper(BaseModel):
    """Optional natural-language wrapper for a task.

    This is not required for the "execution necessity" guarantee; it exists to
    add surface-form variety in the dataset while keeping the underlying instance
    deterministic.
    """

    title: str = Field(..., min_length=1, max_length=120)
    instructions: str = Field(..., min_length=1, max_length=2000)
    output_format: str = Field(..., min_length=1, max_length=400)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    m = _JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _fallback_wrapper(task: TaskData) -> Dict[str, Any]:
    fam = task.family
    title = f"{fam}: solve the task"
    instructions = (
        "Use the declared tools to solve the task. Do not assume hidden values; retrieve them using tools. "
        "Keep track of your state in variables as you proceed."
    )
    output_format = "Return a JSON object with your final answer under key `answer`."
    return NLWrapper(
        title=title, instructions=instructions, output_format=output_format
    ).model_dump(mode="json")


async def generate_nl_wrapper(
    task: TaskData, llm: Optional[LlmProxy]
) -> Dict[str, Any]:
    """Generate a concise natural-language wrapper via LLM.

    If LLM is None, returns a deterministic fallback wrapper.
    """
    if llm is None:
        return _fallback_wrapper(task)

    system = (
        "You are a dataset writer. Write a short task instruction for a tool-grounded agent. "
        "The agent only sees the tool contract and the PUBLIC instance. "
        "Do not reveal PRIVATE state or the reference answer. "
        "Output MUST be strict JSON with keys: title, instructions, output_format."
    )

    user = {
        "family": task.family,
        "public": task.public,  # TODO: get tools here
    }

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Here is the public instance as JSON:\n"
            + json.dumps(user, ensure_ascii=False, indent=2),
        },
    ]

    res = await llm.complete(messages=messages, temperature=0.3, max_tokens=350)
    parsed = _try_parse_json(res.text)
    if parsed is None:
        return _fallback_wrapper(task)

    try:
        wrapper = NLWrapper.model_validate(parsed)
        return wrapper.model_dump(mode="json")
    except ValidationError:
        return _fallback_wrapper(task)
