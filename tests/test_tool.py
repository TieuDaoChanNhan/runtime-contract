import asyncio

import pytest

from codeact_runtime.codeact.tool import (
    Tool,
    ToolInvocationException,
    ToolRuntimeException,
)


class AddTool(Tool):
    name = "add"
    doc = "Add a value to a base."
    arg_doc = {"value": "Value to add.", "base": "Base value."}

    async def run(self, value: int, base: int = 1) -> int:
        return value + base


class ExplodingTool(Tool):
    name = "explode"
    doc = "Raise an error."
    arg_doc = {"value": "Value to explode."}

    async def run(self, value: int) -> int:
        raise RuntimeError(f"boom {value}")


def test_tool_inspects_inputs_and_formats_doc():
    tool = AddTool()

    assert list(tool.inputs.keys()) == ["value", "base"]
    assert tool.inputs["value"].is_required is True
    assert tool.inputs["base"].is_required is False
    assert tool.inputs["base"].default_value == 1

    doc = tool.full_python_doc()
    assert "def add(value: int, base: int = 1) -> int" in doc
    assert "Add a value to a base." in doc


def test_tool_validates_types_and_runs():
    tool = AddTool()

    result = asyncio.run(tool(2, base=3))
    assert result == 5

    with pytest.raises(ToolInvocationException):
        asyncio.run(tool("nope"))


def test_tool_wraps_runtime_errors():
    tool = ExplodingTool()

    with pytest.raises(ToolRuntimeException) as exc:
        asyncio.run(tool(4))

    assert "boom 4" in str(exc.value)
