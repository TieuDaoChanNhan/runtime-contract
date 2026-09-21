import asyncio

import pytest
import pytest_asyncio

from codeact_runtime.codeact.interpreter import AsyncInterpreter

# -------------------------------------------------------------------------
# Fixtures (Updated to use pytest_asyncio.fixture)
# -------------------------------------------------------------------------


@pytest_asyncio.fixture
async def interpreter():
    """
    Returns a fresh interpreter with persistence ENABLED.
    Using @pytest_asyncio.fixture ensures the event loop is properly
    handled before the fixture runs.
    """
    return AsyncInterpreter(persistent_state=True)


@pytest_asyncio.fixture
async def stateless_interpreter():
    """Returns a fresh interpreter with persistence DISABLED."""
    return AsyncInterpreter(persistent_state=False)


# -------------------------------------------------------------------------
# 1. Basic Execution Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_execution(interpreter):
    code = "x = 10 + 5\nprint(x)"
    result = await interpreter.run_code(code)

    assert result["success"] is True
    assert result["error"] is None
    assert "15" in result["output"]
    # Check that x persisted in globals
    assert result["globals"]["x"] == "15"


@pytest.mark.asyncio
async def test_stateless_execution(stateless_interpreter):
    # Run first execution
    await stateless_interpreter.run_code("x = 42")

    # Run second execution - x should NOT exist because persistence is False
    result = await stateless_interpreter.run_code("print(x)")

    assert result["success"] is False
    assert "NameError" in result["error"]


@pytest.mark.asyncio
async def test_stateless_globals_are_reported_per_step(stateless_interpreter):
    """Reset mode should still report the step's defined globals for observability."""
    result = await stateless_interpreter.run_code("x = 42\ny = x + 1\nprint(y)")

    assert result["success"] is True
    assert result["globals"]
    assert result["globals"]["x"] == "42"
    assert result["globals"]["y"] == "43"


@pytest.mark.asyncio
async def test_complex_logic(interpreter):
    code = """
def factorial(n):
    if n == 0: return 1
    return n * factorial(n-1)

print(f"Factorial 5 is {factorial(5)}")
"""
    result = await interpreter.run_code(code)
    assert result["success"] is True
    assert "Factorial 5 is 120" in result["output"]


# -------------------------------------------------------------------------
# 2. Async Tool Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_tool_registration_and_usage(interpreter):
    # Define a mock tool
    async def fetch_data(url):
        await asyncio.sleep(0.01)
        return f"Data from {url}"

    interpreter.register_tool("fetch", fetch_data)

    code = """
result = fetch("http://example.com")
print(result)
"""
    result = await interpreter.run_code(code)

    assert result["success"] is True
    assert "Data from http://example.com" in result["output"].strip()


@pytest.mark.asyncio
async def test_tool_error_propagation(interpreter):
    # Tool that raises an exception
    async def broken_tool():
        raise ValueError("Tool crashed!")

    interpreter.register_tool("crash", broken_tool)

    code = "crash()"
    result = await interpreter.run_code(code)

    assert result["success"] is False
    assert "ValueError: Tool crashed!" in result["error"]


@pytest.mark.asyncio
async def test_multiple_tool_calls(interpreter):
    # Counter to verify async execution
    async def increment(x):
        await asyncio.sleep(0.01)
        return x + 1

    interpreter.register_tool("inc", increment)

    code = """
total = 0
for i in range(5):
    total = inc(total)
print(total)
"""
    result = await interpreter.run_code(code)
    assert result["success"] is True
    assert result["output"].strip() == "5"


# -------------------------------------------------------------------------
# 3. Persistence & Visibility Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_across_runs(interpreter):
    # Step 1: Define variable
    await interpreter.run_code("my_var = 'Hello World'")

    # Step 2: Use variable
    result = await interpreter.run_code("print(my_var)")

    assert result["success"] is True
    assert "Hello World" in result["output"]


@pytest.mark.asyncio
async def test_function_persistence_visibility(interpreter):
    """
    Ensures that user-defined functions are visible in the globals list.
    """
    code = """
def my_helper():
    return "I help!"
"""
    result = await interpreter.run_code(code)

    # Check that function appears in globals
    assert "my_helper" in result["globals"]
    # Verify the repr string looks correct
    assert "function my_helper" in result["globals"]["my_helper"]


# -------------------------------------------------------------------------
# 4. Self-Healing & Safety Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_healing_tools(interpreter):
    """
    Test that if a user overwrites a registered tool, it is restored in the next run.
    """

    # 1. Register tool
    async def safe_tool():
        return "I am safe"

    interpreter.register_tool("safe_tool", safe_tool)

    # 2. Overwrite it
    code_bad = "safe_tool = 'I destroyed the tool'"
    await interpreter.run_code(code_bad)

    # 3. Run new code to verify restoration
    code_check = "print(safe_tool())"
    result = await interpreter.run_code(code_check)

    assert result["success"] is True
    assert "I am safe" in result["output"]


@pytest.mark.asyncio
async def test_builtin_protection(interpreter):
    """
    Test that standard builtins cannot be permanently overwritten.
    """
    # 1. Overwrite 'list'
    code_bad = "list = 'I broke list'"
    await interpreter.run_code(code_bad)

    # 2. Try to use 'list' again
    code_check = "x = list((1, 2)); print(x)"
    result = await interpreter.run_code(code_check)

    assert result["success"] is True
    assert "[1, 2]" in result["output"]


# -------------------------------------------------------------------------
# 5. Error Handling Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_syntax_error(interpreter):
    code = "print('Hello"  # Missing quote
    result = await interpreter.run_code(code)

    assert result["success"] is False
    assert "SyntaxError" in result["error"]


@pytest.mark.asyncio
async def test_runtime_error_clean_traceback(interpreter):
    code = """
def fail():
    raise ValueError("Deep failure")

fail()
"""
    result = await interpreter.run_code(code)

    assert result["success"] is False
    # Check that traceback filtering works (hiding internal interpreter files)
    assert 'File "<string>"' in result["error"]
    assert "ValueError: Deep failure" in result["error"]
    # Ideally, it should NOT show "concurrent/futures" lines
    assert "concurrent.futures" not in result["error"]


# -------------------------------------------------------------------------
# 6. Concurrency & Blocking Tests
# -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocking_behavior(interpreter):
    """
    Verify that long-running sync code doesn't freeze the async tool loop.
    """

    async def fast_tool():
        return "fast"

    interpreter.register_tool("fast_tool", fast_tool)

    # This code sleeps synchronously in the thread
    code = """
import time
time.sleep(0.1) 
print(fast_tool())
"""
    result = await interpreter.run_code(code)
    assert result["success"] is True
    assert "fast" in result["output"]


@pytest.mark.asyncio
async def test_async_interpreter_limits_tool_calls_per_run():
    """
    Verify that the interpreter enforces tool limits and returns partial output
    WITHOUT raising an exception to the caller.
    """
    interpreter = AsyncInterpreter(max_tool_calls=2)

    async def echo(value: str) -> str:
        return value

    interpreter.register_tool("echo", echo)

    code = """
print("Start")
print(echo("First Call"))   # Call 1 (OK)
print(echo("Second Call"))  # Call 2 (OK)
print(echo("Third Call"))   # Call 3 (Should Fail)
print("End")
"""

    # We do NOT expect an exception. We expect a result dict with success=False.
    result = await interpreter.run_code(code)

    # 1. Verify failure status
    assert result["success"] is False

    # 2. Check the error message
    assert result["error"] is not None
    assert "Tool call limit exceeded" in result["error"]

    # 3. Check Partial Output (stdout captured before the limit was hit)
    output = result["output"]
    assert "Start" in output
    assert "First Call" in output
    assert "Second Call" in output

    # 4. Ensure execution stopped immediately
    assert "Third Call" not in output
    assert "End" not in output
