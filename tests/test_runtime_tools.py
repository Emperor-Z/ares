"""Tests for the tool executor: dispatch, approval, timeouts, events."""

import json
import threading
import time

import pytest

from ares.runtime import tools
from ares.runtime.events import EventBus, EventType
from ares.runtime.tools import BaseTool, ToolExecutor, ToolSpec, describe_tools
from ares.runtime.types import ToolCall, ToolResult


class _Echo(BaseTool):
    tool_id = "echo"

    def __init__(self, confirm=False, timeout=5.0, delay=0.0):
        self._confirm, self._timeout, self._delay = confirm, timeout, delay
        self.ran_on = None

    @property
    def spec(self):
        return ToolSpec(name="echo", description="Echo text.", timeout_seconds=self._timeout,
                        requires_confirmation=self._confirm,
                        parameters={"type": "object", "properties": {"text": {"type": "string", "description": "What to say."}},
                                    "required": ["text"]})

    def execute(self, **params):
        self.ran_on = threading.current_thread().name
        time.sleep(self._delay)
        return ToolResult(tool_name="echo", content=params.get("text", ""))


def _call(args='{"text": "hi"}'):
    return ToolCall(id="1", name="echo", arguments=args)


@pytest.fixture(autouse=True)
def _no_approver(monkeypatch):
    monkeypatch.delenv("ARES_SHELL_AUTO", raising=False)
    tools.set_confirm(None)
    yield
    tools.set_confirm(None)


def test_runs_tool_off_the_calling_thread():
    tool = _Echo()
    result = ToolExecutor([tool]).execute(_call())
    assert result.success and result.content == "hi"
    assert tool.ran_on.startswith("ares-tool-")


def test_unknown_tool_lists_the_real_ones():
    result = ToolExecutor([_Echo()]).execute(ToolCall(id="1", name="nope", arguments="{}"))
    assert not result.success and "echo" in result.content


@pytest.mark.parametrize("args", ["not json", "[1, 2]"])
def test_bad_arguments_fail_cleanly(args):
    assert not ToolExecutor([_Echo()]).execute(_call(args)).success


def test_timeout():
    result = ToolExecutor([_Echo(timeout=0.05, delay=0.5)]).execute(_call())
    assert not result.success and "timed out" in result.content


def test_no_timeout_waits():
    tool = _Echo(timeout=None, delay=0.1)
    assert ToolExecutor([tool]).execute(_call()).success


def test_events_carry_tool_and_agent():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.TOOL_CALL_START, lambda e: seen.append(("start", e.data["tool"], e.data["agent"])))
    bus.subscribe(EventType.TOOL_CALL_END, lambda e: seen.append(("end", e.data["tool"], e.data["success"])))
    ToolExecutor([_Echo()], bus=bus, agent_id="coder").execute(_call())
    assert seen == [("start", "echo", "coder"), ("end", "echo", True)]


def test_needs_approval_refused_without_approver():
    tool = _Echo(confirm=True)
    result = ToolExecutor([tool]).execute(_call())
    assert not result.success and "ARES_SHELL_AUTO" in result.content
    assert tool.ran_on is None


def test_approver_sees_the_call_and_can_decline():
    asked = []
    tools.set_confirm(lambda prompt: asked.append(prompt) or False)
    tool = _Echo(confirm=True)
    result = ToolExecutor([tool], agent_id="runner").execute(_call())
    assert not result.success and "declined" in result.content
    assert tool.ran_on is None
    assert asked and asked[0].startswith("runner wants to run: echo")


def test_approver_can_allow():
    tools.set_confirm(lambda prompt: True)
    assert ToolExecutor([_Echo(confirm=True)]).execute(_call()).success


def test_auto_approve_env(monkeypatch):
    monkeypatch.setenv("ARES_SHELL_AUTO", "1")
    assert ToolExecutor([_Echo(confirm=True)]).execute(_call()).success


def test_describe_tools_lists_parameters():
    text = describe_tools([_Echo()])
    assert "### echo" in text and "text (string, required): What to say." in text
    assert describe_tools([]) == "No tools available."


def test_loads_lenient_reads_stray_backslashes_literally():
    from ares.runtime.tools import loads_lenient
    assert loads_lenient(r'{"command": "find . -name \"*\.py\""}') == {"command": r'find . -name "*\.py"'}
    # Valid escapes next to a stray one are left alone.
    assert loads_lenient(r'{"c": "a\\b \.c \n"}') == {"c": "a\\b \\.c \n"}
    with pytest.raises(json.JSONDecodeError):
        loads_lenient('{"c": ')


def test_bare_value_fills_the_one_required_parameter():
    # deepseek-r1 writes Action Input: "3847 * 2219" for single-argument tools.
    result = ToolExecutor([_Echo()]).execute(_call('"hi"'))
    assert result.success and result.content == "hi"


def test_rejected_calls_are_still_published():
    bus = EventBus()
    ends = []
    bus.subscribe(EventType.TOOL_CALL_END, lambda e: ends.append(e.data))
    ToolExecutor([_Echo()], bus=bus, agent_id="thinker").execute(_call("[1, 2]"))
    assert len(ends) == 1 and not ends[0]["success"] and ends[0]["agent"] == "thinker"


def test_missing_required_argument_says_how_to_call():
    tool = _Echo()
    result = ToolExecutor([tool]).execute(_call("{}"))
    assert not result.success and tool.ran_on is None
    assert 'Missing required argument(s): text' in result.content and '{"text": "..."}' in result.content
