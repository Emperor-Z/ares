"""Tests for the agent loops, driven by a scripted fake engine."""

import json

import pytest

from ares.runtime.agents import ActionAgent, ReActAgent, ToolCallingAgent, parse_action, parse_react, strip_think
from ares.runtime.events import EventBus, EventType
from ares.runtime.loop_guard import LoopGuardConfig
from ares.runtime.tools import BaseTool, ToolSpec
from ares.runtime.types import Role, ToolResult


class _Engine:
    engine_id = "fake"
    num_ctx = 16384

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        reply = self.replies.pop(0)
        return reply if isinstance(reply, dict) else {"content": reply, "usage": {"total_tokens": 1}}


class _Add(BaseTool):
    tool_id = "add"

    def __init__(self):
        self.calls = []

    @property
    def spec(self):
        return ToolSpec(name="add", description="Add a and b.",
                        parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}})

    def execute(self, **p):
        self.calls.append(p)
        return ToolResult(tool_name="add", content=str(p["a"] + p["b"]))


# ── ReAct ─────────────────────────────────────────────────────────────────────

def test_react_calls_tool_then_answers():
    engine = _Engine('Thought: add\nAction: add\nAction Input: {"a": 2, "b": 3}', "Thought: done\nFinal Answer: 5")
    tool = _Add()
    result = ReActAgent(engine, "m", tools=[tool], agent_id="coder").run("2+3?")
    assert result.content == "5" and result.turns == 2
    assert tool.calls == [{"a": 2, "b": 3}]
    last_prompt = engine.calls[1][0]
    assert last_prompt[-1].content == "Observation: 5"
    assert "### add" in last_prompt[0].content


def test_react_plain_reply_is_the_answer():
    assert ReActAgent(_Engine("just an answer"), "m").run("hi").content == "just an answer"


def test_react_max_turns():
    engine = _Engine(*['Action: add\nAction Input: {"a": 1, "b": %d}' % i for i in range(3)])
    result = ReActAgent(engine, "m", tools=[_Add()], max_turns=3).run("loop")
    assert result.metadata["max_turns_exceeded"] and result.turns == 3


def test_loop_guard_is_reset_between_runs():
    # Before, counts carried across runs, so a tool got blocked for the rest of the session.
    guard = LoopGuardConfig(poll_tool_budget=1, warn_before_block=False)
    tool = _Add()
    step = 'Action: add\nAction Input: {"a": 1, "b": 1}'
    agent = ReActAgent(_Engine(step, "Final Answer: 2", step, "Final Answer: 2"), "m", tools=[tool], loop_guard=guard)
    agent.run("first")
    agent.run("second")
    assert len(tool.calls) == 2



def test_loop_guard_warning_reaches_the_model():
    # Before, the warned call ran silently, so the warning was just a free extra call.
    guard = LoopGuardConfig(max_identical_calls=1)
    step = 'Action: add\nAction Input: {"a": 1, "b": 1}'
    engine = _Engine(step, step, step, "Final Answer: 2")
    tool = _Add()
    ReActAgent(engine, "m", tools=[tool], loop_guard=guard).run("q")
    observations = [m.content for m in engine.calls[-1][0] if m.content.startswith("Observation:")]
    assert len(tool.calls) == 2  # the third identical call is blocked
    assert "Loop guard warning" in observations[1] and "Loop guard:" in observations[2]

def test_react_parses_text_and_json():
    assert parse_react("Thought: x\nFinal Answer: y")["final_answer"] == "y"
    parsed = parse_react('{"thought": "t", "action": "add", "action_input": {"a": 1}}')
    assert parsed["action"] == "add" and json.loads(parsed["action_input"]) == {"a": 1}
    assert "parse_error" in parse_react('{"action": "add", "final_answer": "both"}')
    assert parse_react('{"answer": 42}')["action"] == ""  # plain JSON is an answer


# ── Action (thinker) ─────────────────────────────────────────────────────────

def test_action_agent_uses_text_tools_and_strips_think():
    engine = _Engine('<think>need a tool</think>Action: add\nAction Input: {"a": 1, "b": 1}',
                     "<think>ok</think>The answer is 2.")
    result = ActionAgent(engine, "m", tools=[_Add()]).run("1+1?")
    assert result.content == "The answer is 2."
    assert all("tools" not in kwargs for _, kwargs in engine.calls)  # never sent as function schemas


def test_action_agent_answers_with_code_are_answers():
    # A python block in the answer used to be sent to a code tool that doesn't exist.
    reply = "Here you go:\n```python\nprint('hi')\n```"
    assert ActionAgent(_Engine(reply), "m").run("show code").content == reply


def test_parse_action_needs_its_own_line():
    assert parse_action("I would take Action: nothing here") is None
    assert parse_action('Action: add\nAction Input: {"a": 1}') == ("add", '{"a": 1}')


def test_action_agent_trims_oversized_input():
    engine = _Engine("ok")
    engine.num_ctx = 1000
    ActionAgent(engine, "m", max_tokens=200).run("x" * 20000)
    sent = engine.calls[0][0][-1].content
    assert len(sent) < 20000 and sent.endswith("[Input truncated to fit context window]")


# ── native tool calling (orchestrator) ───────────────────────────────────────

def _tool_reply(*calls):
    return {"content": "", "tool_calls": [{"id": f"c{i}", "name": n, "arguments": json.dumps(a)} for i, (n, a) in enumerate(calls)]}


def test_tool_calling_runs_calls_in_order():
    engine = _Engine(_tool_reply(("add", {"a": 1, "b": 2}), ("add", {"a": 3, "b": 4})), "3 and 7")
    agent = ToolCallingAgent(engine, "m", tools=[_Add()], system_prompt="route")
    result = agent.run("go")
    assert result.content == "3 and 7"
    messages, kwargs = engine.calls[1]
    assert messages[0].role == Role.SYSTEM and messages[0].content == "route"
    assert [m.content for m in messages if m.role == Role.TOOL] == ["3", "7"]
    assert kwargs["tools"][0]["function"]["name"] == "add"


def test_tool_calling_continues_cut_off_replies():
    engine = _Engine({"content": "first half ", "finish_reason": "length"}, {"content": "second half"})
    assert ToolCallingAgent(engine, "m").run("long").content == "first half second half"


def test_system_prompt_can_be_swapped_between_runs():
    # learning.apply() rewrites the orchestrator's prompt in place.
    engine = _Engine("a", "b")
    agent = ToolCallingAgent(engine, "m", system_prompt="old")
    agent.run("x")
    agent._system_prompt = "new"
    agent.run("y")
    assert engine.calls[1][0][0].content == "new"


def test_turn_and_inference_events():
    bus = EventBus()
    seen = []
    for t in (EventType.AGENT_TURN_START, EventType.INFERENCE_START, EventType.INFERENCE_END, EventType.AGENT_TURN_END):
        bus.subscribe(t, lambda e, t=t: seen.append((t.value, e.data.get("agent"))))
    ReActAgent(_Engine("Final Answer: x"), "m", bus=bus, agent_id="runner").run("q")
    assert seen == [("agent_turn_start", "runner"), ("inference_start", "runner"),
                    ("inference_end", "runner"), ("agent_turn_end", "runner")]


def test_engine_errors_propagate():
    class Broken(_Engine):
        def generate(self, messages, **kwargs):
            raise RuntimeError("ollama down")
    with pytest.raises(RuntimeError, match="ollama down"):
        ReActAgent(Broken(), "m").run("q")


def test_strip_think_handles_missing_open_tag():
    assert strip_think("reasoning</think>\nanswer") == "answer"
