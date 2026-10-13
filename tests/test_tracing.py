"""Tests for the trace store and collector."""

from types import SimpleNamespace

from ares.runtime.events import EventBus, EventType
from ares.runtime.tracing import TraceCollector, TraceStore
from ares.runtime.types import AgentResult, StepType, Trace, TraceStep


def test_store_round_trip_and_feedback(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    trace = Trace(query="q", agent="coder", result="r", steps=[
        TraceStep(step_type=StepType.TOOL_CALL, timestamp=1.0, input={"tool": "t", "arguments": {"x": object()}})])
    store.save(trace)
    assert store.update_feedback(trace.trace_id, 1.0)
    loaded = store.get(trace.trace_id)
    assert loaded.query == "q" and loaded.feedback == 1.0
    assert loaded.steps[0].step_type == StepType.TOOL_CALL and loaded.steps[0].input["tool"] == "t"
    assert store.get("missing") is None


def test_collector_records_inference_and_tools(tmp_path):
    bus = EventBus()

    class Agent:
        agent_id = "orchestrator"

        def run(self, prompt):
            bus.publish(EventType.INFERENCE_START, {"model": "m", "engine": "ollama"})
            bus.publish(EventType.INFERENCE_END, {"usage": {"total_tokens": 7}, "content": "c"})
            bus.publish(EventType.TOOL_CALL_START, {"tool": "call_coder", "arguments": {"task": "x"}})
            bus.publish(EventType.TOOL_CALL_END, {"tool": "call_coder", "success": True, "latency": 0.5, "result": "done"})
            return AgentResult(content="answer", turns=2)

    done = []
    bus.subscribe(EventType.TRACE_COMPLETE, lambda e: done.append(e.data["trace"]))
    store = TraceStore(tmp_path / "t.db")
    collector = TraceCollector(Agent(), store=store, bus=bus)
    assert collector.run("hello").content == "answer"

    trace = collector.last_trace
    assert done == [trace] and store.get(trace.trace_id) is not None
    assert [s.step_type for s in trace.steps] == [StepType.GENERATE, StepType.TOOL_CALL, StepType.RESPOND]
    assert trace.model == "m" and trace.total_tokens == 7
    assert trace.steps[1].input == {"tool": "call_coder", "arguments": {"task": "x"}, "agent": ""}


def test_collector_attributes_handoff_steps():
    """A sub-agent's steps inside a handoff keep their own arguments and model."""
    bus = EventBus()

    class Agent:
        agent_id = "orchestrator"

        def run(self, prompt):
            bus.publish(EventType.INFERENCE_START, {"agent": "orchestrator", "model": "router"})
            bus.publish(EventType.INFERENCE_END, {"agent": "orchestrator", "model": "router", "usage": {}})
            bus.publish(EventType.TOOL_CALL_START, {"agent": "orchestrator", "call_key": "outer",
                                                    "tool": "call_coder", "arguments": {"task": "count"}})
            bus.publish(EventType.INFERENCE_START, {"agent": "coder", "model": "coder-7b"})
            bus.publish(EventType.INFERENCE_END, {"agent": "coder", "model": "coder-7b", "usage": {}})
            bus.publish(EventType.TOOL_CALL_START, {"agent": "coder", "call_key": "inner",
                                                    "tool": "file_read", "arguments": {"path": "x"}})
            bus.publish(EventType.TOOL_CALL_END, {"agent": "coder", "call_key": "inner", "tool": "file_read",
                                                  "success": True, "latency": 3.0, "result": "r"})
            bus.publish(EventType.TOOL_CALL_END, {"agent": "orchestrator", "call_key": "outer",
                                                  "tool": "call_coder", "success": True, "latency": 5.0,
                                                  "result": "2"})
            return AgentResult(content="2", turns=2)

    collector = TraceCollector(Agent(), bus=bus)
    collector.run("q")
    trace = collector.last_trace
    tools = {s.input["tool"]: s.input for s in trace.steps if s.step_type == StepType.TOOL_CALL}
    assert tools["call_coder"]["arguments"] == {"task": "count"}
    assert tools["file_read"]["arguments"] == {"path": "x"}
    assert [s.input["model"] for s in trace.steps if s.step_type == StepType.GENERATE] == ["router", "coder-7b"]
    assert trace.model == "router"
    assert trace.total_latency_seconds < 1  # wall time, not the 8s sum of nested steps


def test_collector_stops_listening_after_the_run():
    bus = EventBus()
    agent = SimpleNamespace(agent_id="a", run=lambda p: AgentResult(content="x"))
    TraceCollector(agent, bus=bus).run("q")
    assert all(not subs for subs in bus._subscribers.values())
