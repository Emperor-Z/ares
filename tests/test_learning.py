"""Tests for learning routing from /good and /bad ratings."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ares import learning
from ares.system import AresSystem, _handoffs


def _step(tool: str, step_type: str = "tool_call"):
    return SimpleNamespace(step_type=step_type, input={"tool": tool})


def test_record_and_load_round_trip(tmp_path):
    path = tmp_path / "feedback.jsonl"
    learning.record("write fib", ["coder"], 1.0, "t1", path=path)
    learning.record("why is the sky blue", [], 0.0, None, path=path)
    path.write_text(path.read_text() + "not json\n")
    ratings = learning.load(path)
    assert [r["prompt"] for r in ratings] == ["write fib", "why is the sky blue"]


def test_latest_rating_per_prompt_wins():
    ratings = [
        {"prompt": "explain  REST", "agents": ["runner"], "score": 1.0},
        {"prompt": "explain REST", "agents": ["runner"], "score": 0.0},
        {"prompt": "write fib", "agents": ["coder"], "score": 1.0},
    ]
    good, bad = learning.routing_examples(ratings)
    assert good == [("write fib", "coder")]
    assert bad == [("explain REST", "runner")]


def test_examples_are_capped_newest_first():
    ratings = [{"prompt": f"task {i}", "agents": ["coder"], "score": 1.0} for i in range(20)]
    good, _ = learning.routing_examples(ratings)
    assert len(good) == learning.MAX_EXAMPLES
    assert good[0][0] == "task 19"


def test_routing_prompt_lists_lessons():
    base = "You are Ares."
    assert learning.routing_prompt(base, []) == base
    ratings = [
        {"prompt": "write fib", "agents": ["coder"], "score": 1.0},
        {"prompt": "hi", "agents": [], "score": 0.0},
    ]
    out = learning.routing_prompt(base, ratings)
    assert out.startswith(base)
    assert '"write fib" -> call_coder' in out
    assert '"hi" -> answered directly' in out


def test_apply_sets_orchestrator_prompt(tmp_path):
    path = tmp_path / "feedback.jsonl"
    learning.record("write fib", ["coder"], 1.0, "t1", path=path)
    orch = SimpleNamespace(_system_prompt="old")
    summary = learning.apply(orch, "base", path=path)
    assert "call_coder" in orch._system_prompt
    assert summary == {"ratings": 1, "good_examples": 1, "bad_examples": 0}


def test_handoffs_come_from_call_tools_only():
    trace = SimpleNamespace(steps=[
        _step("file_read"),
        _step("call_coder"),
        _step("call_coder"),
        _step("call_thinker"),
        _step("call_runner", step_type="generate"),
    ])
    assert _handoffs(trace) == ["coder", "thinker"]
    assert _handoffs(None) == []


def test_rate_records_the_last_reply(monkeypatch, tmp_path):
    path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(learning, "FEEDBACK_FILE", path)
    store = MagicMock()
    monkeypatch.setattr("ares.system.get_trace_store", lambda: store)

    system = AresSystem.__new__(AresSystem)
    system.use_memory = False
    system._router = SimpleNamespace(_system_prompt="")
    system._last = None
    trace = SimpleNamespace(trace_id="t9", steps=[_step("call_thinker")])
    system.orchestrator = MagicMock(last_trace=trace)
    system.orchestrator.run.return_value = SimpleNamespace(content="answer")

    assert system.rate(1.0) is False
    system.run("compare REST and gRPC")
    assert system.rate(0.0) is True

    store.update_feedback.assert_called_once_with("t9", 0.0)
    assert learning.load(path)[0]["agents"] == ["thinker"]
    assert '"compare REST and gRPC" -> call_thinker' in system._router._system_prompt
