"""Tests for conversation history formatting in system.py."""

# Import the private formatter directly without building a full AresSystem
from ares.system import _format_history


def test_no_history_returns_prompt_unchanged():
    result = _format_history([], "what is 2+2?")
    assert result == "what is 2+2?"


def test_history_prepended():
    history = [
        {"role": "user", "content": "explain ares"},
        {"role": "assistant", "content": "Ares is a multi-agent system."},
    ]
    result = _format_history(history, "what about the coder?")
    assert "explain ares" in result
    assert "Ares is a multi-agent system." in result
    assert "what about the coder?" in result


def test_history_window_capped_at_10():
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(20)
    ]
    result = _format_history(history, "current prompt")
    # Only last 10 messages should appear
    assert "msg 0" not in result
    assert "msg 10" in result


def test_role_labels():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    result = _format_history(history, "next")
    assert "User: hello" in result
    assert "Assistant: hi there" in result
