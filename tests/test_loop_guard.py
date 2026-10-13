"""Tests for the loop guard."""

from ares.runtime.loop_guard import LoopGuard, LoopGuardConfig
from ares.runtime.types import Message, Role


def _guard(**kw):
    return LoopGuard(LoopGuardConfig(**kw))


def test_identical_calls_warn_once_then_block():
    g = _guard(max_identical_calls=2, poll_tool_budget=99)
    assert not g.check_call("t", "{}").blocked
    assert not g.check_call("t", "{}").blocked
    third = g.check_call("t", "{}")
    assert not third.blocked and third.warned
    assert g.check_call("t", "{}").blocked


def test_per_tool_budget():
    g = _guard(poll_tool_budget=2, warn_before_block=False, ping_pong_window=99)
    assert not g.check_call("t", "1").blocked
    assert not g.check_call("t", "2").blocked
    assert g.check_call("t", "3").blocked


def test_ping_pong():
    g = _guard(ping_pong_window=4, warn_before_block=False)
    verdicts = [g.check_call(name, str(i)).blocked for i, name in enumerate("abab")]
    assert verdicts == [False, False, False, True]


def test_reset_starts_a_fresh_run():
    g = _guard(poll_tool_budget=1, warn_before_block=False)
    g.check_call("t", "1")
    assert g.check_call("t", "2").blocked
    g.reset()
    assert not g.check_call("t", "3").blocked


def test_compress_keeps_system_and_recent():
    g = _guard(max_context_messages=4)
    msgs = [Message(role=Role.SYSTEM, content="sys")] + [Message(role=Role.USER, content=str(i)) for i in range(10)]
    out = g.compress_context(msgs)
    assert [m.content for m in out] == ["sys", "7", "8", "9"]


def test_compress_leaves_short_context_alone():
    msgs = [Message(role=Role.USER, content="hi")]
    assert _guard().compress_context(msgs) is msgs
